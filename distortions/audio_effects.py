# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
音频效果处理模块
===============

本模块提供了一套完整的音频增强和失真效果工具，用于音频数据的增强处理。
主要用于训练和测试音频水印系统的鲁棒性。

主要组件
--------

1. **AudioEffects 类**
   包含所有可用的音频效果方法，每个方法都是静态方法，可以独立调用。

   音频效果分类：
   - 时域效果：
     * speed: 改变音频播放速度
     * echo: 添加回声效果
     * smooth: 平滑滤波

   - 频域效果：
     * lowpass_filter: 低通滤波
     * highpass_filter: 高通滤波
     * bandpass_filter: 带通滤波

   - 噪声添加：
     * random_noise: 添加高斯白噪声
     * pink_noise: 添加粉红噪声

   - 采样与量化：
     * updownresample: 上下采样（模拟重采样失真）
     * quantize_audio: 音频量化

   - 音量调整：
     * boost_audio: 增强音量
     * duck_audio: 降低音量

   - 压缩编码：
     * mp3_compression: MP3 有损压缩
     * aac_compression: AAC 有损压缩
     * compress_with_encodec: 使用 Encodec 模型压缩

   - 工具方法：
     * identity: 恒等变换（不做任何处理）

2. **辅助函数**
   - select_audio_effects: 根据模式和权重选择音频效果子集
   - get_audio_effects: 从配置中自动提取所有可用的音频效果
   - audio_effect_return: 统一的返回值处理函数
   - generate_pink_noise: 使用 Voss-McCartney 算法生成粉红噪声
   - apply_compression_skip_grad: 应用可微分的压缩函数（使用直通估计器）

技术特点
--------
- 所有效果都支持批处理（batch processing）
- 支持可选的掩码（mask）参数用于选择性处理
- 压缩类效果使用直通估计器（Straight-Through Estimator）保持可微分性
- 自动处理 NaN 检测和数值稳定性
- 支持 GPU 加速处理

使用示例
--------
```python
# 获取所有可用的音频效果
effects = get_audio_effects(cfg)

# 选择特定的效果子集
selected_effects = select_audio_effects(
    effects,
    mode="weighted",
    weights={"random_noise": 0.5, "lowpass_filter": 0.3}
)

# 应用单个效果
noisy_audio = AudioEffects.random_noise(audio_tensor, noise_std=0.01)

# 应用带掩码的效果
filtered_audio, mask = AudioEffects.lowpass_filter(
    audio_tensor,
    cutoff_freq=5000,
    mask=input_mask
)
```

注意事项
--------
- 输入张量格式：(batch_size, channels, time_steps)
- 所有频率参数单位为 Hz
- 压缩效果会改变音频质量，用于测试鲁棒性
- 某些效果（如 speed）会改变音频长度
"""

import inspect
import random
import typing as tp
from functools import partial

import julius
import omegaconf
import torch
import torchaudio
from julius import fft_conv1d, resample_frac

from .audio_utils import get_aac, get_mp3, get_opus


# if tp.TYPE_CHECKING:
#     from ..models.encodec import CompressionModel


# ============================================================================
# 音频效果选择与管理函数
# ============================================================================

def select_audio_effects(
    audio_effects: tp.Dict,
    weights: tp.Optional[tp.Dict] = None,
    mode: str = "all",
    max_length: tp.Optional[int] = None,
):
    """从 `AudioEffects` 类中采样音频效果方法的子集。

    此函数允许你基于选择的模式和可选权重来选择音频效果的子集。

    Args:
        audio_effects (dict): 可用音频增强的字典，通常从 'get_audio_effects'
            函数的输出中获得。
        weights (dict): 将增强名称映射到其对应选择概率的字典。当 'mode' 设置为
            "weighted" 时使用此参数。如果 'weights' 为 None，则所有增强具有相等的
            选择概率。
        mode (str): 选择模式，可以是以下之一：
            - "all": 选择所有可用的增强。
            - "weighted": 基于 'weights' 字典中的概率选择增强。
        max_length (int): 要选择的增强的最大数量。如果 'max_length' 为 None，
            则不应用限制。

    Returns:
        dict: 包含所选音频增强的 'audio_effects' 字典的子集。

    Note:
        - 在 "all" 模式下，选择所有可用的增强。
        - 在 "weighted" 模式下，增强的选择概率与 'weights' 字典中指定的权重成正比。
        - 如果设置了 'max_length'，函数会限制所选增强的数量。
        - 如果没有选择任何增强或 'audio_effects' 为空，函数默认包含一个 "identity" 增强。
        - "identity" 增强表示不应用任何音频效果。
    """

    # 根据不同的模式选择音频效果
    if mode == "all":  # 原始代码：选择所有效果
        out = audio_effects
    elif mode=='none':  # 不选择任何效果
        out = {}
    elif mode == "weighted":  # 加权随机选择
        # 概率与权重成正比
        assert weights is not None
        out = {
            name: value
            for name, value in audio_effects.items()
            if random.random() < weights.get(name, 1.0)  # 按权重概率选择
        }
    else:
        raise ValueError(f"Unknown mode {mode}")

    # 限制选择的效果数量（防止 GPU 内存溢出）
    if max_length is not None:
        # 有助于设定GPU内存使用的确定性限制
        if len(list(out.keys()))>=max_length:
            random_keys = random.sample(list(out.keys()), max_length)
            out = {key: out[key] for key in random_keys}

    # 如果没有选择任何效果，默认使用恒等变换
    if len(out) == 0:  # 检查不返回空字典
        out = {"identity": AudioEffects.identity}
    return out

def get_audio_effects(cfg: omegaconf.DictConfig):
    """根据配置参数自动提取此类中所有可用效果的列表

    Returns:
        dict: 包含此类中所有方法的名称和指针的字典。
    """

    # 确保配置中包含 audio_effects 字段
    assert hasattr(cfg, "audio_effects")
    cfg_audio_effects = dict(cfg["audio_effects"])

    # 通过反射获取 AudioEffects 类中的所有函数
    # 并使用配置中的参数创建偏函数（partial function）
    return {
        name: partial(value, **cfg_audio_effects.get(name, {}))
        for name, value in inspect.getmembers(AudioEffects)
        if inspect.isfunction(value)
    }

def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """如果输入中包含mask则返回mask，否则只返回输出张量

    这个函数还负责：
    1. 检测 NaN 值
    2. 将音频归一化到 [-1, 1] 范围内
    3. 统一返回格式
    """
    # 安全性检查：检测 NaN 值
    if torch.isnan(tensor).any():
        raise ValueError("NaN detected in tensor, terminating program.")

    # 数值稳定性：确保音频在有效范围内 [-1, 1]
    if tensor.abs().max() >= 1:
        tensor=tensor/(tensor.abs().max()+ 1e-6)

    # 根据是否有 mask 返回相应的格式
    if mask is None:
        return tensor
    else:
        return tensor, mask


def generate_pink_noise(length: int) -> torch.Tensor:
    """使用Voss-McCartney算法和PyTorch生成粉红噪声。

    粉红噪声（1/f 噪声）在每个倍频程内具有相等的能量，
    相比白噪声更接近自然界的声音特征。

    Args:
        length: 生成的噪声样本长度

    Returns:
        归一化的粉红噪声张量
    """
    # Voss-McCartney 算法参数
    num_rows = 16

    # 生成随机数组并累积求和（这是生成 1/f 噪声的关键）
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)  # 累积和产生 1/f 特性
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]

    # 归一化到 [-1, 1] 范围
    pink_noise = reshaped_array / (torch.max(torch.abs(reshaped_array))+1e-6)
    return pink_noise


def compress_with_encodec(
    tensor: torch.Tensor,
    n_q: int,
    model: "CompressionModel",
    sample_rate: int,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """使用带有n_q个码本的压缩模型来压缩和解压缩wav张量的特殊增强函数

    Encodec 是一个神经音频编解码器，使用向量量化进行压缩。
    码本数量(n_q)控制压缩质量：更多码本 = 更高质量 = 更大比特率

    处理流程：
    1. 重采样到模型采样率
    2. 编码（压缩）
    3. 解码（解压缩）
    4. 重采样回原始采样率
    """
    # 将模型移到与输入相同的设备
    model.to(tensor.device)

    # 设置使用的码本数量（控制压缩质量）
    model.set_num_codebooks(n_q)

    # 编码：重采样 -> 压缩
    codes, scale = model.encode(
        julius.resample_frac(tensor, old_sr=sample_rate, new_sr=model.sample_rate)
    )

    # 解码：解压缩
    compressed = model.decode(codes=codes, scale=scale)

    # 重采样回原始采样率并返回
    return audio_effect_return(
        tensor=julius.resample_frac(
            compressed, old_sr=model.sample_rate, new_sr=sample_rate
        ),
        mask=mask,
    )


def apply_compression_skip_grad(tensor: torch.Tensor, compression_fn, **kwargs):
    """对音频张量应用指定的压缩函数。
    通过直通估计器将梯度传递到输出张量
    这是一个直通估计器，使mp3/aac压缩可微分
    详见：Yin et al. 2019 https://arxiv.org/pdf/1903.05662.pdf

    Args:
        tensor (torch.Tensor): 输入音频张量。
        compression_fn (function): 要应用的压缩函数。
        **kwargs: 压缩函数的其他关键字参数。

    Returns:
        torch.Tensor: 应用压缩和直通估计器后的输出张量。
    """
    # 步骤1: 在分离的张量上应用压缩（阻断梯度）
    compressed = compression_fn(tensor.detach(), **kwargs)

    # 步骤2: 如果需要，裁剪压缩输出以匹配输入长度
    compressed = compressed[:, :, : tensor.size(-1)]

    # 步骤3: 使用直通估计器使压缩操作可微分
    # 前向传播：使用压缩后的值
    # 反向传播：梯度直接传递，就像压缩操作不存在一样
    out = tensor + (compressed - tensor).detach()

    # 步骤4: 安全性检查 - 确保梯度计算图没有被破坏
    if out.requires_grad:
        assert (
            out.grad_fn
        ), "The computation graph might be broken due to compression augmentation."

    return out


# ============================================================================
# AudioEffects 类 - 音频效果集合
# ============================================================================

class AudioEffects:
    """
    音频效果类 - 包含所有可用的音频增强和失真方法

    所有方法都是静态方法，可以直接通过类名调用。
    每个方法都支持批处理和可选的掩码参数。
    """

    # ------------------------------------------------------------------------
    # 时域效果
    # ------------------------------------------------------------------------

    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed_range: tuple = (0.5, 1.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """改变批量音频数据速度的函数。
        输出将具有不同的长度！

        Args:
            audio_batch (torch.Tensor): torch张量格式的音频数据批次。
            speed (float): 要改变的音频速度。

        Returns:
            torch.Tensor: 速度已改变的音频数据批次。
        """
        # 随机选择一个速度因子
        speed = torch.FloatTensor(1).uniform_(*speed_range)

        # 通过改变采样率来改变速度
        # 速度变快 -> 采样率降低；速度变慢 -> 采样率升高
        new_sr = int(sample_rate * 1 / speed)
        resampled_tensor = julius.resample.resample_frac(tensor, sample_rate, new_sr)

        # 如果有掩码，也需要相应地调整掩码大小
        if mask is None:
            return resampled_tensor
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )

    # ------------------------------------------------------------------------
    # 采样与量化效果
    # ------------------------------------------------------------------------

    @staticmethod
    def updownresample(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        intermediate_freq: int = 32000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        # 保存原始长度
        orig_length = tensor.shape[-1]

        # 上采样到中间频率（提高采样率）
        tensor = torchaudio.functional.resample(tensor, sample_rate, intermediate_freq)

        # 下采样回原始频率（降低采样率）
        # 这个过程会引入重采样失真，模拟真实场景中的质量损失
        tensor = torchaudio.functional.resample(tensor, intermediate_freq, sample_rate)

        # 由于浮点精度问题，重采样后长度可能略有变化，需要调整
        curr_length = tensor.shape[-1]
        if curr_length > orig_length:
            tensor = tensor[..., :orig_length]
        elif curr_length < orig_length:
            pad_size = orig_length - curr_length
            tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)

        return audio_effect_return(tensor=tensor, mask=mask)
    

    @staticmethod
    def quantize_audio(
        tensor: torch.Tensor,
        bits_range: tuple = (6,12), 
        mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """对音频进行量化
        
        Args:
            tensor: 输入音频张量
            bits: 量化位数
            mask: 掩码张量
            
        Returns:
            量化后的音频张量
        """
        # 计算当前张量的最大绝对值（用于归一化）
        max_val = torch.max(torch.abs(tensor))

        # 随机选择量化位数
        bits=torch.randint(bits_range[0],bits_range[1],(1,))[0].to(tensor.device)

        # 计算量化步长：范围除以量化级别数
        step = 2 * max_val / (2**bits - 1)

        # 执行量化：舍入到最近的量化级别
        quant = torch.round(tensor / step) * step

        # 使用直通估计器保持可微分性
        out = tensor + (quant - tensor).detach()

        return audio_effect_return(tensor=out, mask=mask)

    # ------------------------------------------------------------------------
    # 回声与混响效果
    # ------------------------------------------------------------------------

    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """将音频音量衰减0.4倍，延迟100ms，
        然后与原音频叠加。

        Args:
            tensor: 表示音频信号的3D张量 [bsz, channels, frames]
            volumne range: 回声信号的音量范围
            duration range: 回声信号的持续时间范围
            sample_rate: 音频信号的采样率。
        Returns:
            带有混响的音频信号。
        """

        # 创建一个简单的脉冲响应
        # 脉冲响应的持续时间（秒）
        duration = torch.FloatTensor(1).uniform_(*duration_range)
        volume = torch.FloatTensor(1).uniform_(*volume_range)

        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

        # 定义几个振幅递减的反射
        impulse_response[0] = 1.0  # 直达声

        impulse_response[
            int(sample_rate * duration) - 1
        ] = volume  # 100ms后的第一次反射

        # 为脉冲响应添加批次和通道维度
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        # 将音频信号与脉冲响应进行卷积
        reverbed_signal = fft_conv1d(tensor, impulse_response)

        # 为了稳定性，归一化到原始振幅范围
        reverbed_signal = (
            reverbed_signal
            / (torch.max(torch.abs(reverbed_signal))+1e-6)
            * torch.max(torch.abs(tensor))
        )

        # 确保张量大小不变
        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp

        return audio_effect_return(tensor=reverbed_signal, mask=mask)

    # ------------------------------------------------------------------------
    # 噪声添加效果
    # ------------------------------------------------------------------------

    @staticmethod
    def random_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """向波形添加高斯噪声。

        高斯白噪声在所有频率上具有相等的能量密度。
        """
        # 生成与输入相同形状的高斯噪声
        noise = torch.randn_like(waveform) * noise_std

        # 将噪声添加到原始波形
        noisy_waveform = waveform + noise
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def pink_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.01,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """向波形添加粉红背景噪声。

        粉红噪声（1/f噪声）在对数频率尺度上能量分布均匀，
        听起来比白噪声更自然。
        """
        # 生成粉红噪声
        noise = generate_pink_noise(waveform.shape[-1])

        # 计算当前噪声的标准差
        current_std = torch.std(noise)

        # 缩放噪声使其标准差等于目标值
        scaled_noise = noise * (noise_std / (current_std + 1e-6))
        scaled_noise = scaled_noise.to(waveform.device)
        noise = scaled_noise.to(waveform.device)

        # 假设波形的形状为 (bsz, channels, length)
        # 添加批次和通道维度
        noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0).to(waveform.device)
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    # ------------------------------------------------------------------------
    # 频域滤波效果
    # ------------------------------------------------------------------------

    @staticmethod
    def lowpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 5000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """对波形应用低通滤波

        低通滤波器保留低于截止频率的信号，衰减高于截止频率的信号。
        常用于去除高频噪声或模拟带宽限制。
        """
        return audio_effect_return(
            tensor=julius.lowpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def highpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 500,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """对波形应用高通滤波

        高通滤波器保留高于截止频率的信号，衰减低于截止频率的信号。
        常用于去除低频噪声（如轰鸣声）或模拟电话线路效果。
        """
        return audio_effect_return(
            tensor=julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def bandpass_filter(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """通过级联高通滤波器和低通滤波器对波形应用带通滤波。

        Args:
            waveform (torch.Tensor): 输入音频波形。
            low_cutoff (float): 低截止频率。
            high_cutoff (float): 高截止频率。
            sample_rate (int): 波形的采样率。

        Returns:
            torch.Tensor: 滤波后的音频波形。
        """

        return audio_effect_return(
            tensor=julius.bandpass_filter(
                waveform,
                cutoff_low=cutoff_freq_low / sample_rate,
                cutoff_high=cutoff_freq_high / sample_rate,
            ),
            mask=mask,
        )

    # ------------------------------------------------------------------------
    # 平滑与音量调整
    # ------------------------------------------------------------------------

    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """使用指定窗口大小的移动平均滤波器平滑输入张量（音频信号）。

        Args:
            tensor (torch.Tensor): 输入音频张量。假设张量形状为 (batch_size,
            channels, time)。
            window_size (int): 移动平均窗口的大小。
            mask: 输入波形的掩码

        Returns:
            torch.Tensor: 平滑后的音频张量。
        """
        # 随机选择窗口大小
        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))

        # 创建一个均匀平滑核（移动平均滤波器）
        kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
        kernel = kernel.to(tensor.device)

        # 使用 FFT 快速卷积进行平滑
        smoothed = fft_conv1d(tensor, kernel)

        # 确保张量大小不变（处理卷积可能改变的长度）
        tmp = torch.zeros_like(tensor)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return audio_effect_return(tensor=smoothed, mask=mask)

    @staticmethod
    def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """增强音频音量

        Args:
            amount: 增强百分比（如 20 表示增强 20%）
        """
        return audio_effect_return(tensor=tensor * (1 + amount / 100), mask=mask)

    @staticmethod
    def duck_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """用衰减信号对输入波形进行掩码处理（降低音量）

        Args:
            amount: 衰减百分比（如 20 表示降低 20%）
        """
        return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)

    # ------------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------------

    @staticmethod
    def identity(
        tensor: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """恒等变换 - 不做任何处理，直接返回输入

        用作基准对照或在不需要应用任何效果时使用。
        """
        return audio_effect_return(tensor=tensor, mask=mask)

    # ------------------------------------------------------------------------
    # 音频压缩编码
    # ------------------------------------------------------------------------

    @staticmethod
    def mp3_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        bitrate_range: tp.Optional[tp.List[str]] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        使用MP3算法压缩音频
        Args:
            tensor (torch.Tensor): 输入音频张量。
            sample_rate (int): 音频的采样率。
            bitrate (str): MP3压缩的比特率。
            bitrate_range (List[str]): 比特率列表，随机选择一个。
                例如: ["64k", "128k", "192k"]。如果提供，将忽略bitrate参数。

        Returns:
            torch.Tensor: 应用MP3压缩后的输出张量。
        """
        # 如果提供了比特率范围，随机选择一个
        if bitrate_range is not None and len(bitrate_range) > 0:
            bitrate = random.choice(bitrate_range)

        # 使用直通估计器应用 MP3 压缩
        # 这样在训练时可以反向传播梯度
        out = apply_compression_skip_grad(
            tensor, get_mp3, sr=sample_rate, bitrate=bitrate
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def aac_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        bitrate_range: tp.Optional[tp.List[str]] = None,
        lowpass_freq: tp.Optional[int] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """对音频张量应用AAC压缩。

        Args:
            tensor (torch.Tensor): 输入音频张量。
            sample_rate (int): 音频的采样率。
            bitrate (str): AAC压缩的比特率。
            bitrate_range (List[str]): 比特率列表，随机选择一个。
                例如: ["64k", "128k", "192k"]。如果提供，将忽略bitrate参数。
            lowpass_freq (Optional[int]): 低通滤波器的频率。

        Returns:
            torch.Tensor: 应用AAC压缩后的输出张量。
        """
        # 如果提供了比特率范围，随机选择一个
        if bitrate_range is not None and len(bitrate_range) > 0:
            bitrate = random.choice(bitrate_range)

        # 使用直通估计器应用 AAC 压缩
        # AAC 通常比 MP3 提供更好的音质（在相同比特率下）
        out = apply_compression_skip_grad(
            tensor, get_aac, sr=sample_rate, bitrate=bitrate, lowpass_freq=lowpass_freq
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def opus_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "64k",
        bitrate_range: tp.Optional[tp.List[str]] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """对音频张量应用Opus压缩。

        Opus是一种开放、免版税的音频编解码器，特别适合语音和音乐的低延迟传输。
        在低比特率下表现优于MP3和AAC。

        Args:
            tensor (torch.Tensor): 输入音频张量。
            sample_rate (int): 音频的采样率。
            bitrate (str): Opus压缩的比特率，默认64k。
            bitrate_range (List[str]): 比特率列表，随机选择一个。
                例如: ["32k", "64k", "128k"]。如果提供，将忽略bitrate参数。

        Returns:
            torch.Tensor: 应用Opus压缩后的输出张量。
        """
        # 如果提供了比特率范围，随机选择一个
        if bitrate_range is not None and len(bitrate_range) > 0:
            bitrate = random.choice(bitrate_range)

        out = apply_compression_skip_grad(
            tensor, get_opus, sr=sample_rate, bitrate=bitrate
        )
        return audio_effect_return(tensor=out, mask=mask)
