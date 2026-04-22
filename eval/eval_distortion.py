"""
音频水印失真鲁棒性评估脚本 - eval_distortion.py
=====================================================

功能：评估水印在各种音频失真（噪声、压缩、滤波等）下的鲁棒性

评估指标：
1. det_acc: 水印检测准确率
2. rec_acc: temp消息重构准确率
3. val_acc: val消息重构准确率
4. det_sdr: 检测SDR (Signal Detection Rate)
5. rec_sdr: temp消息SDR
6. val_sdr: val消息SDR

输出：
- CSV文件: experiments/distortion_eval/{失真类型}.csv
- 日志文件: experiments/distortion_eval/experments.result.log
- 控制台: 结构化的统计结果表格
"""

import os
import hydra
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

from tqdm import tqdm
import time
import torch, copy
import torchaudio

import matplotlib.pyplot as plt

from utils.tools import load_ckpt, save_ckpt
from itertools import chain
from scipy.stats import multivariate_normal

from distortions.audio_effects import (
    get_audio_effects,
    select_audio_effects,
)
from utils.wm_process import crop, MsgGenerator

from models.WMbuilder import WMEmbedder, WMExtractor
from dataset.data import wav_dataset as used_dataset

import omegaconf

from hydra.experimental import compose, initialize
import matplotlib.pyplot as plt
from utils.wm_process import crop, MsgGenerator, post_process, sequence_to_segments_fast
from pesq import pesq as pesq_fn

from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn

import csv
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast, SDR_Evaluator

# ============ 全局配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint路径（训练好的模型）
# ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_70_2026-01-08_22_01_07.pth'
ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_74_2026-01-11_18_37_35.pth'

def main(eval_args):
    """
    主评估函数

    流程：
    1. 加载模型和配置
    2. 准备失真效果和数据集
    3. 对每个测试样本应用所有失真
    4. 评估水印在失真后的性能
    5. 计算并保存统计结果

    Args:
        eval_args: Hydra配置对象，包含数据集和增强参数
    """

    # ============ 第1步：加载配置和初始化 ============

    # 从checkpoint目录加载训练配置
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]

    # 设置日志文件路径
    logfile = 'experiments/distortion_eval/experments.result.log'
    logger = get_logger(log_file=logfile)
    logger.info(f'=' * 80)
    logger.info(f'开始失真鲁棒性评估')
    logger.info(f'时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    logger.info(f'Checkpoint: {ckpt_path}')
    logger.info(f'=' * 80)

    # 加载训练时的水印配置
    wm_args = omegaconf.OmegaConf.load(ckpt_config)

    # 获取消息长度配置
    msg_fix_length = wm_args.wm_msg.fix.length      # 固定消息位数
    msg_rec_length = wm_args.wm_msg.temp.length     # 时序消息位数
    msg_val_length = wm_args.wm_msg.val.length      # 验证消息位数

    logger.info(f'\n消息配置: fix={msg_fix_length}, temp={msg_rec_length}, val={msg_val_length}')

    # ============ 第2步：加载模型 ============

    # 创建水印嵌入器
    embedder = WMEmbedder(
        model_config=wm_args.model,
        klass=wm_args.embedder,
        msg_fix_nbits=msg_fix_length,
        msg_temp_nbits=msg_rec_length,
        msg_val_nbits=msg_val_length
    ).to(device)

    # 创建水印检测器
    detector = WMExtractor(
        model_config=wm_args.model,
        klass=wm_args.detector,
        output_dim=32,
        nbits=msg_fix_length + msg_rec_length + msg_val_length
    ).to(device)

    # 从checkpoint加载模型权重
    module_state_dict = load_ckpt(ckpt_path)
    embedder.load_state_dict(module_state_dict['embedder'], strict=False)
    detector.load_state_dict(module_state_dict['detector'], strict=False)

    logger.info(f'✓ 模型加载成功')

    # ============ 第3步：准备数据集和失真效果 ============

    # 加载测试数据集
    val_audios = used_dataset(config=eval_args.dataset, flag='test')
    logger.info(f'✓ 加载测试数据集: {len(val_audios)} 个样本')

    # 获取所有失真效果（噪声、压缩、滤波等）
    effects = get_audio_effects(eval_args.augmentation)

    # 只测试指定的失真效果（测试完成后删除此行）
    effects = {k: v for k, v in effects.items() if k in ['opus_compression']}

    # 移除不稳定的失真效果
    if 'speed' in effects:
        del effects['speed']  # 速度变化可能导致不稳定
    if 'echo' in effects:
        del effects['echo']   # 回声效果可能导致NaN

    logger.info(f'✓ 加载失真效果: {len(effects)} 种')
    logger.info(f'  失真类型: {", ".join(effects.keys())}')

    # 创建水印消息生成器
    msg_generator = MsgGenerator(wm_args.wm_msg)

    # 创建SNR计算器
    snr_fn = SISNR()

    # ============ 第4步：初始化结果存储 ============

    # 为每种失真创建CSV文件和结果字典
    results = {}        # 存储所有样本的指标
    csv_fns = {}        # CSV文件句柄
    csv_writers = {}    # CSV写入器
    stats = {}          # 统计信息

    for (augmentation_name, augmentation_method) in effects.items():
        # 创建CSV文件
        csv_file = f'experiments/distortion_eval/{augmentation_name}.csv'
        csv_fns[augmentation_name] = open(csv_file, 'w')
        csv_writers[augmentation_name] = csv.writer(csv_fns[augmentation_name])

        # 写入CSV表头
        csv_str = ['file', 'det_acc', 'rec_acc', 'val_acc', 'det_sdr', 'rec_sdr', 'val_sdr']
        csv_writers[augmentation_name].writerow(csv_str)

        # 初始化结果存储
        stats[augmentation_name] = {}
        results[augmentation_name] = {
            'det_acc': [],  # 检测准确率列表
            'rec_acc': [],  # temp消息准确率列表
            'val_acc': [],  # val消息准确率列表
            'det_sdr': [],  # 检测SDR列表
            'rec_sdr': [],  # temp消息SDR列表
            'val_sdr': []   # val消息SDR列表
        }

    # 创建SDR评估器
    sdr_evaluator = SDR_Evaluator()

    # ============ 第5步：开始评估 ============

    logger.info(f'\n开始评估...\n')

    with torch.no_grad():  # 评估时不需要梯度
        # 遍历所有测试样本
        for i, sample in tqdm(enumerate(val_audios), total=len(val_audios), desc="评估进度"):

            # 准备输入音频 [batch=1, channel=1, time]
            x = sample["matrix"].reshape(1, 1, -1).to(device)

            # -------- 生成水印消息 --------
            # msg_total: 完整消息 [B, fix+temp+val, T]
            # msg_seg: (msg_rec_seg, msg_val_seg) 消息段信息
            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
            msg_rec_seg, msg_val_seg = msg_seg

            # -------- 嵌入水印 --------
            # wm: 水印信号 [B, 1, T]
            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

            # 生成带水印音频
            x_wm = x + wm

            # -------- 准备篡改检测数据 --------
            # crop函数模拟音频篡改（删除、插入、替换片段）
            x_ = copy.deepcopy(x.detach())
            wm_ = copy.deepcopy(wm.detach())
            msg_total_ = copy.deepcopy(msg_total.detach())

            # x_det: 篡改后的音频
            # mask: 掩码，1表示有水印，0表示被篡改
            # msg_det_total: 篡改后的消息
            x_det, wm_det, mask, msg_det_total = crop(x_, wm_, wm_args.crop, msg_total_, return_seg=True)
            x_wm_det = x_det + wm_det
            msg_det_seg = sequence_to_segments_fast(mask)

            # -------- 对每种失真进行评估 --------
            for (augmentation_name, augmentation_method) in effects.items():
                csv_str = [sample['name']]  # CSV行数据，首列是文件名

                # === 1. 检测任务：检测音频是否被篡改 ===
                # 对篡改后的带水印音频应用失真
                aug_x_wm_det = augmentation_method(x_wm_det)

                # 检测水印
                wm_pred_det = detector(aug_x_wm_det)

                # 后处理：平滑检测结果
                # res_det: [B, 1, T] 检测置信度
                # seg_det: 检测到的水印段
                res_det, seg_det = post_process(wm_pred_det[0], window_size=63, return_seg=True)

                # === 2. 重构任务：从失真音频中提取消息 ===
                # 对完整带水印音频应用失真
                aug_x_wm = augmentation_method(x_wm)

                # 检测并提取消息
                # wm_pred[0]: 检测结果
                # wm_pred[1]: 消息重构结果 [B, fix+temp+val, T]
                wm_pred = detector(aug_x_wm)

                # 从原始音频提取（对比基线，理论上应该没有水印）
                wm_pred_neg = detector(x)

                # 分离temp和val消息
                # res_rec: temp消息的重构结果
                # res_val: val消息的重构结果
                res_rec, seg_rec = post_process(wm_pred[1][:, :msg_rec_length, :], window_size=63, return_seg=True)
                res_val, seg_val = post_process(wm_pred[1][:, msg_rec_length:, :], window_size=63, return_seg=True)

                # -------- 计算评估指标 --------

                # 1. 检测准确率：能否正确区分有水印和无水印的区域
                det_acc = compute_accuracy(res_det, wm_pred_neg[0], mask=mask)

                # 2. temp消息重构准确率
                rec_acc = compute_bit_acc(res_rec, msg_total[:, :-msg_val_length], sigmoid=False)

                # 3. val消息重构准确率
                val_acc = compute_bit_acc(res_val, msg_total[:, -msg_val_length:], sigmoid=False)

                # 4. SDR指标（Signal Detection Rate）
                # 衡量检测/重构的信号质量
                curr_sdr_det = sdr_evaluator.update(res_det, msg_det_seg[0])
                curr_sdr_rec = sdr_evaluator.update(res_rec, msg_rec_seg)
                curr_sdr_val = sdr_evaluator.update(res_val, msg_val_seg)

                # -------- 保存结果 --------

                # 添加到结果列表
                results[augmentation_name]['det_acc'].append(det_acc.item())
                results[augmentation_name]['rec_acc'].append(rec_acc.item())
                results[augmentation_name]['val_acc'].append(val_acc.item())
                results[augmentation_name]['det_sdr'].append(curr_sdr_det)
                results[augmentation_name]['rec_sdr'].append(curr_sdr_rec)
                results[augmentation_name]['val_sdr'].append(curr_sdr_val)

                # 写入CSV
                csv_str.append(f'{det_acc.item():.3f}')
                csv_str.append(f'{rec_acc.item():.3f}')
                csv_str.append(f'{val_acc.item():.3f}')
                csv_str.append(f'{curr_sdr_det:.3f}')
                csv_str.append(f'{curr_sdr_rec:.3f}')
                csv_str.append(f'{curr_sdr_val:.3f}')

                csv_writers[augmentation_name].writerow(csv_str)
                csv_fns[augmentation_name].flush()  # 立即写入磁盘

    # ============ 第6步：计算统计信息并保存 ============

    logger.info(f'\n' + '=' * 80)
    logger.info(f'评估完成！计算统计信息...')
    logger.info(f'=' * 80 + '\n')

    # 保存统计信息到CSV并打印
    for (augmentation_name, augmentation_method) in effects.items():
        csv_str = ['stats']
        stats_dict = {}

        # 计算每个指标的均值和标准差
        for key in ['det_acc', 'rec_acc', 'val_acc', 'det_sdr', 'rec_sdr', 'val_sdr']:
            mean, std = trim_mean_std(torch.tensor(results[augmentation_name][key]))
            csv_str.append(f"{mean.item():.3f}({std:.3f})")
            stats_dict[key] = (mean.item(), std.item())

        # 保存统计行到CSV
        csv_writers[augmentation_name].writerow(csv_str)

        # 保存到stats字典（用于最终输出）
        stats[augmentation_name] = stats_dict

        # 写入日志
        logger.info(f'{augmentation_name}: {csv_str}')

    # ============ 第7步：结构化输出最终结果 ============

    print("\n" + "=" * 100)
    print("失真鲁棒性评估结果汇总".center(100))
    print("=" * 100)

    # 打印表头
    print(f"\n{'失真类型':<20} {'检测准确率':<15} {'Temp准确率':<15} {'Val准确率':<15} {'检测SDR':<15} {'TempSDR':<15} {'ValSDR':<15}")
    print("-" * 100)

    # 打印每种失真的结果
    for augmentation_name in sorted(effects.keys()):
        s = stats[augmentation_name]
        print(f"{augmentation_name:<20} "
              f"{s['det_acc'][0]:.3f}±{s['det_acc'][1]:.3f}   "
              f"{s['rec_acc'][0]:.3f}±{s['rec_acc'][1]:.3f}   "
              f"{s['val_acc'][0]:.3f}±{s['val_acc'][1]:.3f}   "
              f"{s['det_sdr'][0]:.3f}±{s['det_sdr'][1]:.3f}   "
              f"{s['rec_sdr'][0]:.3f}±{s['rec_sdr'][1]:.3f}   "
              f"{s['val_sdr'][0]:.3f}±{s['val_sdr'][1]:.3f}")

    print("-" * 100)

    # 计算并打印平均性能
    avg_stats = {}
    for key in ['det_acc', 'rec_acc', 'val_acc', 'det_sdr', 'rec_sdr', 'val_sdr']:
        all_means = [stats[aug][key][0] for aug in effects.keys()]
        avg_stats[key] = sum(all_means) / len(all_means)

    print(f"{'平均性能':<20} "
          f"{avg_stats['det_acc']:.3f}        "
          f"{avg_stats['rec_acc']:.3f}        "
          f"{avg_stats['val_acc']:.3f}        "
          f"{avg_stats['det_sdr']:.3f}        "
          f"{avg_stats['rec_sdr']:.3f}        "
          f"{avg_stats['val_sdr']:.3f}")

    print("=" * 100)

    # 输出最佳/最差性能
    print(f"\n{'指标':<15} {'最佳失真':<20} {'最佳值':<10} {'最差失真':<20} {'最差值':<10}")
    print("-" * 75)

    for key, key_name in [
        ('det_acc', '检测准确率'),
        ('rec_acc', 'Temp准确率'),
        ('val_acc', 'Val准确率'),
        ('det_sdr', '检测SDR'),
        ('rec_sdr', 'TempSDR'),
        ('val_sdr', 'ValSDR')
    ]:
        best_aug = max(effects.keys(), key=lambda x: stats[x][key][0])
        worst_aug = min(effects.keys(), key=lambda x: stats[x][key][0])
        best_val = stats[best_aug][key][0]
        worst_val = stats[worst_aug][key][0]

        print(f"{key_name:<15} {best_aug:<20} {best_val:.3f}      {worst_aug:<20} {worst_val:.3f}")

    print("=" * 100)

    # 输出文件位置
    print(f"\n详细结果已保存至:")
    print(f"  - CSV文件: experiments/distortion_eval/{{失真类型}}.csv")
    print(f"  - 日志文件: {logfile}")
    print("=" * 100 + "\n")

    # 关闭所有CSV文件
    for f in csv_fns.values():
        f.close()


def trim_mean_std(values, trim_percent=1):
    """
    计算修剪均值和标准差（去除极端值）

    通过去除最高和最低的百分比数据来减少异常值的影响，
    使统计结果更加稳健。

    Args:
        values: tensor, 需要计算统计量的数值
        trim_percent: float, 从两端各去除的百分比（默认1%）

    Returns:
        mean: 修剪后的均值
        std: 修剪后的标准差

    Example:
        >>> values = torch.tensor([1, 2, 3, 100, 4, 5])  # 100是异常值
        >>> mean, std = trim_mean_std(values, trim_percent=10)
        >>> # 去除最高10%和最低10%后计算统计量
    """
    sorted_vals, _ = torch.sort(values)
    n = len(sorted_vals)
    k = int(n * trim_percent / 100)
    trimmed_vals = sorted_vals[k:n - k]
    return trimmed_vals.mean(), trimmed_vals.std()


@hydra.main(config_path="../config", config_name="eval")
def run(args):
    """
    Hydra入口函数

    使用Hydra管理配置，自动切换到原始工作目录

    Args:
        args: Hydra配置对象
    """
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    main(args)


if __name__ == "__main__":
    run()
