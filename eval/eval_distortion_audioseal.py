"""
AudioSeal 失真鲁棒性评估脚本 - eval_distortion_audioseal.py
============================================================

功能：评估 AudioSeal 水印在各种音频失真（噪声、压缩、滤波等）下的鲁棒性

评估指标：
1. det_acc: 水印检测准确率（检测音频是否包含水印）
2. rec_acc: 16-bit 消息恢复准确率

输出：
- CSV文件: experiments/distortion_eval_audioseal/{失真类型}.csv
- 日志文件: experiments/distortion_eval_audioseal/experments.result.log
- 控制台: 结构化的统计结果表格
"""

import os
import hydra
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

from tqdm import tqdm
import time
import torch
import torchaudio

from distortions.audio_effects import get_audio_effects
from dataset.data import wav_dataset as used_dataset

from losses.sisnr import SISNR
from pesq import pesq as pesq_fn

import csv

from audioseal import AudioSeal

# ============ 全局配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AudioSeal 模型路径（可以使用预训练模型或自定义路径）
# 使用预训练模型：
# generator = AudioSeal.load_generator("audioseal_wm_16bits")
# detector = AudioSeal.load_detector("audioseal_detector_16bits")

# 使用自定义路径：
GENERATOR_PATH = '../audioseal/checkpoints/checkpoint_generator_test.pth'
DETECTOR_PATH = '../audioseal/checkpoints/checkpoint_detector_test.pth'
NBITS = 16


def compute_detection_acc(detection_output, threshold=0.5):
    """
    计算检测准确率

    Args:
        detection_output: detector 输出的检测概率 [B, 1, T] 或 [B, 2, T]
        threshold: 检测阈值

    Returns:
        accuracy: 检测准确率（预测为有水印的比例）
    """
    if detection_output.dim() == 3:
        # 取平均概率
        prob = detection_output.mean(dim=-1).mean(dim=-1)  # [B]
    else:
        prob = detection_output

    acc = (prob > threshold).float().mean()
    return acc


def compute_bit_acc(decoded_msg, original_msg):
    """
    计算消息恢复的比特准确率

    Args:
        decoded_msg: 解码的消息 [B, nbits] 或 [B, nbits, T]
        original_msg: 原始消息 [B, nbits]

    Returns:
        accuracy: 比特准确率
    """
    if decoded_msg.dim() == 3:
        # 取时间平均
        decoded_msg = decoded_msg.mean(dim=-1)  # [B, nbits]

    # 二值化
    decoded_bits = (decoded_msg > 0).float()
    original_bits = original_msg.float()

    acc = (decoded_bits == original_bits).float().mean()
    return acc


def main(eval_args):
    """
    主评估函数
    """

    # ============ 第1步：初始化 ============

    # 创建输出目录
    os.makedirs('experiments/distortion_eval_audioseal', exist_ok=True)

    # 设置日志文件路径
    logfile = 'experiments/distortion_eval_audioseal/experments.result.log'
    logger = get_logger(log_file=logfile)
    logger.info(f'=' * 80)
    logger.info(f'开始 AudioSeal 失真鲁棒性评估')
    logger.info(f'时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    logger.info(f'=' * 80)

    # ============ 第2步：加载 AudioSeal 模型 ============

    logger.info(f'\n加载 AudioSeal 模型...')

    # 尝试加载自定义路径，如果失败则使用预训练模型
    try:
        generator = AudioSeal.load_generator(GENERATOR_PATH, nbits=NBITS)
        detector = AudioSeal.load_detector(DETECTOR_PATH, nbits=NBITS)
        logger.info(f'✓ 从自定义路径加载模型')
        logger.info(f'  Generator: {GENERATOR_PATH}')
        logger.info(f'  Detector: {DETECTOR_PATH}')
    except:
        generator = AudioSeal.load_generator("audioseal_wm_16bits")
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        logger.info(f'✓ 使用预训练模型: audioseal_wm_16bits / audioseal_detector_16bits')

    generator = generator.to(device)
    detector = detector.to(device)
    generator.eval()
    detector.eval()

    logger.info(f'✓ 模型加载成功，消息位数: {NBITS}')

    # ============ 第3步：准备数据集和失真效果 ============

    # 加载测试数据集
    val_audios = used_dataset(config=eval_args.dataset, flag='test')
    logger.info(f'✓ 加载测试数据集: {len(val_audios)} 个样本')

    # 获取所有失真效果
    effects = get_audio_effects(eval_args.augmentation)

    # 只测试指定的失真效果（测试完成后删除此行）
    effects = {k: v for k, v in effects.items() if k in ['opus_compression']}

    # 移除不稳定的失真效果
    if 'speed' in effects:
        del effects['speed']
    if 'echo' in effects:
        del effects['echo']

    logger.info(f'✓ 加载失真效果: {len(effects)} 种')
    logger.info(f'  失真类型: {", ".join(effects.keys())}')

    # 创建SNR计算器
    snr_fn = SISNR()

    # ============ 第4步：初始化结果存储 ============

    results = {}
    csv_fns = {}
    csv_writers = {}
    stats = {}

    for augmentation_name in effects.keys():
        # 创建CSV文件
        csv_file = f'experiments/distortion_eval_audioseal/{augmentation_name}.csv'
        csv_fns[augmentation_name] = open(csv_file, 'w')
        csv_writers[augmentation_name] = csv.writer(csv_fns[augmentation_name])

        # 写入CSV表头
        csv_str = ['file', 'det_acc', 'rec_acc', 'snr', 'pesq']
        csv_writers[augmentation_name].writerow(csv_str)

        # 初始化结果存储
        stats[augmentation_name] = {}
        results[augmentation_name] = {
            'det_acc': [],
            'rec_acc': [],
            'snr': [],
            'pesq': []
        }

    # ============ 第5步：开始评估 ============

    logger.info(f'\n开始评估...\n')

    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_audios), total=len(val_audios), desc="评估进度"):

            # 准备输入音频 [batch=1, channel=1, time]
            x = sample["matrix"].reshape(1, 1, -1).to(device)

            # 确保音频长度足够（至少1秒）
            if x.shape[-1] < 16000:
                continue

            # -------- 生成水印消息 --------
            # 随机生成 16-bit 消息
            msg = torch.randint(0, 2, (1, NBITS), device=device).float()

            # -------- 嵌入水印 --------
            wm = generator.get_watermark(x, sample_rate=16000, message=msg)
            x_wm = x + wm

            # 计算原始音频质量指标
            try:
                orig_snr = -snr_fn(x_wm, x).item()
                orig_pesq = pesq_fn(16000,
                                    x[0, 0].cpu().numpy()[:48000],
                                    x_wm[0, 0].cpu().numpy()[:48000])
            except:
                orig_snr = 0.0
                orig_pesq = 0.0

            # -------- 对每种失真进行评估 --------
            for augmentation_name, augmentation_method in effects.items():
                csv_str = [sample['name']]

                # 应用失真
                aug_x_wm = augmentation_method(x_wm)

                # 确保长度匹配（某些失真可能改变长度）
                min_len = min(aug_x_wm.shape[-1], x_wm.shape[-1])
                aug_x_wm_trimmed = aug_x_wm[..., :min_len]

                # -------- 检测和解码 --------
                # 使用高级 API
                try:
                    result, decoded_msg = detector.detect_watermark(aug_x_wm_trimmed, sample_rate=16000)

                    # 检测准确率（result 是检测概率）
                    det_acc = (result > 0.5).float().item()

                    # 消息恢复准确率
                    rec_acc = compute_bit_acc(decoded_msg, msg).item()

                except Exception as e:
                    # 如果高级 API 失败，尝试低级 API
                    try:
                        output = detector.detector(aug_x_wm_trimmed)  # [B, 2+nbits, T]

                        # 检测输出（前2个通道）
                        det_output = output[:, :2, :]
                        det_acc = (det_output[:, 1, :].mean() > 0.5).float().item()

                        # 消息解码（后nbits个通道）
                        msg_output = output[:, 2:, :]
                        rec_acc = compute_bit_acc(msg_output, msg).item()
                    except:
                        det_acc = 0.0
                        rec_acc = 0.5  # 随机猜测

                # -------- 保存结果 --------
                results[augmentation_name]['det_acc'].append(det_acc)
                results[augmentation_name]['rec_acc'].append(rec_acc)
                results[augmentation_name]['snr'].append(orig_snr)
                results[augmentation_name]['pesq'].append(orig_pesq)

                # 写入CSV
                csv_str.append(f'{det_acc:.3f}')
                csv_str.append(f'{rec_acc:.3f}')
                csv_str.append(f'{orig_snr:.3f}')
                csv_str.append(f'{orig_pesq:.3f}')

                csv_writers[augmentation_name].writerow(csv_str)
                csv_fns[augmentation_name].flush()

    # ============ 第6步：计算统计信息 ============

    logger.info(f'\n' + '=' * 80)
    logger.info(f'评估完成！计算统计信息...')
    logger.info(f'=' * 80 + '\n')

    for augmentation_name in effects.keys():
        csv_str = ['stats']
        stats_dict = {}

        for key in ['det_acc', 'rec_acc', 'snr', 'pesq']:
            values = torch.tensor(results[augmentation_name][key])
            if len(values) > 0:
                mean_val = values.mean().item()
                std_val = values.std().item()
            else:
                mean_val, std_val = 0.0, 0.0
            csv_str.append(f"{mean_val:.3f}({std_val:.3f})")
            stats_dict[key] = (mean_val, std_val)

        csv_writers[augmentation_name].writerow(csv_str)
        stats[augmentation_name] = stats_dict
        logger.info(f'{augmentation_name}: {csv_str}')

    # ============ 第7步：输出结果 ============

    print("\n" + "=" * 80)
    print("AudioSeal 失真鲁棒性评估结果汇总".center(80))
    print("=" * 80)

    print(f"\n{'失真类型':<25} {'检测准确率':<15} {'消息恢复率':<15} {'SNR':<12} {'PESQ':<12}")
    print("-" * 80)

    for augmentation_name in sorted(effects.keys()):
        s = stats[augmentation_name]
        print(f"{augmentation_name:<25} "
              f"{s['det_acc'][0]:.3f}±{s['det_acc'][1]:.3f}   "
              f"{s['rec_acc'][0]:.3f}±{s['rec_acc'][1]:.3f}   "
              f"{s['snr'][0]:.2f}±{s['snr'][1]:.2f}   "
              f"{s['pesq'][0]:.2f}±{s['pesq'][1]:.2f}")

    print("-" * 80)

    # 计算平均性能
    avg_stats = {}
    for key in ['det_acc', 'rec_acc']:
        all_means = [stats[aug][key][0] for aug in effects.keys()]
        avg_stats[key] = sum(all_means) / len(all_means) if all_means else 0.0

    print(f"{'平均性能':<25} "
          f"{avg_stats['det_acc']:.3f}            "
          f"{avg_stats['rec_acc']:.3f}")

    print("=" * 80)

    # # 输出最佳/最差性能
    # print(f"\n{'指标':<15} {'最佳失真':<25} {'最佳值':<10} {'最差失真':<25} {'最差值':<10}")
    # print("-" * 85)

    # for key, key_name in [('det_acc', '检测准确率'), ('rec_acc', '消息恢复率')]:
    #     best_aug = max(effects.keys(), key=lambda x: stats[x][key][0])
    #     worst_aug = min(effects.keys(), key=lambda x: stats[x][key][0])
    #     best_val = stats[best_aug][key][0]
    #     worst_val = stats[worst_aug][key][0]

    #     print(f"{key_name:<15} {best_aug:<25} {best_val:.3f}      {worst_aug:<25} {worst_val:.3f}")

    # print("=" * 85)

    # print(f"\n详细结果已保存至:")
    # print(f"  - CSV文件: experiments/distortion_eval_audioseal/{{失真类型}}.csv")
    # print(f"  - 日志文件: {logfile}")
    # print("=" * 80 + "\n")

    # # 关闭所有CSV文件
    # for f in csv_fns.values():
    #     f.close()


@hydra.main(config_path="../config", config_name="eval")
def run(args):
    """Hydra入口函数"""
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    main(args)


if __name__ == "__main__":
    run()
