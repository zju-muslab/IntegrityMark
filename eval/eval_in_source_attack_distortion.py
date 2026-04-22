"""
======================================================================================================
eval_in_source_attack_distortion.py - 失真条件下的同源篡改检测评估脚本
======================================================================================================

【脚本功能】
评估水印系统在音频失真(如MP3/AAC压缩)条件下对同源篡改攻击的检测能力

【评估流程】
1. 加载带水印音频
2. 应用音频失真(压缩等)
3. 进行同源篡改攻击(用同一音频的其他片段替换部分内容)
4. 检测篡改位置
5. 计算检测准确性指标

【主要指标】
- TP (True Positive): 正确检测到的篡改片段数
- TN (True Negative): 正确识别的未篡改片段数
- FP (False Positive): 误报为篡改的未篡改片段数
- FN (False Negative): 漏检的篡改片段数
- ACC (Accuracy): 总体准确率 = (TP+TN)/(TP+TN+FP+FN)
- TPR (True Positive Rate/Recall): 召回率 = TP/(TP+FN)
- TNR (True Negative Rate): 特异度 = TN/(TN+FP)
- FPR (False Positive Rate): 误报率 = FP/(FP+TN)
- FNR (False Negative Rate): 漏检率 = FN/(FN+TP)
- Precision: 精确率 = TP/(TP+FP)
- F1: F1分数 = 2*Precision*Recall/(Precision+Recall)
- le: 定位误差(Location Error)

【输出文件】
- experiments/in_source_attack_eval/distortion/{失真类型}.csv

【使用方法】
python eval_in_source_attack_distortion.py

【注意事项】
- 本脚本专门针对压缩失真(MP3/AAC)进行测试
- 同源篡改指用同一音频的其他片段替换，比跨源篡改更难检测
- 适用于val消息检测模型(0+0+4配置)
======================================================================================================
"""

import os
import hydra
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

from collections import deque
from tqdm import tqdm
import time
import torch,copy
import torchaudio
import random
import matplotlib.pyplot as plt

from collections import deque
import torch  # 仅当 state 需要 tensor 操作时

import random

from utils.tools import load_ckpt,save_ckpt
from itertools import chain
# from torch.optim.lr_scheduler import StepLR
from scipy.stats import multivariate_normal

from distortions.audio_effects import (
    get_audio_effects,
    select_audio_effects,
)
from utils.wm_process import crop, MsgGenerator

# from My_model.modules import Encoder, Decoder, Discriminator

# from model.My_model.privacy_wm import WMEmbedder
# from models.WMBuilder import WMEmbedder, WMExtractor
from models.WMbuilder import WMEmbedder,WMExtractor
from dataset.data import wav_dataset as used_dataset

import omegaconf

from hydra.experimental import compose, initialize
import matplotlib.pyplot as plt
from utils.wm_process import crop, MsgGenerator, post_process, sequence_to_segments_fast
from pesq import pesq as pesq_fn

from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn

# 音频篡改攻击函数
# in_source_replace: 同源替换攻击 - 用同一音频的其他片段替换部分内容
# cross_source_*: 跨源篡改攻击 - 用其他音频的片段替换
# delete: 删除攻击 - 删除部分音频片段
from utils.audio_tamper_attack import cross_source_replace, cross_source_insert, cross_source_multi_insert, delete, in_source_insert, in_source_replace

import csv
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast, SDR_Evaluator,calculate_tiou_state1
# seg_results: 计算篡改检测的TP/TN/FP/FN
# get_invalid_points_robust: 获取检测到的异常点
from utils.in_source_detection import seg_results, get_invalid_points_robust
from statistics import mean

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================================================
# Checkpoint配置
# ======================================================================================================
# 注意：这些checkpoint都是0+0+4配置(fix=0, temp=0, val=4)
# 专门用于val消息的篡改检测
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_38_2025-02-25_18_43_58.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-27_00_10_39.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-28_03_54_05.pth'
# ckpt_path = '/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_49_2025-03-01_18_45_58.pth'
# ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+2+4_32bps/0+2+4_32bps_ep_21_2025-02-22_20_52_37.pth'
ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_88_2026-01-15_21_34_30.pth'  # 当前使用的checkpoint

debug=False

def main(eval_args):
    """
    ======================================================================================================
    主评估函数 - 失真条件下的同源篡改检测评估
    ======================================================================================================
    """

    # ==================================================================================
    # 步骤1: 初始化配置和日志
    # ==================================================================================
    # 从checkpoint目录加载原始训练配置
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    # 提取checkpoint文件名(不含扩展名)
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    # 日志文件路径
    logfile='experiments/in_source_attack_eval/distortion/experments.result.log'

    logger = get_logger(log_file=logfile)
    logger.info(f'time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    # 加载训练时的配置
    wm_args = omegaconf.OmegaConf.load(ckpt_config)

    # ==================================================================================
    # 步骤2: 提取消息配置
    # ==================================================================================
    # fix消息：固定不变的消息，嵌入在模型瓶颈层
    msg_fix_length=wm_args.wm_msg.fix.length  # 本实验中为0
    # temp消息：时变消息，嵌入在输入层
    msg_rec_length=wm_args.wm_msg.temp.length  # 本实验中为0
    # val消息：用于篡改检测的消息
    msg_val_length=wm_args.wm_msg.val.length  # 本实验中为4

    # ==================================================================================
    # 步骤3: 创建并加载模型
    # ==================================================================================
    # 水印嵌入器(Embedder)：将水印嵌入到音频中
    embedder = WMEmbedder(model_config=wm_args.model, klass=wm_args.embedder,  msg_fix_nbits=msg_fix_length, msg_temp_nbits=msg_rec_length, msg_val_nbits=msg_val_length).to(device)
    # 水印提取器(Detector)：从音频中提取水印消息
    # output_dim=32: 输出32维特征用于后处理
    # nbits: 总消息位数 = fix + temp + val
    detector = WMExtractor(model_config=wm_args.model, klass=wm_args.detector, output_dim=32, nbits=msg_fix_length+msg_rec_length+msg_val_length).to(device)

    # 加载训练好的权重
    module_state_dict=load_ckpt(ckpt_path)
    embedder.load_state_dict(module_state_dict['embedder'],strict=False)
    detector.load_state_dict(module_state_dict['detector'],strict=False)

    # ==================================================================================
    # 步骤4: 准备测试数据和失真效果
    # ==================================================================================
    # 加载测试集音频
    val_audios = used_dataset(config=eval_args.dataset,flag='test')

    # 获取所有可用的音频失真效果
    effects = get_audio_effects(eval_args.augmentation)  # noqa
    # 移除不稳定的效果
    if 'speed' in effects:
        del effects['speed']  # 变速会改变音频长度，影响评估
    if 'echo' in effects:
        del effects['echo']  # 回声效果不稳定

    # ==================================================================================
    # 关键：只保留压缩失真进行测试
    # ==================================================================================
    # 原因：压缩失真是最常见的音频处理，对篡改检测影响最大
    # MP3/AAC压缩会引入量化噪声，可能干扰水印的完整性检测
    # new_effects={}
    # for (augmentation_name, augmentation_method) in effects.items():
    #     if augmentation_name in ['mp3_compression','aac_compression']:
    #         new_effects[augmentation_name]=augmentation_method
    # effects=new_effects

    # ==================================================================================
    # 步骤5: 初始化消息生成器
    # ==================================================================================
    # 消息生成器：为每个音频生成对应的水印消息
    msg_generator=MsgGenerator(wm_args.wm_msg)

    # ==================================================================================
    # 步骤6: 初始化结果存储
    # ==================================================================================
    results={}        # 存储每种失真的评估结果
    csv_fns={}        # CSV文件句柄
    csv_writers={}    # CSV写入器
    stats={}          # 统计信息

    # 为每种失真类型创建单独的CSV文件
    for (augmentation_name, augmentation_method) in effects.items():
        csv_file = f'experiments/in_source_attack_eval/distortion/{augmentation_name}.csv'
        csv_fns[augmentation_name]=open(csv_file,'w')
        csv_writers[augmentation_name]=csv.writer(csv_fns[augmentation_name])
        # CSV表头
        csv_str=['Name', 'TP','TN','FP','FN','ACC','TPR','TNR','FPR','FNR', 'le']
        csv_writers[augmentation_name].writerow(csv_str)
        stats[augmentation_name]={}
        # 为每种失真初始化结果字典
        results[augmentation_name] = {
            'TP': [],   # True Positive: 正确检测到的篡改片段
            'TN': [],   # True Negative: 正确识别的未篡改片段
            'FP': [],   # False Positive: 误报为篡改的片段
            'FN': [],   # False Negative: 漏检的篡改片段
            'le': [],   # Location Error: 定位误差
        }

    # ==================================================================================
    # 步骤7: 开始评估循环
    # ==================================================================================
    with torch.no_grad():

        cnt=0  # 跳过的样本计数

        for i, sample in tqdm(enumerate(val_audios)):
            # 设置样本名称
            sample['name'] = f'{i:04d}'
            # 加载音频数据
            # x shape: (batch=1, channel=1, length)
            x = sample["matrix"].reshape(1,1,-1).to(device)

            # 跳过太短的音频(小于4秒)
            if x.shape[-1] < 64000:  # 64000 samples = 4 seconds at 16kHz
                cnt+=1
                print('skip', cnt)
                continue

            # 限制最大长度为8秒(避免内存溢出)
            max_length = 8 * 16_000  # 8 seconds at 16kHz sample rate
            if x.shape[-1] > max_length:
                # 随机裁剪8秒片段
                start = random.randint(0, x.shape[-1] - max_length)
                x = x[:, :, start:start + max_length]

            # 重新加载(确保使用正确的数据)
            x = sample["matrix"].reshape(1,1,-1).to(device)
            # 对齐到100的倍数(便于后续处理)
            x = x[..., :x.shape[-1] - (x.shape[-1] % 100)]

            # ----------------------------------------------------------------------------------
            # 步骤7.1: 生成水印消息
            # ----------------------------------------------------------------------------------
            # msg_total: 完整消息 (fix + temp + val)
            # msg_fix: 固定消息部分
            # msg_rec: 时变消息部分
            # msg_val: 篡改检测消息部分
            # msg_seg: 消息的时间段信息
            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)

            # ----------------------------------------------------------------------------------
            # 步骤7.2: 嵌入水印
            # ----------------------------------------------------------------------------------
            # wm: 水印信号 shape: (1, 1, length)
            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)
            # x_wm: 带水印音频 = 原始音频 + 水印
            x_wm = x+wm

            # ----------------------------------------------------------------------------------
            # 步骤7.3: 对每种失真进行测试
            # ----------------------------------------------------------------------------------
            for (augmentation_name, augmentation_method) in effects.items():
                # 应用失真(如MP3/AAC压缩)
                aug_x_wm = augmentation_method(x_wm)

                # ----------------------------------------------------------------------------------
                # 关键：同源篡改攻击
                # ----------------------------------------------------------------------------------
                # in_source_replace: 用同一音频的其他片段替换部分内容
                # 返回:
                #   - tamper_aug_x_wm: 篡改后的音频
                #   - discrete_points: 篡改点的位置列表(起始和结束位置)
                tamper_aug_x_wm, discrete_points = in_source_replace(aug_x_wm)
                discrete_points.sort()  # 按位置排序

                # ----------------------------------------------------------------------------------
                # 步骤7.4: 检测篡改
                # ----------------------------------------------------------------------------------
                # 提取水印预测
                # wm_pred_tamper[0]: 检测分数(用于判断是否有水印)
                # wm_pred_tamper[1]: 消息预测(用于篡改检测)
                wm_pred_tamper = detector(tamper_aug_x_wm)

                # 后处理：平滑预测结果并转换为时间段
                # window_size=63: 滑动窗口大小，用于平滑预测
                # res_det_tamper: 篡改检测结果(每个位置是否篡改)
                # seg_det_tamper: 检测到的篡改时间段列表
                res_det_tamper,seg_det_tamper = post_process(wm_pred_tamper[1],window_size=63,return_seg=True)

                # ----------------------------------------------------------------------------------
                # 步骤7.5: 计算评估指标
                # ----------------------------------------------------------------------------------
                # seg_results: 比较检测结果和真实篡改位置
                # 返回:
                #   - tp: True Positive - 正确检测到的篡改片段数
                #   - fn: False Negative - 漏检的篡改片段数
                #   - tn: True Negative - 正确识别的未篡改片段数
                #   - fp: False Positive - 误报为篡改的片段数
                #   - le: Location Error - 每个检测片段的定位误差列表
                tp, fn, tn, fp, le = seg_results(seg_det_tamper[0], discrete_points)

                # ----------------------------------------------------------------------------------
                # 注：下面是备选方案，可以同时测试未篡改音频的误报率
                # ----------------------------------------------------------------------------------
                # wm_pred_neg = detector(aug_x_wm)
                # res_det_neg,seg_det_neg = post_process(wm_pred_neg[1],window_size=63,return_seg=True)
                # label_neg = []  # 未篡改音频的标签为空
                # tp2, fn2, tn2, fp2, _ = seg_results(seg_det_neg[0], label_neg)
                # # 累加两种情况的指标
                # tp=tp+tp2
                # tn=tn+tn2
                # fp=fp+fp2
                # fn=fn+fn2

                # tiou = calculate_tiou_state1(tamper_segs, seg_det_tamper[0])

                # ----------------------------------------------------------------------------------
                # 步骤7.6: 累积结果
                # ----------------------------------------------------------------------------------
                results[augmentation_name]['TP'].append(tp)
                results[augmentation_name]['TN'].append(tn)
                results[augmentation_name]['FP'].append(fp)
                results[augmentation_name]['FN'].append(fn)
                results[augmentation_name]['le'].extend(le)  # extend因为le是列表
                # results[augmentation_name]['tiou'].append(tiou)

                # ----------------------------------------------------------------------------------
                # 步骤7.7: 计算累积指标
                # ----------------------------------------------------------------------------------
                # 计算到目前为止的总指标
                TP=sum(results[augmentation_name]['TP'])
                TN=sum(results[augmentation_name]['TN'])
                FP=sum(results[augmentation_name]['FP'])
                FN=sum(results[augmentation_name]['FN'])

                # 计算各种评估指标
                ACC = (TP+TN) / (TP+TN+FP+FN)  # 准确率
                TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  # 召回率/真阳性率
                TNR = TN / (TN + FP) if (TN + FP) != 0 else 0  # 特异度/真阴性率
                FPR = FP / (FP + TN) if (FP + TN) != 0 else 0  # 误报率
                FNR = FN / (FN + TP) if (FN + TP) != 0 else 0  # 漏检率

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # 精确率
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0     # 召回率
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  # F1分数

                # ----------------------------------------------------------------------------------
                # 步骤7.8: 写入CSV
                # ----------------------------------------------------------------------------------
                csv_str=[sample['name']]
                csv_str.append(f'{tp}')  # 当前样本的TP
                csv_str.append(f'{tn}')  # 当前样本的TN
                csv_str.append(f'{fp}')  # 当前样本的FP
                csv_str.append(f'{fn}')  # 当前样本的FN
                csv_str.append(f'{ACC:.3f}')  # 累积准确率
                csv_str.append(f'precision: {precision:.3f}')  # 累积精确率
                csv_str.append(f'recall:{recall:.3f}')  # 累积召回率
                csv_str.append(f'f1:{f1:.3f}')  # 累积F1分数
                csv_str.append(f'{TPR:.3f}')  # 累积TPR
                csv_str.append(f'{TNR:.3f}')  # 累积TNR
                csv_str.append(f'{FPR:.3f}')  # 累积FPR
                csv_str.append(f'{FNR:.3f}|')  # 累积FNR
                csv_str.append(f'{mean(le) if len(le)>0 else -1}')  # 当前样本的平均定位误差
                # csv_str.append(f'{tiou:.3f}')

                # 写入文件并刷新缓冲区
                csv_writers[augmentation_name].writerow(csv_str)
                csv_fns[augmentation_name].flush()

    # ==================================================================================
    # 步骤8: 计算并输出最终统计结果
    # ==================================================================================
    for (augmentation_name, augmentation_method) in effects.items():

        # ----------------------------------------------------------------------------------
        # 计算每种失真的最终统计指标
        # ----------------------------------------------------------------------------------
        TP=sum(results[augmentation_name]['TP'])
        TN=sum(results[augmentation_name]['TN'])
        FP=sum(results[augmentation_name]['FP'])
        FN=sum(results[augmentation_name]['FN'])

        # 计算最终评估指标
        ACC = (TP+TN) / (TP+TN+FP+FN)  # 准确率
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  # 召回率
        TNR = TN / (TN + FP) if (TN + FP) != 0 else 0  # 特异度
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0  # 误报率
        FNR = FN / (FN + TP) if (FN + TP) != 0 else 0  # 漏检率

        # LE = mean(results[augmentation_name]["le"])  # 平均定位误差
        # TIOU=mean(results[augmentation_name]["tiou"])  # 时间IoU

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # 精确率
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0     # 召回率
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  # F1分数

        # ----------------------------------------------------------------------------------
        # 写入最终统计行到CSV
        # ----------------------------------------------------------------------------------
        csv_str = [f'{augmentation_name}']
        csv_str.append(f'TP={TP}')
        csv_str.append(f'TN={TN}')
        csv_str.append(f'FP={FP}')
        csv_str.append(f'FN={FN}')
        csv_str.append(f'precision: {precision:.3f}')
        csv_str.append(f'recall:{recall:.3f}')
        csv_str.append(f'f1:{f1:.3f}')
        csv_str.append(f'ACC={ACC:.3f}')
        csv_str.append(f'TPR={TPR:.3f}')
        csv_str.append(f'TNR={TNR:.3f}')
        csv_str.append(f'FPR={FPR:.3f}')
        csv_str.append(f'FNR={FNR:.3f}')
        # csv_str.append(f'LE={LE:.3f}')
        # csv_str.append(f'TIOU={TIOU:.3f}')

        # 写入CSV并记录到日志
        csv_writers[augmentation_name].writerow(csv_str)
        logger.info(csv_str)


# ==================================================================================
# 辅助函数：计算修剪均值和标准差
# ==================================================================================
def trim_mean_std(values, trim_percent=1):
    """
    计算修剪后的均值和标准差（去除极端值）

    参数:
        values: 数值tensor
        trim_percent: 从两端各去除的百分比（默认1%）

    返回:
        mean, std: 修剪后的均值和标准差

    说明:
        修剪均值可以减少极端值的影响，提供更稳健的统计结果
    """
    sorted_vals, _ = torch.sort(values)
    n = len(sorted_vals)
    k = int(n * trim_percent / 100)  # 计算要去除的样本数
    trimmed_vals = sorted_vals[k:n-k]  # 去除两端的k个样本
    return trimmed_vals.mean(), trimmed_vals.std()

# ==================================================================================
# 主程序入口
# ==================================================================================
@hydra.main(config_path="../config", config_name="eval")
def run(args):
    """
    Hydra主函数入口

    使用Hydra管理配置，从config/eval.yaml加载评估配置
    """
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)  # 切换回原始工作目录
    main(args)

if __name__ == "__main__":
    run()