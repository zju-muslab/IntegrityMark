import os
import hydra
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

from tqdm import tqdm
import time
import torch,copy
import torchaudio
import random
import matplotlib.pyplot as plt

import random

from utils.tools import load_ckpt,save_ckpt
from itertools import chain

from scipy.stats import multivariate_normal

from distortions.audio_effects import (
    get_audio_effects,
    select_audio_effects,
)
from utils.wm_process import crop, MsgGenerator





from models.WMbuilder import WMEmbedder,WMExtractor
from dataset.data import wav_dataset as used_dataset

import omegaconf

from hydra.experimental import compose, initialize
import matplotlib.pyplot as plt
from utils.wm_process import crop, MsgGenerator, post_process, sequence_to_segments_fast
from pesq import pesq as pesq_fn, PesqError

from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn

snr_fn=SISNR()

from utils.audio_tamper_attack import cross_source_replace, cross_source_insert, cross_source_multi_insert

import csv
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast, SDR_Evaluator,calculate_tiou_state1

from utils.cross_source_detection import evaluate_tampering_overlap_ratio
import numpy as np
from statistics import mean

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_38_2025-02-25_18_43_58.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-27_00_10_39.pth'

ckpt_path = '/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_49_2025-03-01_18_45_58.pth'

ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_88_2026-01-15_21_34_30.pth'
def main(eval_args):
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    logfile='experiments/cross_source_attack_eval/distortion/experments.result.log'

    logger = get_logger(log_file=logfile)
    logger.info(f'time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    wm_args = omegaconf.OmegaConf.load(ckpt_config)

    msg_fix_length=wm_args.wm_msg.fix.length
    msg_rec_length=wm_args.wm_msg.temp.length
    msg_val_length=wm_args.wm_msg.val.length

    embedder = WMEmbedder(model_config=wm_args.model, klass=wm_args.embedder,  msg_fix_nbits=msg_fix_length, msg_temp_nbits=msg_rec_length, msg_val_nbits=msg_val_length).to(device)
    detector = WMExtractor(model_config=wm_args.model, klass=wm_args.detector, output_dim=32, nbits=msg_fix_length+msg_rec_length+msg_val_length).to(device)


    module_state_dict=load_ckpt(ckpt_path)
    embedder.load_state_dict(module_state_dict['embedder'],strict=False)
    detector.load_state_dict(module_state_dict['detector'],strict=False)

    val_audios = used_dataset(config=eval_args.dataset,flag='test')

    effects = get_audio_effects(eval_args.augmentation)
    if 'speed' in effects:
        del effects['speed']
    if 'echo' in effects:
        del effects['echo']








    sdr_evaluator=SDR_Evaluator()

    cross_source_audio_path = '/data0/datasets/VCTK-Corpus/wav16_resample'
    cross_source_audios=used_dataset(config=eval_args.dataset,flag='test',dataset_path=cross_source_audio_path,test_num=10000)


    msg_generator=MsgGenerator(wm_args.wm_msg)

    results={}
    csv_fns={}
    csv_writers={}
    stats={}

    for (augmentation_name, augmentation_method) in effects.items():
        csv_file = f'experiments/cross_source_attack_eval/distortion/{augmentation_name}.csv'
        csv_fns[augmentation_name]=open(csv_file,'w')
        csv_writers[augmentation_name]=csv.writer(csv_fns[augmentation_name])
        csv_str=['Name', 'TP','TN','FP','FN','ACC','TPR','TNR','FPR','FNR', 'le','tiou']
        csv_writers[augmentation_name].writerow(csv_str)
        stats[augmentation_name]={}
        results[augmentation_name] = {
            'TP': [],
            'precesion': [],
            'recall': [],
            'toe': [],
            'f1': [],
            'TN': [],
            'FP': [],
            'FN': [],
            'le': [],
            'tiou': []
        }

    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_audios)):
            sample['name'] = f'{i:04d}'
            x = sample["matrix"].reshape(1,1,-1).to(device)
            x = x[..., :x.shape[-1] - (x.shape[-1] % 100)]
            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)
            x_wm = x+wm
            for (augmentation_name, augmentation_method) in effects.items():
                aug_x_wm = augmentation_method(x_wm)
                dataset_idx = random.randint(0, len(cross_source_audios)-1)
                tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)
                while tamper_audio.shape[-1] < 32000:
                    dataset_idx = random.randint(0, len(cross_source_audios)-1)
                    tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)
                tamper_aug_x_wm, tamper_segs = cross_source_insert(aug_x_wm, tamper_audio)
                tamper_segs.sort(key=lambda x: x[0])

                wm_pred_tamper = detector(tamper_aug_x_wm)
                wm_pred_neg = detector(aug_x_wm)

                res_det_tamper,seg_det_tamper = post_process(wm_pred_tamper[0],window_size=63,return_seg=True)

                for i, seg in enumerate(seg_det_tamper[0]):
                    seg_det_tamper[0][i] = [seg[0], seg[1], 1-seg[2][0]]









                precesion, recall, f1, toe = evaluate_tampering_overlap_ratio(seg_det_tamper[0], tamper_segs)

                toe=np.clip(toe/16000,0,0.1)


















                results[augmentation_name]['precesion'].append(precesion)
                results[augmentation_name]['recall'].append(recall)
                results[augmentation_name]['f1'].append(f1)
                results[augmentation_name]['toe'].append(toe)



















                mean_precesion=np.mean(results[augmentation_name]['precesion'])
                mean_recall=np.mean(results[augmentation_name]['recall'])
                mean_f1=np.mean(results[augmentation_name]['f1'])
                mean_toe=np.mean(results[augmentation_name]['toe'])

                csv_str=[sample['name']]




                csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
                csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
                csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
                csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')








                csv_writers[augmentation_name].writerow(csv_str)
                csv_fns[augmentation_name].flush()

    for (augmentation_name, augmentation_method) in effects.items():

        mean_precesion=np.mean(results[augmentation_name]['precesion'])
        mean_recall=np.mean(results[augmentation_name]['recall'])
        mean_f1=np.mean(results[augmentation_name]['f1'])
        mean_toe=np.mean(results[augmentation_name]['toe'])















        csv_str = [f'{augmentation_name}']
        csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
        csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
        csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
        csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')











        csv_writers[augmentation_name].writerow(csv_str)
        logger.info(csv_str)


def seg_results(detection_segs, tamper_segs):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    le = []



    tamper_segs_1 = [seg for seg in tamper_segs if seg[2] == 1]
    detection_segs_1 = [seg for seg in detection_segs if seg[2] == 1]

    for tamper_seg in tamper_segs_1:
        found_match = False
        tamper_duration = tamper_seg[1] - tamper_seg[0]
        min_error = float('inf')
        best_head_error = 1600
        best_tail_error = 1600

        for detect_seg in detection_segs_1:

            overlap_start = max(tamper_seg[0], detect_seg[0])
            overlap_end = min(tamper_seg[1], detect_seg[1])

            if overlap_end > overlap_start:
                overlap_length = overlap_end - overlap_start
                overlap_ratio = overlap_length / tamper_duration

                head_error = abs(tamper_seg[0] - detect_seg[0])
                tail_error = abs(tamper_seg[1] - detect_seg[1])
                total_error = head_error + tail_error

                if (overlap_ratio > 0.5 and
                    head_error <= 1600 and
                    tail_error <= 1600 and
                    total_error < min_error):
                    found_match = True
                    min_error = total_error
                    best_head_error = head_error
                    best_tail_error = tail_error

        if found_match:
            TP += 1

            le.append(best_head_error)
            le.append(best_tail_error)
        else:
            FN += 1





    for detect_seg in detection_segs_1:
        if detect_seg[1]-detect_seg[0] < 1600:
            continue
        is_fp = True

        for tamper_seg in tamper_segs_1:
            if (detect_seg[0] < tamper_seg[1] and
                detect_seg[1] > tamper_seg[0]):

                is_fp = False
                break

        if is_fp:
            FP =1





    if not detection_segs_1:
        long_tamper = False
        for seg in tamper_segs_1:
            if seg[1] - seg[0] > 1600:
                long_tamper = True
                break
        if not long_tamper:
            TN = 1

    return TP, FN, TN, FP, le




def trim_mean_std(values, trim_percent=1):
    sorted_vals, _ = torch.sort(values)
    n = len(sorted_vals)
    k = int(n * trim_percent / 100)
    trimmed_vals = sorted_vals[k:n-k]
    return trimmed_vals.mean(), trimmed_vals.std()

@hydra.main(config_path="../config", config_name="eval")
def run(args):
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    main(args)

if __name__ == "__main__":
    run()
