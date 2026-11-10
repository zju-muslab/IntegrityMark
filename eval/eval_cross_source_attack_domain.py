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
import numpy as np
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

from statistics import mean

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_88_2026-01-15_21_34_30.pth'


language_paths={
    'en':'/data0/datasets/CommonVoice/en/clips',
    'ar':'/data0/datasets/CommonVoice/arabic/clips',
    'ch':'/data0/datasets/CommonVoice/zh-CN/clips',
    'de':'/data0/datasets/CommonVoice/de/clips',
    'fr':'/data0/datasets/CommonVoice/fr/clips',
    'hi':'/data0/datasets/CommonVoice/hindi/clips',
    'ja':'/data0/datasets/CommonVoice/ja/clips',
    'kr':'/data0/datasets/CommonVoice/kroean/clips',
    'ru':'/data0/datasets/CommonVoice/russian/clips',
    'sp':'/data0/datasets/CommonVoice/sp/clips',
    'librispeech':'/data0/datasets/LibriSpeech/test-clean',
    'voxpupuli':'/data0/datasets/Voxpopuli/unlabelled_data/en',
    'timit':'/data0/datasets/TIMIT/TRAIN',
    'voxceleb':'/data0/datasets/VoxCeleb1/test',
    'vctk':'/data0/datasets/VCTK-Corpus/wav16_resample',
    'live':'/home/chenqn/dev/watermark/dataset/multimedia/live',
    'podcast':'/home/chenqn/dev/watermark/dataset/multimedia/podcast',
    'singing':'/home/chenqn/dev/watermark/dataset/multimedia/singing'
}


def main(eval_args):
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    logfile='experiments/cross_source_attack_eval/domain/experments.result.log'

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

    for lan in  language_paths:
        csv_file=f'experiments/cross_source_attack_eval/domain/{lan}.csv'
        csv_file_fn=open(csv_file,'w')
        csv_writer=csv.writer(csv_file_fn)
        val_audios = used_dataset(config=eval_args.dataset,flag='test',dataset_path=language_paths[lan],test_num=1000)
        msg_generator=MsgGenerator(wm_args.wm_msg)
        results={
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
        csv_str=['TP','TN','FP','FN','ACC','TPR','TNR','FPR','FNR', 'le','tiou']
        csv_writer.writerow(csv_str)
        csv_file_fn.flush()

        cross_source_audio_path = '/data0/datasets/VCTK-Corpus/wav16_resample'
        cross_source_audios=used_dataset(config=eval_args.dataset,flag='test',dataset_path=cross_source_audio_path,test_num=10000)

        with torch.no_grad():
            for i, sample in tqdm(enumerate(val_audios)):
                sample['name'] = f'{i:04d}'
                x = sample["matrix"].reshape(1,1,-1).to(device)
                x = x[..., :x.shape[-1] - (x.shape[-1] % 100)]
                msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
                wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

                dataset_idx = random.randint(0, len(cross_source_audios)-1)
                tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)
                while tamper_audio.shape[-1] < 32000:
                    dataset_idx = random.randint(0, len(cross_source_audios)-1)
                    tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)

                x_wm = x+wm
                tamper_x_wm, tamper_segs = cross_source_insert(x_wm, tamper_audio)
                tamper_segs.sort(key=lambda x: x[0])

                wm_pred_tamper = detector(tamper_x_wm)


                res_det_tamper,seg_det_tamper = post_process(wm_pred_tamper[0],window_size=63,return_seg=True)

                for i, seg in enumerate(seg_det_tamper[0]):
                    seg_det_tamper[0][i] = [seg[0], seg[1], 1-seg[2][0]]







                precesion, recall, f1, toe = evaluate_tampering_overlap_ratio(seg_det_tamper[0], tamper_segs)

                toe=np.clip(toe/16000,0,0.1)
































                results['precesion'].append(precesion)
                results['recall'].append(recall)
                results['f1'].append(f1)
                results['toe'].append(toe)


                mean_precesion=np.mean(results['precesion'])
                mean_recall=np.mean(results['recall'])
                mean_f1=np.mean(results['f1'])
                mean_toe=np.mean(results['toe'])







                csv_str = [f'{sample["path"]}']




                csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
                csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
                csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
                csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')









                csv_writer.writerow(csv_str)
                csv_file_fn.flush()






        mean_precesion=np.mean(results['precesion'])
        mean_recall=np.mean(results['recall'])
        mean_f1=np.mean(results['f1'])
        mean_toe=np.mean(results['toe'])












        csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
        csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
        csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
        csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')















        csv_writer.writerow(csv_str)
        logger.info(csv_str)


















































































def seg_results(detection_segs, tamper_segs):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    le = []


    tamper_segs_1 = [seg for seg in tamper_segs if seg[2] == 1]


    detection_segs_1 = [seg for seg in detection_segs if seg[2] == 1 and seg[1] - seg[0] >= 3200]


    tp_detection_indices = set()


    for tamper_seg in tamper_segs_1:
        found_match = False
        min_error = float('inf')
        best_index = -1
        best_head_error = float('inf')
        best_tail_error = float('inf')


        for i, detect_seg in enumerate(detection_segs_1):

            if detect_seg[0] < tamper_seg[1] and detect_seg[1] > tamper_seg[0]:
                head_error = abs(tamper_seg[0] - detect_seg[0])
                tail_error = abs(tamper_seg[1] - detect_seg[1])
                total_error = head_error + tail_error

                if total_error < min_error:
                    min_error = total_error
                    best_index = i
                    best_head_error = head_error
                    best_tail_error = tail_error
                    found_match = True


        if found_match and best_head_error < 1600 and best_tail_error < 1600:
            TP += 1
            tp_detection_indices.add(best_index)
            le.append(best_head_error)
            le.append(best_tail_error)
        else:
            FN += 1


    FP = len(detection_segs_1) - len(tp_detection_indices)


    if not detection_segs_1 and not tamper_segs_1:
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
