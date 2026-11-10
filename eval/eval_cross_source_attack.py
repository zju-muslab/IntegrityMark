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






ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4 wobound/0+0+4 wobound_ep_73_2026-01-15_16_54_36.pth'
def main(eval_args):
    print(f"loading ckpt from {ckpt_path}")
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    logfile='experiments/cross_source_attack_eval/experments.result_test.log'

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

    cross_source_audio_path = '/data0/datasets/VCTK-Corpus/wav16_resample'
    cross_source_audios=used_dataset(config=eval_args.dataset,flag='test',dataset_path=cross_source_audio_path,test_num=10000)


    msg_generator=MsgGenerator(wm_args.wm_msg)

    results={}
    csv_fns={}
    csv_writers={}
    stats={}
    for tamper_name in ['cross_source_insert', 'cross_source_replace', 'cross_source_multi_insert']:
        csv_file = f'experiments/cross_source_attack_eval/{tamper_name}.csv'
        csv_fns[tamper_name]=open(csv_file,'w')
        csv_writers[tamper_name]=csv_writer=csv.writer(csv_fns[tamper_name])
        csv_str=['Name', 'TP','TN','FP','FN','ACC','TPR','TNR','FPR','FNR', 'le','tiou']
        csv_writers[tamper_name].writerow(csv_str)
        stats[tamper_name]={}
        results[tamper_name] = {
            'PESQ': [],
            'SNR': [],
            'STOI': [],
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
            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
            msg_rec_seg, msg_val_seg=msg_seg
            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

            x_wm = x+wm

            pesq=pesq_fn(16000,x[0,0].detach().cpu().numpy(),x_wm[0,0].detach().cpu().numpy())
            snr=-snr_fn(x_wm,x)
            stoi = stoi_fn(x.detach().cpu().numpy().reshape(-1), x_wm.detach().cpu().numpy().reshape(-1), 16000, extended=False)

            dataset_idx = random.randint(0, len(cross_source_audios)-1)
            tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)
            while tamper_audio.shape[-1] < 32000:
                dataset_idx = random.randint(0, len(cross_source_audios)-1)
                tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)


            for tamper_name in [ 'cross_source_replace', 'cross_source_multi_insert']:
                if tamper_name == 'cross_source_insert':
                    tampered_x_wm, tamper_segs = cross_source_insert(x_wm, tamper_audio)
                elif tamper_name == 'cross_source_replace':
                    tampered_x_wm, tamper_segs = cross_source_replace(x_wm, tamper_audio)
                elif tamper_name == 'cross_source_multi_insert':
                    tampered_x_wm, tamper_segs = cross_source_multi_insert(x_wm, tamper_audio)



                tamper_segs.sort(key=lambda x: x[0])
                wm_pred_tamper = detector(tampered_x_wm)

                res_det_tamper,seg_det_tamper = post_process(wm_pred_tamper[0],window_size=63,return_seg=True)

                for i, seg in enumerate(seg_det_tamper[0]):
                    seg_det_tamper[0][i] = [seg[0], seg[1], 1-seg[2][0]]









                precesion, recall, f1, toe = evaluate_tampering_overlap_ratio(seg_det_tamper[0], tamper_segs)

                toe=np.clip(toe/16000,0,0.1)




















                results[tamper_name]['SNR'].append(snr.item())
                results[tamper_name]['PESQ'].append(pesq)
                results[tamper_name]['STOI'].append(stoi)
                results[tamper_name]['precesion'].append(precesion)
                results[tamper_name]['recall'].append(recall)
                results[tamper_name]['f1'].append(f1)
                results[tamper_name]['toe'].append(toe)























                mean_precesion=np.mean(results[tamper_name]['precesion'])
                mean_recall=np.mean(results[tamper_name]['recall'])
                mean_f1=np.mean(results[tamper_name]['f1'])
                mean_toe=np.mean(results[tamper_name]['toe'])

                csv_str=[sample['name']]




                csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
                csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
                csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
                csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')







                csv_str.append(f'{snr.item():.3f}')
                csv_str.append(f'{pesq:.3f}')
                csv_str.append(f'{stoi:.3f}')

                csv_writers[tamper_name].writerow(csv_str)
                csv_fns[tamper_name].flush()

    for tamper_name in ['cross_source_replace', 'cross_source_multi_insert']:
















        mean_precesion=np.mean(results[tamper_name]['precesion'])
        mean_recall=np.mean(results[tamper_name]['recall'])
        mean_f1=np.mean(results[tamper_name]['f1'])
        mean_toe=np.mean(results[tamper_name]['toe'])




        SNR=mean(results[tamper_name]['SNR'])
        PESQ=mean(results[tamper_name]['PESQ'])
        STOI=mean(results[tamper_name]['STOI'])


        csv_str = [f'{tamper_name}']




        csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
        csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
        csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
        csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')







        csv_str.append(f'SNR={SNR:.3f}')
        csv_str.append(f'PESQ={PESQ:.3f}')
        csv_str.append(f'STOI={STOI:.3f}')
        csv_writers[tamper_name].writerow(csv_str)
        logger.info(csv_str)



















































































































































































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
