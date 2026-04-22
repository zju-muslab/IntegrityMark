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
from models.WMbuilder import WMEmbedder,WMExtractor
from dataset.data import wav_dataset as used_dataset
import omegaconf
from hydra.experimental import compose, initialize
import matplotlib.pyplot as plt
from utils.wm_process import crop, MsgGenerator, post_process, sequence_to_segments_fast
from pesq import pesq as pesq_fn
from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn
from utils.audio_tamper_attack import cross_source_replace, cross_source_insert, cross_source_multi_insert, delete, in_source_insert, in_source_replace
import csv
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast, SDR_Evaluator,calculate_tiou_state1
from utils.in_source_detection import seg_results, get_invalid_points_robust
from statistics import mean

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'

ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_31_2025-02-26_19_31_56.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-27_00_10_39.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_38_2025-02-27_23_29_20.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_37_2025-02-28_01_05_12.pth' # nice
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-28_03_54_05.pth'
ckpt_path = '/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_49_2025-03-01_18_45_58.pth'
# ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+2+4_32bps/0+2+4_32bps_ep_21_2025-02-22_20_52_37.pth'

debug=False

def main(eval_args):
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    wm_args = omegaconf.OmegaConf.load(ckpt_config)

    msg_fix_length=wm_args.wm_msg.fix.length
    msg_rec_length=wm_args.wm_msg.temp.length
    msg_val_length=wm_args.wm_msg.val.length

    # embedder = WMEmbedder(model_config=wm_args.model, klass=wm_args.embedder,  msg_fix_nbits=msg_fix_length, msg_temp_nbits=msg_rec_length, msg_val_nbits=msg_val_length).to(device)
    detector = WMExtractor(model_config=wm_args.model, klass=wm_args.detector, output_dim=32, nbits=msg_fix_length+msg_rec_length+msg_val_length).to(device)

    module_state_dict=load_ckpt(ckpt_path)
    # embedder.load_state_dict(module_state_dict['embedder'],strict=False)
    detector.load_state_dict(module_state_dict['detector'],strict=False)


    val_audios = used_dataset(config=eval_args.dataset, flag='test')
    # val_audios = used_dataset(config=eval_args.dataset, dataset_path='/data0/datasets/CommonVoice/de/clips',flag='test')

    wm_args.wm_msg.val.d_max=1984
    # msg_generator=MsgGenerator(wm_args.wm_msg)

    # tamper_list=['delete', 'in_source_insert', 'in_source_replace']
    # tamper_list=['in_source_replace']
    results1=[]
    results2=[]
    results3=[]
    with torch.no_grad():
        cnt=0
        for i, sample in tqdm(enumerate(val_audios)):
            x = sample["matrix"]
            if i <10:
                continue
            sample['name'] = f'{i:04d}'
            x = sample["matrix"].reshape(1,1,-1).to(device)
            if x.size(-1)<160000:
                x=x.repeat(1,1,160000//x.size(-1)+1)
            x=x[...,:160000]
            # print(x.shape)
            x=x.repeat(1,1,1)
            t0=time.time()

            wm_pred_neg = detector(x)
            t1=time.time()
            res_det_neg,seg_det_neg = post_process(wm_pred_neg[1].cpu(),window_size=31,return_seg=True)
            label_neg = []
            
            tp2, fp2, tn2, fn2, _ = seg_results(seg_det_neg[0], label_neg, debug=debug)
            t2=time.time()
            results1.append(t1-t0)
            results2.append(t2-t1)
            results3.append(t2-t0)
            
            print(f'time: {sum(results1)/len(results1):.3f}/{sum(results2)/len(results2):.3f}/{sum(results3)/len(results3):.3f}), rtf: {sum(results1)/len(results1)/10:.4f}/{sum(results2)/len(results2)/10:.4f}/{sum(results3)/len(results3)/10:.4f}')
                  


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