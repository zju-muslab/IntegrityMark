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
from pesq import pesq as pesq_fn, PesqError

from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn

snr_fn=SISNR()

from utils.audio_tamper_attack import cross_source_replace, cross_source_insert, cross_source_multi_insert

import csv
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast, SDR_Evaluator,calculate_tiou_state1

from utils.cross_source_detection import evaluate_tampering_overlap_ratio

from statistics import mean

from audioseal import AudioSeal
import torchaudio
import torch
import IPython.display as ipd
import matplotlib.pyplot as plt
import copy,os
import soundfile
from tqdm import tqdm
import numpy as np
from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn, PesqError


# model = AudioSeal.load_generator("audioseal_wm_16bits")
# detector = AudioSeal.load_detector("audioseal_detector_16bits")
model = AudioSeal.load_generator('../audioseal/checkpoints/checkpoint_generator_test.pth', nbits=16)
detector = AudioSeal.load_detector('../audioseal/checkpoints/checkpoint_detector_test.pth', nbits=16)
device='cuda'
model.to(device)
detector.to(device)
print()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_38_2025-02-25_18_43_58.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-27_00_10_39.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-28_03_54_05.pth'
# ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+2+4_32bps/0+2+4_32bps_ep_21_2025-02-22_20_52_37.pth'
ckpt_path = '/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_49_2025-03-01_18_45_58.pth'
def main(eval_args):
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    logfile='experiments/cross_source_attack_eval/experments.result.audioseal.log'
    
    logger = get_logger(log_file=logfile)
    logger.info(f'time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    wm_args = omegaconf.OmegaConf.load(ckpt_config)

    msg_fix_length=wm_args.wm_msg.fix.length
    msg_rec_length=wm_args.wm_msg.temp.length
    msg_val_length=wm_args.wm_msg.val.length

    # embedder = WMEmbedder(model_config=wm_args.model, klass=wm_args.embedder,  msg_fix_nbits=msg_fix_length, msg_temp_nbits=msg_rec_length, msg_val_nbits=msg_val_length).to(device)
    # detector = WMExtractor(model_config=wm_args.model, klass=wm_args.detector, output_dim=32, nbits=msg_fix_length+msg_rec_length+msg_val_length).to(device)


    # module_state_dict=load_ckpt(ckpt_path)
    # embedder.load_state_dict(module_state_dict['embedder'],strict=False)
    # detector.load_state_dict(module_state_dict['detector'],strict=False)

    val_audios = used_dataset(config=eval_args.dataset,flag='test')

    cross_source_audio_path = '/data0/datasets/VCTK-Corpus/wav16_resample'
    cross_source_audios=used_dataset(config=eval_args.dataset,flag='test',dataset_path=cross_source_audio_path,test_num=10000)


    msg_generator=MsgGenerator(wm_args.wm_msg)

    results={}
    csv_fns={}
    csv_writers={}
    stats={}
    for tamper_name in ['cross_source_insert', 'cross_source_replace', 'cross_source_multi_insert']:
        csv_file = f'experiments/cross_source_attack_eval/{tamper_name}_audioseal.csv'
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
            # wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

            msg = torch.Tensor([[0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0]]).to(x.device)
            wm = model.get_watermark(x.to(device), 16000, message=msg.to(device))

            x_wm = x+wm

            pesq=pesq_fn(16000,x[0,0].detach().cpu().numpy(),x_wm[0,0].detach().cpu().numpy())
            snr=-snr_fn(x_wm,x)
            stoi = stoi_fn(x.detach().cpu().numpy().reshape(-1), x_wm.detach().cpu().numpy().reshape(-1), 16000, extended=False)

            dataset_idx = random.randint(0, len(cross_source_audios)-1)
            tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)  # 假设音频在第一个位置
            while tamper_audio.shape[-1] < 32000:
                dataset_idx = random.randint(0, len(cross_source_audios)-1)
                tamper_audio = cross_source_audios[dataset_idx]["matrix"].reshape(1,1,-1).to(device)  # 假设音频在第一个位置

            # for tamper_name in ['cross_source_insert', 'cross_source_replace', 'cross_source_multi_insert']:
            for tamper_name in [ 'cross_source_replace', 'cross_source_multi_insert']:
                if tamper_name == 'cross_source_insert':
                    tampered_x_wm, tamper_segs = cross_source_insert(x_wm, tamper_audio)
                elif tamper_name == 'cross_source_replace':
                    tampered_x_wm, tamper_segs = cross_source_replace(x_wm, tamper_audio)
                elif tamper_name == 'cross_source_multi_insert':
                    tampered_x_wm, tamper_segs = cross_source_multi_insert(x_wm, tamper_audio)


                # get_mask
                tamper_segs.sort(key=lambda x: x[0])
                # wm_pred_tamper = detector(tampered_x_wm)
                # wm_pred_neg = detector(x_wm)

                wm_pred_tamper=detector.detector(tampered_x_wm.reshape(1,1,-1).to('cuda'))[0][0].reshape(1,1,-1)*(-1)
                
                
                res_det_tamper,seg_det_tamper = post_process(wm_pred_tamper,window_size=63,return_seg=True)

                for i, seg in enumerate(seg_det_tamper[0]):
                    seg_det_tamper[0][i] = [seg[0], seg[1], 1-seg[2][0]]
                
                # wm_pred_neg=detector.detector(x_wm.reshape(1,1,-1).to('cuda'))[0][0].reshape(1,1,-1)*-1
                # res_det_neg,seg_det_neg = post_process(wm_pred_neg,window_size=63,return_seg=True)
                # for i, seg in enumerate(seg_det_neg[0]):
                #     seg_det_neg[0][i] = [seg[0], seg[1], 1-seg[2][0]]

                # print('---')
                # print(tamper_segs)
                # print(label_neg)
                precesion, recall, f1, toe = evaluate_tampering_overlap_ratio(seg_det_tamper[0], tamper_segs)

                toe/=16000
                

                # tp1, fn1, tn1, fp1, le = seg_results(seg_det_tamper[0], tamper_segs)
                # tp2, fn2, tn2, fp2, _ = seg_results(seg_det_neg[0], label_neg)
                
                # if fn1 !=0 :
                #     print('---')
                #     print(sample['audio_path'])
                #     print(seg_det_tamper[0])
                #     print(seg_det_neg[0])

                # tp=tp1+tp2
                # tn=tn1+tn2
                # fp=fp1+fp2
                # fn=fn1+fn2
                
                # tiou = calculate_tiou_state1(tamper_segs, seg_det_tamper[0])



                results[tamper_name]['SNR'].append(snr.item())
                results[tamper_name]['PESQ'].append(pesq)
                results[tamper_name]['STOI'].append(stoi)
                results[tamper_name]['precesion'].append(precesion)
                results[tamper_name]['recall'].append(recall)
                results[tamper_name]['f1'].append(f1)
                if toe<0.2:
                    results[tamper_name]['toe'].append(toe) 


                # results[tamper_name]['TP'].append(tp)
                # results[tamper_name]['TN'].append(tn)
                # results[tamper_name]['FP'].append(fp)
                # results[tamper_name]['FN'].append(fn)
                # results[tamper_name]['le'].extend(le)
                # results[tamper_name]['tiou'].append(tiou)

                # TP=sum(results[tamper_name]['TP'])
                # TN=sum(results[tamper_name]['TN'])
                # FP=sum(results[tamper_name]['FP'])
                # FN=sum(results[tamper_name]['FN'])

                # ACC = (TP+TN) / (TP+TN+FP+FN)
                # TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
                # TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
                # FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
                # FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

                # precesion = TP / (TP + FP) if (TP + FP) != 0 else 0
                # recall = TP / (TP + FN) if (TP + FN) != 0 else 0

                mean_precesion=np.mean(results[tamper_name]['precesion'])
                mean_recall=np.mean(results[tamper_name]['recall'])
                mean_f1=np.mean(results[tamper_name]['f1'])
                mean_toe=np.mean(results[tamper_name]['toe'])

                csv_str=[sample['name']]
                # csv_str.append(f'{tp}')
                # csv_str.append(f'{tn}')
                # csv_str.append(f'{fp}')
                # csv_str.append(f'{fn}')
                csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
                csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
                csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
                csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')
                # csv_str.append(f'acc:{ACC:.3f}')
                # csv_str.append(f'tpr:{TPR:.3f}')
                # csv_str.append(f'tnr:{TNR:.3f}')
                # csv_str.append(f'fpr:{FPR:.3f}')
                # csv_str.append(f'fnr:{FNR}')
                # csv_str.append(f'{mean(le) if len(le)>0 else -1}')
                # csv_str.append(f'{tiou:.3f}')
                csv_str.append(f'{snr.item():.3f}')
                csv_str.append(f'{pesq:.3f}')
                csv_str.append(f'{stoi:.3f}')
            
                csv_writers[tamper_name].writerow(csv_str)
                csv_fns[tamper_name].flush()

    for tamper_name in ['cross_source_insert', 'cross_source_replace', 'cross_source_multi_insert']:

        # TP=sum(results[tamper_name]['TP'])
        # TN=sum(results[tamper_name]['TN'])
        # FP=sum(results[tamper_name]['FP'])
        # FN=sum(results[tamper_name]['FN'])

        # ACC = (TP+TN) / (TP+TN+FP+FN)
        # TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        # TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
        # FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        # FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

        # precesion = TP / (TP + FP) if (TP + FP) != 0 else 0
        # recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        

        mean_precesion=np.mean(results[tamper_name]['precesion'])
        mean_recall=np.mean(results[tamper_name]['recall'])
        mean_f1=np.mean(results[tamper_name]['f1'])
        mean_toe=np.mean(results[tamper_name]['toe'])

        # LE = mean(results[tamper_name]["le"])
        # TIOU=mean(results[tamper_name]["tiou"])

        SNR=mean(results[tamper_name]['SNR'])
        PESQ=mean(results[tamper_name]['PESQ'])
        STOI=mean(results[tamper_name]['STOI'])


        csv_str = [f'{tamper_name}']
        # csv_str.append(f'TP={TP}')
        # csv_str.append(f'TN={TN}')
        # csv_str.append(f'FP={FP}')
        # csv_str.append(f'FN={FN}')
        csv_str.append(f'pre:{precesion:.3f}({mean_precesion:.3f})')
        csv_str.append(f'recall:{recall:.3f}({mean_recall:.3f})')
        csv_str.append(f'f1:{f1:.3f}({mean_f1:.3f})')
        csv_str.append(f'toe:{toe:.5f}({mean_toe:.5f})')
        # csv_str.append(f'ACC={ACC:.3f}')
        # csv_str.append(f'TPR={TPR:.3f}')
        # csv_str.append(f'TNR={TNR:.3f}')
        # csv_str.append(f'FPR={FPR:.3f}')
        # csv_str.append(f'FNR={FNR:.3f}')
        # csv_str.append(f'LE={LE:.3f}')
        # csv_str.append(f'TIOU={TIOU:.3f}')
        csv_str.append(f'SNR={SNR:.3f}')
        csv_str.append(f'PESQ={PESQ:.3f}')
        csv_str.append(f'STOI={STOI:.3f}')
        csv_writers[tamper_name].writerow(csv_str)
        logger.info(csv_str)


def seg_results(detection_segs, tamper_segs):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    le = []  # 用于存储边界误差
    
    # 提取所有状态为1的段

    tamper_segs_1 = [seg for seg in tamper_segs if seg[2] == 1]
    detection_segs_1 = [seg for seg in detection_segs if seg[2] == 1]
    # 计算TP和FN
    for tamper_seg in tamper_segs_1:
        found_match = False
        tamper_duration = tamper_seg[1] - tamper_seg[0]
        min_error = float('inf')  # 记录最小误差
        best_head_error = 1600
        best_tail_error = 1600
        
        for detect_seg in detection_segs_1:
            # 计算重叠部分
            overlap_start = max(tamper_seg[0], detect_seg[0])
            overlap_end = min(tamper_seg[1], detect_seg[1])
            
            if overlap_end > overlap_start:  # 有重叠
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
            # 记录实际的边界误差
            le.append(best_head_error)
            le.append(best_tail_error)
        else:
            FN +=1
            # FN情况记录最大误差1600
            # le.append(1600)
            # le.append(1600)
    
    # 计算FP
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
            

            FP += 1
            # FP情况记录最大误差1600
            # le.append(1600)
            # le.append(1600)
    
    # 计算TN
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