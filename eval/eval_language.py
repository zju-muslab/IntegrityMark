import os
import hydra
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

from tqdm import tqdm
import time
import torch,copy
import torchaudio

import matplotlib.pyplot as plt

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


import csv
from utils.metric import compute_accuracy, compute_bit_acc, SDR_Evaluator

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/temp-chadd-bit0+4+2_095105_256_ft/temp-chadd-bit0+4+2_095105_256_ft_ep_8_2025-01-31_01_20_05.pth'

language_paths={
    # 'ar':'/data0/datasets/CommonVoice/arabic/clips',
    # 'ch':'/data0/datasets/CommonVoice/zh-CN/clips',
    # 'de':'/data0/datasets/CommonVoice/de/clips',
    # 'fr':'/data0/datasets/CommonVoice/fr/clips',
    # 'hi':'/data0/datasets/CommonVoice/hindi/clips',
    # 'ja':'/data0/datasets/CommonVoice/ja/clips',
    # 'kr':'/data0/datasets/CommonVoice/kroean/clips',
    # 'ru':'/data0/datasets/CommonVoice/russian/clips',
    # 'sp':'/data0/datasets/CommonVoice/sp/clips',
    # 'librispeech':'/data0/datasets/LibriSpeech/test-clean',
    # 'voxpupuli':'/data0/datasets/Voxpopuli/unlabelled_data/en',
    # 'timit':'/data0/datasets/TIMIT/TRAIN',
    # 'voxceleb':'/data0/datasets/VoxCeleb1/test',
    # 'vctk':'/data0/datasets/VCTK-Corpus/wav16_resample',
    # 'live':'/home/chenqn/dev/watermark/dataset/multimedia/live',
    # 'podcast':'/home/chenqn/dev/watermark/dataset/multimedia/podcast',
    # 'singing':'/home/chenqn/dev/watermark/dataset/multimedia/singing'

}


def main(eval_args):
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    logfile='experiments/experments.result.log'

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
        csv_file=f'experiments/language_eval/{lan}.csv'
        csv_file_fn=open(csv_file,'w')
        csv_writer=csv.writer(csv_file_fn)
        val_audios=used_dataset(config=eval_args.dataset,flag='test',dataset_path=language_paths[lan],test_num=1000)
        msg_generator=MsgGenerator(wm_args.wm_msg)
        snr_fn=SISNR()
        results={
                'det_acc': [],
                'rec_acc': [],
                'val_acc': [],
                'det_sdr': [],
                'rec_sdr': [],
                'val_sdr': []
        }
        csv_str=['det_acc','rec_acc','val_acc','det_sdr','rec_sdr','val_sdr']
        csv_writer.writerow(csv_str)
        csv_file_fn.flush()
        
        sdr_det=SDR_Evaluator()
        sdr_rec=SDR_Evaluator()
        sdr_val=SDR_Evaluator()

        with torch.no_grad():
            for i, sample in tqdm(enumerate(val_audios)):
                x = sample["matrix"].reshape(1,1,-1).to(device)
                msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
                msg_rec_seg, msg_val_seg=msg_seg
                wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)
                
                x_=copy.deepcopy(x.detach())
                wm_=copy.deepcopy(wm.detach())
                msg_total_=copy.deepcopy(msg_total.detach())
                x_det, wm_det, mask, msg_det_total= crop(x_, wm_, wm_args.crop, msg_total_,return_seg=True)
                x_wm_det = x_det+wm_det
                msg_det_seg = sequence_to_segments_fast(mask)

                csv_str=[sample['name']]
                wm_pred_det = detector(x_wm_det)
                res_det,seg_det = post_process(wm_pred_det[0],window_size=63,return_seg=True)

                # wm_pred = detector(x_wm)

                x_wm = x+wm
                wm_pred_neg = detector(x)
                wm_pred = detector(x_wm)            

                res_rec,seg_rec = post_process(wm_pred[1][:,:msg_rec_length,:],window_size=63,return_seg=True)
                res_val,seg_val = post_process(wm_pred[1][:,msg_rec_length:,:],window_size=63,return_seg=True)
                
                # pesq=pesq_fn(16000,x[0,0].detach().cpu().numpy(),x_wm[0,0].detach().cpu().numpy())
                # snr=-snr_fn(x_wm,x)
                # stoi = stoi_fn(x.detach().cpu().numpy().reshape(-1), x_wm.detach().cpu().numpy().reshape(-1), 16000, extended=False)

                det_acc=compute_accuracy(res_det,wm_pred_neg[0],mask=mask)
                rec_acc=compute_bit_acc(res_rec, msg_total[:,:-msg_val_length], sigmoid=False)
                val_acc=compute_bit_acc(res_val, msg_total[:,-msg_val_length:], sigmoid=False)

                # def update(self, pred_scores, ground_truth_segments):
                curr_sdr_det=sdr_det.update(res_det,msg_det_seg[0])
                curr_sdr_rec=sdr_rec.update(res_rec,msg_rec_seg)
                curr_sdr_val=sdr_val.update(res_val,msg_val_seg)

                # tiou_det, _=calculate_tiou_by_sample(mask, res_det)
                # tiou_rec, _=calculate_tiou_by_sample(msg_total[:,:-msg_val_length], res_rec)
                # tiou_val, _=calculate_tiou_by_sample(msg_total[:,-msg_val_length:], res_val)

                # tiou_det=calculate_tiou_fast(msg_det_seg[0],seg_det[0])
                # tiou_rec=calculate_tiou_fast(msg_rec_seg,seg_rec[0])
                # tiou_val=calculate_tiou_fast(msg_val_seg,seg_val[0])

                results['det_acc'].append(det_acc.item())
                results['rec_acc'].append(rec_acc.item())
                results['val_acc'].append(val_acc.item())
                results['det_sdr'].append(curr_sdr_det)
                results['rec_sdr'].append(curr_sdr_rec)
                results['val_sdr'].append(curr_sdr_val)     
            
                csv_str.append(f'{det_acc.item():.3f}')
                csv_str.append(f'{rec_acc.item():.3f}')
                csv_str.append(f'{val_acc.item():.3f}')
                csv_str.append(f'{curr_sdr_det:.3f}')
                csv_str.append(f'{curr_sdr_rec:.3f}')
                csv_str.append(f'{curr_sdr_val:.3f}')
            

                csv_writer.writerow(csv_str)
                csv_file_fn.flush()


            csv_str = ['stats']
            for key in ['det_acc', 'rec_acc', 'val_acc','det_sdr','rec_sdr','val_sdr']: 
                mean, std = trim_mean_std(torch.tensor(results[key]))
                csv_str.append(f"{mean.item():.3f}({std:.3f})")

            # csv_str.append(f"{sdr_det.get_sdr():.3f}")
            # csv_str.append(f"{sdr_rec.get_sdr():.3f}")
            # csv_str.append(f"{sdr_val.get_sdr():.3f}")

            csv_writer.writerow(csv_str)
            csv_file_fn.flush()

def trim_mean_std(values, trim_percent=1):
    """
    Calculate mean and std after removing top and bottom percentiles
    
    Args:
        values: tensor of values
        trim_percent: percentage to trim from each end (default 5%)
    
    Returns:
        mean, std of trimmed values
    """
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