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
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/temp-chadd-bit0+4+2_095105_512/temp-chadd-bit0+4+2_095105_512_ep_19_2025-01-24_03_43_10.pth'

def main(eval_args):

    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    csv_file = os.path.join('experiments',f'{ckpt_file_name}.result.csv')
    csv_file_f = open(csv_file, 'w')
    csv_writer=csv.writer(csv_file_f)
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

    val_audios = used_dataset(config=eval_args.dataset,flag='test')

    effects = get_audio_effects(eval_args.augmentation)  # noqa
    if 'speed' in effects:
        del effects['speed']
    if 'echo' in effects:
        del effects['echo']

    msg_generator=MsgGenerator(wm_args.wm_msg)

    snr_fn=SISNR()

    # test start
    results={   'snr':[],
                'pesq':[],
                'stoi':[],
                'det_acc':[],
                'rec_acc':[],
                'val_acc':[],
                'tiou_det':[],
                'tiou_rec':[],
                'tiou_val':[]
            }
            
    csv_str=['file','snr','stoi','pesq,det_acc','rec_acc','val_acc','tiou_det','tiou_rec','tiou_val']
    csv_writer.writerow(csv_str)

    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_audios)):
            csv_str=[sample['name']]
            x = sample["matrix"].reshape(1,1,-1).to(device)
            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
            msg_rec_seg, msg_val_seg=msg_seg
            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

            x_wm = x+wm
        
            wm_pred_neg = detector(x)
            wm_pred = detector(x_wm)

            x_=copy.deepcopy(x.detach())
            wm_=copy.deepcopy(wm.detach())
            msg_total_=copy.deepcopy(msg_total.detach())
            x_det, wm_det, mask, msg_det_total= crop(x_, wm_, wm_args.crop, msg_total_,return_seg=True)
            x_wm_det = x_det+wm_det
            msg_det_seg = sequence_to_segments_fast(mask)
            wm_pred_det = detector(x_wm_det)
            res_det,seg_det = post_process(wm_pred_det[0],window_size=63,return_seg=True)

            # res_det=wm_pred[0]
            # res_rec=wm_pred[1][:,:msg_rec_length,:]
            # res_val=wm_pred[1][:,msg_rec_length:,:]
            
            res_rec,seg_rec = post_process(wm_pred[1][:,:msg_rec_length,:],window_size=63,return_seg=True)
            res_val,seg_val = post_process(wm_pred[1][:,msg_rec_length:,:],window_size=63,return_seg=True)
            # post_res = res

            pesq=pesq_fn(16000,x[0,0].detach().cpu().numpy(),x_wm[0,0].detach().cpu().numpy())
            snr=-snr_fn(x_wm,x)
            stoi = stoi_fn(x.detach().cpu().numpy().reshape(-1), x_wm.detach().cpu().numpy().reshape(-1), 16000, extended=False)

            det_acc=compute_accuracy(res_det,wm_pred_neg[0],mask=mask)
            rec_acc=compute_bit_acc(res_rec, msg_total[:,:-msg_val_length], sigmoid=False)
            val_acc=compute_bit_acc(res_val, msg_total[:,-msg_val_length:], sigmoid=False)

            tiou_det=calculate_tiou_fast(msg_det_seg[0],seg_det[0])
            tiou_rec=calculate_tiou_fast(msg_rec_seg,seg_rec[0])
            tiou_val=calculate_tiou_fast(msg_val_seg,seg_val[0])
            
            results['snr'].append(snr.item())
            results['pesq'].append(pesq)
            results['stoi'].append(stoi)
            results['det_acc'].append(det_acc.item())
            results['rec_acc'].append(rec_acc.item())
            results['val_acc'].append(val_acc.item())
            results['tiou_det'].append(tiou_det)
            results['tiou_rec'].append(tiou_rec)
            results['tiou_val'].append(tiou_val)
            

            csv_str.append(f'{snr.item():.3f}')
            csv_str.append(f'{pesq:.3f}')
            csv_str.append(f'{stoi:.3f}')
            csv_str.append(f'{det_acc.item():.3f}')
            csv_str.append(f'{rec_acc.item():.3f}')
            csv_str.append(f'{val_acc.item():.3f}')
            csv_str.append(f'{tiou_det:.3f}')
            csv_str.append(f'{tiou_rec:.3f}')
            csv_str.append(f'{tiou_val:.3f}')
            

            csv_writer.writerow(csv_str)
            csv_file_f.flush()

    snr_stats=[torch.tensor(results['snr']).mean(),torch.tensor(results['snr']).std()]
    pesq_stats=[torch.tensor(results['pesq']).mean(),torch.tensor(results['pesq']).std()]
    stoi_stats=[torch.tensor(results['stoi']).mean(),torch.tensor(results['stoi']).std()]
    det_acc_stats=[torch.tensor(results['det_acc']).mean(),torch.tensor(results['det_acc']).std()]
    rec_acc_stats=[torch.tensor(results['rec_acc']).mean(),torch.tensor(results['rec_acc']).std()]
    val_acc_stats=[torch.tensor(results['val_acc']).mean(),torch.tensor(results['val_acc']).std()]
    tiou_det_stats=[torch.tensor(results['tiou_det']).mean(),torch.tensor(results['tiou_det']).std()]
    tiou_rec_stats=[torch.tensor(results['tiou_rec']).mean(),torch.tensor(results['tiou_rec']).std()]
    tiou_val_stats=[torch.tensor(results['tiou_val']).mean(),torch.tensor(results['tiou_val']).std()]
    
    
    csv_str=['stats']
    csv_str.append(f'{snr_stats[0].item():.3f}({snr_stats[1].item():.3f})')
    csv_str.append(f'{pesq_stats[0].item():.3f}({pesq_stats[1].item():.3f})')
    csv_str.append(f'{stoi_stats[0].item():.3f}({stoi_stats[1].item():.3f})')
    csv_str.append(f'{det_acc_stats[0].item():.3f}({det_acc_stats[1].item():.3f})')
    csv_str.append(f'{rec_acc_stats[0].item():.3f}({rec_acc_stats[1].item():.3f})')
    csv_str.append(f'{val_acc_stats[0].item():.3f}({val_acc_stats[1].item():.3f})')
    csv_str.append(f'{tiou_det_stats[0].item():.3f}({tiou_det_stats[1].item():.3f})')
    csv_str.append(f'{tiou_rec_stats[0].item():.3f}({tiou_rec_stats[1].item():.3f})')
    csv_str.append(f'{tiou_val_stats[0].item():.3f}({tiou_val_stats[1].item():.3f})')
    csv_writer.writerow(csv_str)

    logger.info(f'{ckpt_path} | ' +\
                f'snr: {snr_stats[0].item():.3f}/{snr_stats[1].item():.3f},' +\
                f'pesq: {pesq_stats[0].item():.3f}/{pesq_stats[1].item():.3f},' +\
                f'stoi: {stoi_stats[0].item():.3f}/{stoi_stats[1].item():.3f},' +\
                f'det_acc: {det_acc_stats[0].item():.3f}/{det_acc_stats[1].item():.3f},' +\
                f'rec_acc: {rec_acc_stats[0].item():.3f}/{rec_acc_stats[1].item():.3f},' +\
                f'val_acc: {val_acc_stats[0].item():.3f}/{val_acc_stats[1].item():.3f},' +\
                f'tiou_det: {tiou_det_stats[0].item():.3f}/{tiou_det_stats[1].item():.3f},' +\
                f'tiou_rec: {tiou_rec_stats[0].item():.3f}/{tiou_rec_stats[1].item():.3f},' +\
                f'tiou_val: {tiou_val_stats[0].item():.3f}/{tiou_val_stats[1].item():.3f},'
    )
    csv_file_f.close()
@hydra.main(config_path="../config", config_name="eval")
def run(args):
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    main(args)

if __name__ == "__main__":
    run()