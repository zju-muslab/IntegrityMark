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
from utils.checkpoints import resolve_checkpoint
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
from pesq import pesq as pesq_fn

from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn


import csv
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast, SDR_Evaluator

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")









def main(eval_args):
    ckpt_path = resolve_checkpoint(eval_args.ckpt_name, eval_args.ckpt_path)
    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    cwd=os.getcwd()
    print(cwd)
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]
    csv_file = os.path.join('experiments',f'{ckpt_file_name}.result.csv')
    csv_file_f = open(csv_file, 'w')
    csv_writer=csv.writer(csv_file_f)
    logfile='experiments/experments.result.log'


    temp_audio_dir = os.path.join('experiments', f'{ckpt_file_name}_watermarked_samples')
    os.makedirs(temp_audio_dir, exist_ok=True)

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

    msg_generator=MsgGenerator(wm_args.wm_msg)

    snr_fn=SISNR()


    results={   'snr':[],
                'pesq':[],
                'stoi':[],
                'det_acc':[],
                'rec_acc':[],
                'val_acc':[],
                'det_sdr': [],
                'rec_sdr': [],
                'val_sdr': []
            }
    csv_str=['file','snr','pesq','stoi','det_acc','rec_acc','val_acc','det_sdr','rec_sdr','val_sdr']

    csv_writer.writerow(csv_str)

    sdr_det=SDR_Evaluator()
    sdr_rec=SDR_Evaluator()
    sdr_val=SDR_Evaluator()

    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_audios)):
            csv_str=[sample['name']]
            x = sample["matrix"].reshape(1,1,-1).to(device)
            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
            msg_rec_seg, msg_val_seg=msg_seg
            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

            x_wm = x+wm


            audio_filename = os.path.join(temp_audio_dir, f"{sample['name'].split('.')[0]}_watermarked.wav")
            torchaudio.save(audio_filename, x_wm[0].cpu(), 16000)

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






            if msg_rec_length > 0:
                res_rec,seg_rec = post_process(wm_pred[1][:,:msg_rec_length,:],window_size=63,return_seg=True)
            else:
                res_rec = torch.zeros(1, 0, wm_pred[1].shape[-1], dtype=torch.long).to(device)
                seg_rec = []

            res_val,seg_val = post_process(wm_pred[1][:,msg_rec_length:,:],window_size=63,return_seg=True)


            pesq=pesq_fn(16000,x[0,0].detach().cpu().numpy(),x_wm[0,0].detach().cpu().numpy())
            snr=-snr_fn(x_wm,x)
            stoi = stoi_fn(x.detach().cpu().numpy().reshape(-1), x_wm.detach().cpu().numpy().reshape(-1), 16000, extended=False)

            det_acc=compute_accuracy(res_det,wm_pred_neg[0],mask=mask)

            rec_acc=compute_bit_acc(res_rec, msg_rec, sigmoid=False)
            val_acc=compute_bit_acc(res_val, msg_val, sigmoid=False)

            curr_sdr_det=sdr_det.update(res_det,msg_det_seg[0])

            curr_sdr_rec = sdr_rec.update(res_rec, msg_rec_seg) if msg_rec_length > 0 else None
            curr_sdr_val=sdr_val.update(res_val,msg_val_seg)

            results['snr'].append(snr.item())
            results['pesq'].append(pesq)
            results['stoi'].append(stoi)
            results['det_acc'].append(det_acc.item())
            results['rec_acc'].append(rec_acc.item())
            results['val_acc'].append(val_acc.item())
            results['det_sdr'].append(curr_sdr_det)
            results['rec_sdr'].append(curr_sdr_rec)
            results['val_sdr'].append(curr_sdr_val)


            csv_str.append(f'{snr.item():.3f}')
            csv_str.append(f'{pesq:.3f}')
            csv_str.append(f'{stoi:.3f}')
            csv_str.append(f'{det_acc.item():.3f}')
            csv_str.append(f'{rec_acc.item():.3f}')
            csv_str.append(f'{val_acc.item():.3f}')
            csv_str.append(f'{curr_sdr_det:.3f}' if curr_sdr_det is not None else '0.000')
            csv_str.append(f'{curr_sdr_rec:.3f}' if curr_sdr_rec is not None else '0.000')
            csv_str.append(f'{curr_sdr_val:.3f}' if curr_sdr_val is not None else '0.000')


            csv_writer.writerow(csv_str)
            csv_file_f.flush()

    csv_str=['stats']
    for key in ['snr', 'pesq', 'stoi', 'det_acc', 'rec_acc', 'val_acc','det_sdr','rec_sdr','val_sdr']:

        values = [v for v in results[key] if v is not None]
        if len(values) > 0:
            mean, std = trim_mean_std(torch.tensor(values))
            csv_str.append(f"{mean.item():.3f}({std:.3f})")
        else:
            csv_str.append("N/A")
    csv_writer.writerow(csv_str)
    csv_file_f.close()

    print("===== Final Metrics =====")
    for key in ['snr', 'pesq', 'stoi', 'det_acc', 'rec_acc', 'val_acc', 'det_sdr', 'rec_sdr', 'val_sdr']:

        values = [v for v in results[key] if v is not None]
        if len(values) > 0:
            mean, std = trim_mean_std(torch.tensor(values))
            print(f"{key}: {mean.item():.3f} (std: {std:.3f})")
        else:
            print(f"{key}: N/A (no valid data)")

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
