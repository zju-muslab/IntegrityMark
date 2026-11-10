
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
import torch

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
from pesq import pesq as pesq_fn

from losses.sisnr import SISNR
from pystoi import stoi as stoi_fn





from utils.audio_tamper_attack import cross_source_replace, cross_source_insert, cross_source_multi_insert, delete, in_source_insert, in_source_replace

import csv
from utils.metric import compute_accuracy, compute_bit_acc, calculate_tiou, calculate_tiou_fast, SDR_Evaluator,calculate_tiou_state1


from utils.in_source_detection import seg_results, get_invalid_points_robust
from statistics import mean

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")






ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_38_2025-02-25_18_43_58.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-27_00_10_39.pth'
ckpt_path='/home/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_40_2025-02-28_03_54_05.pth'


ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_88_2026-01-15_21_34_30.pth'

debug=False

def main(eval_args):





    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'

    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]

    logfile='experiments/in_source_attack_eval/distortion/experments.result.log'

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




    results={}
    csv_fns={}
    csv_writers={}
    stats={}


    for (augmentation_name, augmentation_method) in effects.items():
        csv_file = f'experiments/in_source_attack_eval/distortion/{augmentation_name}.csv'
        csv_fns[augmentation_name]=open(csv_file,'w')
        csv_writers[augmentation_name]=csv.writer(csv_fns[augmentation_name])

        csv_str=['Name', 'TP','TN','FP','FN','ACC','TPR','TNR','FPR','FNR', 'le']
        csv_writers[augmentation_name].writerow(csv_str)
        stats[augmentation_name]={}

        results[augmentation_name] = {
            'TP': [],
            'TN': [],
            'FP': [],
            'FN': [],
            'le': [],
        }




    with torch.no_grad():

        cnt=0

        for i, sample in tqdm(enumerate(val_audios)):

            sample['name'] = f'{i:04d}'


            x = sample["matrix"].reshape(1,1,-1).to(device)


            if x.shape[-1] < 64000:
                cnt+=1
                print('skip', cnt)
                continue


            max_length = 8 * 16_000
            if x.shape[-1] > max_length:

                start = random.randint(0, x.shape[-1] - max_length)
                x = x[:, :, start:start + max_length]


            x = sample["matrix"].reshape(1,1,-1).to(device)

            x = x[..., :x.shape[-1] - (x.shape[-1] % 100)]









            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)





            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

            x_wm = x+wm




            for (augmentation_name, augmentation_method) in effects.items():

                aug_x_wm = augmentation_method(x_wm)








                tamper_aug_x_wm, discrete_points = in_source_replace(aug_x_wm)
                discrete_points.sort()







                wm_pred_tamper = detector(tamper_aug_x_wm)





                res_det_tamper,seg_det_tamper = post_process(wm_pred_tamper[1],window_size=63,return_seg=True)











                tp, fp, tn, fn, le = seg_results(
                    seg_det_tamper[0],
                    discrete_points,
                    manifest=msg_generator.last_manifests[0],
                    master_key=msg_generator.master_key,
                    history_size=getattr(wm_args.wm_msg.val, "history_size", 8),
                    check_window=getattr(wm_args.wm_msg.val, "lookahead_window", 5),
                    tolerated_errors=getattr(wm_args.wm_msg.val, "length_tolerance", 64),
                )



















                results[augmentation_name]['TP'].append(tp)
                results[augmentation_name]['TN'].append(tn)
                results[augmentation_name]['FP'].append(fp)
                results[augmentation_name]['FN'].append(fn)
                results[augmentation_name]['le'].extend(le)






                TP=sum(results[augmentation_name]['TP'])
                TN=sum(results[augmentation_name]['TN'])
                FP=sum(results[augmentation_name]['FP'])
                FN=sum(results[augmentation_name]['FN'])


                ACC = (TP+TN) / (TP+TN+FP+FN)
                TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
                TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
                FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
                FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0




                csv_str=[sample['name']]
                csv_str.append(f'{tp}')
                csv_str.append(f'{tn}')
                csv_str.append(f'{fp}')
                csv_str.append(f'{fn}')
                csv_str.append(f'{ACC:.3f}')
                csv_str.append(f'precision: {precision:.3f}')
                csv_str.append(f'recall:{recall:.3f}')
                csv_str.append(f'f1:{f1:.3f}')
                csv_str.append(f'{TPR:.3f}')
                csv_str.append(f'{TNR:.3f}')
                csv_str.append(f'{FPR:.3f}')
                csv_str.append(f'{FNR:.3f}|')
                csv_str.append(f'{mean(le) if len(le)>0 else -1}')



                csv_writers[augmentation_name].writerow(csv_str)
                csv_fns[augmentation_name].flush()




    for (augmentation_name, augmentation_method) in effects.items():




        TP=sum(results[augmentation_name]['TP'])
        TN=sum(results[augmentation_name]['TN'])
        FP=sum(results[augmentation_name]['FP'])
        FN=sum(results[augmentation_name]['FN'])


        ACC = (TP+TN) / (TP+TN+FP+FN)
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) != 0 else 0




        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0




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




        csv_writers[augmentation_name].writerow(csv_str)
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
