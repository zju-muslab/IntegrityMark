
import os
import hydra
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

from tqdm import tqdm
import time
import torch, copy
import torchaudio

import matplotlib.pyplot as plt

from utils.tools import load_ckpt, save_ckpt
from itertools import chain
from scipy.stats import multivariate_normal

from distortions.audio_effects import (
    get_audio_effects,
    select_audio_effects,
)
from utils.wm_process import crop, MsgGenerator

from models.WMbuilder import WMEmbedder, WMExtractor
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



ckpt_path = '/mushome/chenqn/dev/watermark/outputs/ckpt/0+0+4/0+0+4_ep_74_2026-01-11_18_37_35.pth'

def main(eval_args):




    ckpt_config = os.path.dirname(ckpt_path) + '/hydra_config.yaml'
    ckpt_file_name = os.path.basename(ckpt_path).split('.')[0]


    logfile = 'experiments/distortion_eval/experments.result.log'
    logger = get_logger(log_file=logfile)
    logger.info(f'=' * 80)
    logger.info(f'开始失真鲁棒性评估')
    logger.info(f'时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    logger.info(f'Checkpoint: {ckpt_path}')
    logger.info(f'=' * 80)


    wm_args = omegaconf.OmegaConf.load(ckpt_config)


    msg_fix_length = wm_args.wm_msg.fix.length
    msg_rec_length = wm_args.wm_msg.temp.length
    msg_val_length = wm_args.wm_msg.val.length

    logger.info(f'\n消息配置: fix={msg_fix_length}, temp={msg_rec_length}, val={msg_val_length}')




    embedder = WMEmbedder(
        model_config=wm_args.model,
        klass=wm_args.embedder,
        msg_fix_nbits=msg_fix_length,
        msg_temp_nbits=msg_rec_length,
        msg_val_nbits=msg_val_length
    ).to(device)


    detector = WMExtractor(
        model_config=wm_args.model,
        klass=wm_args.detector,
        output_dim=32,
        nbits=msg_fix_length + msg_rec_length + msg_val_length
    ).to(device)


    module_state_dict = load_ckpt(ckpt_path)
    embedder.load_state_dict(module_state_dict['embedder'], strict=False)
    detector.load_state_dict(module_state_dict['detector'], strict=False)

    logger.info(f'✓ 模型加载成功')




    val_audios = used_dataset(config=eval_args.dataset, flag='test')
    logger.info(f'✓ 加载测试数据集: {len(val_audios)} 个样本')


    effects = get_audio_effects(eval_args.augmentation)


    effects = {k: v for k, v in effects.items() if k in ['opus_compression']}


    if 'speed' in effects:
        del effects['speed']
    if 'echo' in effects:
        del effects['echo']

    logger.info(f'✓ 加载失真效果: {len(effects)} 种')
    logger.info(f'  失真类型: {", ".join(effects.keys())}')


    msg_generator = MsgGenerator(wm_args.wm_msg)


    snr_fn = SISNR()




    results = {}
    csv_fns = {}
    csv_writers = {}
    stats = {}

    for (augmentation_name, augmentation_method) in effects.items():

        csv_file = f'experiments/distortion_eval/{augmentation_name}.csv'
        csv_fns[augmentation_name] = open(csv_file, 'w')
        csv_writers[augmentation_name] = csv.writer(csv_fns[augmentation_name])


        csv_str = ['file', 'det_acc', 'rec_acc', 'val_acc', 'det_sdr', 'rec_sdr', 'val_sdr']
        csv_writers[augmentation_name].writerow(csv_str)


        stats[augmentation_name] = {}
        results[augmentation_name] = {
            'det_acc': [],
            'rec_acc': [],
            'val_acc': [],
            'det_sdr': [],
            'rec_sdr': [],
            'val_sdr': []
        }


    sdr_evaluator = SDR_Evaluator()



    logger.info(f'\n开始评估...\n')

    with torch.no_grad():

        for i, sample in tqdm(enumerate(val_audios), total=len(val_audios), desc="评估进度"):


            x = sample["matrix"].reshape(1, 1, -1).to(device)




            msg_total, msg_fix, msg_rec, msg_val, msg_seg = msg_generator.msg_generate(x, return_seg=True)
            msg_rec_seg, msg_val_seg = msg_seg



            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)


            x_wm = x + wm



            x_ = copy.deepcopy(x.detach())
            wm_ = copy.deepcopy(wm.detach())
            msg_total_ = copy.deepcopy(msg_total.detach())




            x_det, wm_det, mask, msg_det_total = crop(x_, wm_, wm_args.crop, msg_total_, return_seg=True)
            x_wm_det = x_det + wm_det
            msg_det_seg = sequence_to_segments_fast(mask)


            for (augmentation_name, augmentation_method) in effects.items():
                csv_str = [sample['name']]



                aug_x_wm_det = augmentation_method(x_wm_det)


                wm_pred_det = detector(aug_x_wm_det)




                res_det, seg_det = post_process(wm_pred_det[0], window_size=63, return_seg=True)



                aug_x_wm = augmentation_method(x_wm)




                wm_pred = detector(aug_x_wm)


                wm_pred_neg = detector(x)




                res_rec, seg_rec = post_process(wm_pred[1][:, :msg_rec_length, :], window_size=63, return_seg=True)
                res_val, seg_val = post_process(wm_pred[1][:, msg_rec_length:, :], window_size=63, return_seg=True)




                det_acc = compute_accuracy(res_det, wm_pred_neg[0], mask=mask)


                rec_acc = compute_bit_acc(res_rec, msg_total[:, :-msg_val_length], sigmoid=False)


                val_acc = compute_bit_acc(res_val, msg_total[:, -msg_val_length:], sigmoid=False)



                curr_sdr_det = sdr_evaluator.update(res_det, msg_det_seg[0])
                curr_sdr_rec = sdr_evaluator.update(res_rec, msg_rec_seg)
                curr_sdr_val = sdr_evaluator.update(res_val, msg_val_seg)




                results[augmentation_name]['det_acc'].append(det_acc.item())
                results[augmentation_name]['rec_acc'].append(rec_acc.item())
                results[augmentation_name]['val_acc'].append(val_acc.item())
                results[augmentation_name]['det_sdr'].append(curr_sdr_det)
                results[augmentation_name]['rec_sdr'].append(curr_sdr_rec)
                results[augmentation_name]['val_sdr'].append(curr_sdr_val)


                csv_str.append(f'{det_acc.item():.3f}')
                csv_str.append(f'{rec_acc.item():.3f}')
                csv_str.append(f'{val_acc.item():.3f}')
                csv_str.append(f'{curr_sdr_det:.3f}')
                csv_str.append(f'{curr_sdr_rec:.3f}')
                csv_str.append(f'{curr_sdr_val:.3f}')

                csv_writers[augmentation_name].writerow(csv_str)
                csv_fns[augmentation_name].flush()



    logger.info(f'\n' + '=' * 80)
    logger.info(f'评估完成！计算统计信息...')
    logger.info(f'=' * 80 + '\n')


    for (augmentation_name, augmentation_method) in effects.items():
        csv_str = ['stats']
        stats_dict = {}


        for key in ['det_acc', 'rec_acc', 'val_acc', 'det_sdr', 'rec_sdr', 'val_sdr']:
            mean, std = trim_mean_std(torch.tensor(results[augmentation_name][key]))
            csv_str.append(f"{mean.item():.3f}({std:.3f})")
            stats_dict[key] = (mean.item(), std.item())


        csv_writers[augmentation_name].writerow(csv_str)


        stats[augmentation_name] = stats_dict


        logger.info(f'{augmentation_name}: {csv_str}')



    print("\n" + "=" * 100)
    print("失真鲁棒性评估结果汇总".center(100))
    print("=" * 100)


    print(f"\n{'失真类型':<20} {'检测准确率':<15} {'Temp准确率':<15} {'Val准确率':<15} {'检测SDR':<15} {'TempSDR':<15} {'ValSDR':<15}")
    print("-" * 100)


    for augmentation_name in sorted(effects.keys()):
        s = stats[augmentation_name]
        print(f"{augmentation_name:<20} "
              f"{s['det_acc'][0]:.3f}±{s['det_acc'][1]:.3f}   "
              f"{s['rec_acc'][0]:.3f}±{s['rec_acc'][1]:.3f}   "
              f"{s['val_acc'][0]:.3f}±{s['val_acc'][1]:.3f}   "
              f"{s['det_sdr'][0]:.3f}±{s['det_sdr'][1]:.3f}   "
              f"{s['rec_sdr'][0]:.3f}±{s['rec_sdr'][1]:.3f}   "
              f"{s['val_sdr'][0]:.3f}±{s['val_sdr'][1]:.3f}")

    print("-" * 100)


    avg_stats = {}
    for key in ['det_acc', 'rec_acc', 'val_acc', 'det_sdr', 'rec_sdr', 'val_sdr']:
        all_means = [stats[aug][key][0] for aug in effects.keys()]
        avg_stats[key] = sum(all_means) / len(all_means)

    print(f"{'平均性能':<20} "
          f"{avg_stats['det_acc']:.3f}        "
          f"{avg_stats['rec_acc']:.3f}        "
          f"{avg_stats['val_acc']:.3f}        "
          f"{avg_stats['det_sdr']:.3f}        "
          f"{avg_stats['rec_sdr']:.3f}        "
          f"{avg_stats['val_sdr']:.3f}")

    print("=" * 100)


    print(f"\n{'指标':<15} {'最佳失真':<20} {'最佳值':<10} {'最差失真':<20} {'最差值':<10}")
    print("-" * 75)

    for key, key_name in [
        ('det_acc', '检测准确率'),
        ('rec_acc', 'Temp准确率'),
        ('val_acc', 'Val准确率'),
        ('det_sdr', '检测SDR'),
        ('rec_sdr', 'TempSDR'),
        ('val_sdr', 'ValSDR')
    ]:
        best_aug = max(effects.keys(), key=lambda x: stats[x][key][0])
        worst_aug = min(effects.keys(), key=lambda x: stats[x][key][0])
        best_val = stats[best_aug][key][0]
        worst_val = stats[worst_aug][key][0]

        print(f"{key_name:<15} {best_aug:<20} {best_val:.3f}      {worst_aug:<20} {worst_val:.3f}")

    print("=" * 100)


    print(f"\n详细结果已保存至:")
    print(f"  - CSV文件: experiments/distortion_eval/{{失真类型}}.csv")
    print(f"  - 日志文件: {logfile}")
    print("=" * 100 + "\n")


    for f in csv_fns.values():
        f.close()


def trim_mean_std(values, trim_percent=1):
    sorted_vals, _ = torch.sort(values)
    n = len(sorted_vals)
    k = int(n * trim_percent / 100)
    trimmed_vals = sorted_vals[k:n - k]
    return trimmed_vals.mean(), trimmed_vals.std()


@hydra.main(config_path="../config", config_name="eval")
def run(args):
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    main(args)


if __name__ == "__main__":
    run()
