
import os
import hydra
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

from tqdm import tqdm
import time
import torch
import torchaudio

from distortions.audio_effects import get_audio_effects
from dataset.data import wav_dataset as used_dataset

from losses.sisnr import SISNR
from pesq import pesq as pesq_fn

import csv

from audioseal import AudioSeal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







GENERATOR_PATH = '../audioseal/checkpoints/checkpoint_generator_test.pth'
DETECTOR_PATH = '../audioseal/checkpoints/checkpoint_detector_test.pth'
NBITS = 16


def compute_detection_acc(detection_output, threshold=0.5):
    if detection_output.dim() == 3:

        prob = detection_output.mean(dim=-1).mean(dim=-1)
    else:
        prob = detection_output

    acc = (prob > threshold).float().mean()
    return acc


def compute_bit_acc(decoded_msg, original_msg):
    if decoded_msg.dim() == 3:

        decoded_msg = decoded_msg.mean(dim=-1)


    decoded_bits = (decoded_msg > 0).float()
    original_bits = original_msg.float()

    acc = (decoded_bits == original_bits).float().mean()
    return acc


def main(eval_args):




    os.makedirs('experiments/distortion_eval_audioseal', exist_ok=True)


    logfile = 'experiments/distortion_eval_audioseal/experments.result.log'
    logger = get_logger(log_file=logfile)
    logger.info(f'=' * 80)
    logger.info(f'开始 AudioSeal 失真鲁棒性评估')
    logger.info(f'时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    logger.info(f'=' * 80)



    logger.info(f'\n加载 AudioSeal 模型...')


    try:
        generator = AudioSeal.load_generator(GENERATOR_PATH, nbits=NBITS)
        detector = AudioSeal.load_detector(DETECTOR_PATH, nbits=NBITS)
        logger.info(f'✓ 从自定义路径加载模型')
        logger.info(f'  Generator: {GENERATOR_PATH}')
        logger.info(f'  Detector: {DETECTOR_PATH}')
    except:
        generator = AudioSeal.load_generator("audioseal_wm_16bits")
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        logger.info(f'✓ 使用预训练模型: audioseal_wm_16bits / audioseal_detector_16bits')

    generator = generator.to(device)
    detector = detector.to(device)
    generator.eval()
    detector.eval()

    logger.info(f'✓ 模型加载成功，消息位数: {NBITS}')




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


    snr_fn = SISNR()



    results = {}
    csv_fns = {}
    csv_writers = {}
    stats = {}

    for augmentation_name in effects.keys():

        csv_file = f'experiments/distortion_eval_audioseal/{augmentation_name}.csv'
        csv_fns[augmentation_name] = open(csv_file, 'w')
        csv_writers[augmentation_name] = csv.writer(csv_fns[augmentation_name])


        csv_str = ['file', 'det_acc', 'rec_acc', 'snr', 'pesq']
        csv_writers[augmentation_name].writerow(csv_str)


        stats[augmentation_name] = {}
        results[augmentation_name] = {
            'det_acc': [],
            'rec_acc': [],
            'snr': [],
            'pesq': []
        }



    logger.info(f'\n开始评估...\n')

    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_audios), total=len(val_audios), desc="评估进度"):


            x = sample["matrix"].reshape(1, 1, -1).to(device)


            if x.shape[-1] < 16000:
                continue



            msg = torch.randint(0, 2, (1, NBITS), device=device).float()


            wm = generator.get_watermark(x, sample_rate=16000, message=msg)
            x_wm = x + wm


            try:
                orig_snr = -snr_fn(x_wm, x).item()
                orig_pesq = pesq_fn(16000,
                                    x[0, 0].cpu().numpy()[:48000],
                                    x_wm[0, 0].cpu().numpy()[:48000])
            except:
                orig_snr = 0.0
                orig_pesq = 0.0


            for augmentation_name, augmentation_method in effects.items():
                csv_str = [sample['name']]


                aug_x_wm = augmentation_method(x_wm)


                min_len = min(aug_x_wm.shape[-1], x_wm.shape[-1])
                aug_x_wm_trimmed = aug_x_wm[..., :min_len]



                try:
                    result, decoded_msg = detector.detect_watermark(aug_x_wm_trimmed, sample_rate=16000)


                    det_acc = (result > 0.5).float().item()


                    rec_acc = compute_bit_acc(decoded_msg, msg).item()

                except Exception as e:

                    try:
                        output = detector.detector(aug_x_wm_trimmed)


                        det_output = output[:, :2, :]
                        det_acc = (det_output[:, 1, :].mean() > 0.5).float().item()


                        msg_output = output[:, 2:, :]
                        rec_acc = compute_bit_acc(msg_output, msg).item()
                    except:
                        det_acc = 0.0
                        rec_acc = 0.5


                results[augmentation_name]['det_acc'].append(det_acc)
                results[augmentation_name]['rec_acc'].append(rec_acc)
                results[augmentation_name]['snr'].append(orig_snr)
                results[augmentation_name]['pesq'].append(orig_pesq)


                csv_str.append(f'{det_acc:.3f}')
                csv_str.append(f'{rec_acc:.3f}')
                csv_str.append(f'{orig_snr:.3f}')
                csv_str.append(f'{orig_pesq:.3f}')

                csv_writers[augmentation_name].writerow(csv_str)
                csv_fns[augmentation_name].flush()



    logger.info(f'\n' + '=' * 80)
    logger.info(f'评估完成！计算统计信息...')
    logger.info(f'=' * 80 + '\n')

    for augmentation_name in effects.keys():
        csv_str = ['stats']
        stats_dict = {}

        for key in ['det_acc', 'rec_acc', 'snr', 'pesq']:
            values = torch.tensor(results[augmentation_name][key])
            if len(values) > 0:
                mean_val = values.mean().item()
                std_val = values.std().item()
            else:
                mean_val, std_val = 0.0, 0.0
            csv_str.append(f"{mean_val:.3f}({std_val:.3f})")
            stats_dict[key] = (mean_val, std_val)

        csv_writers[augmentation_name].writerow(csv_str)
        stats[augmentation_name] = stats_dict
        logger.info(f'{augmentation_name}: {csv_str}')



    print("\n" + "=" * 80)
    print("AudioSeal 失真鲁棒性评估结果汇总".center(80))
    print("=" * 80)

    print(f"\n{'失真类型':<25} {'检测准确率':<15} {'消息恢复率':<15} {'SNR':<12} {'PESQ':<12}")
    print("-" * 80)

    for augmentation_name in sorted(effects.keys()):
        s = stats[augmentation_name]
        print(f"{augmentation_name:<25} "
              f"{s['det_acc'][0]:.3f}±{s['det_acc'][1]:.3f}   "
              f"{s['rec_acc'][0]:.3f}±{s['rec_acc'][1]:.3f}   "
              f"{s['snr'][0]:.2f}±{s['snr'][1]:.2f}   "
              f"{s['pesq'][0]:.2f}±{s['pesq'][1]:.2f}")

    print("-" * 80)


    avg_stats = {}
    for key in ['det_acc', 'rec_acc']:
        all_means = [stats[aug][key][0] for aug in effects.keys()]
        avg_stats[key] = sum(all_means) / len(all_means) if all_means else 0.0

    print(f"{'平均性能':<25} "
          f"{avg_stats['det_acc']:.3f}            "
          f"{avg_stats['rec_acc']:.3f}")

    print("=" * 80)

























@hydra.main(config_path="../config", config_name="eval")
def run(args):
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    main(args)


if __name__ == "__main__":
    run()
