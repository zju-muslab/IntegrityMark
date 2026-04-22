import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HYDRA_FULL_ERROR'] = '1'

import yaml
import random
import shutil
import wandb
import socket
import hydra
from omegaconf import OmegaConf
import datetime
import warnings
warnings.filterwarnings("ignore")
from utils.logger import get_logger

import torch
import numpy as np
import torchaudio
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.tools import load_ckpt,save_ckpt
from itertools import chain
# from torch.optim.lr_scheduler import StepLR
from scipy.stats import multivariate_normal

import time
from distortions.audio_effects import (
    get_audio_effects,
    select_audio_effects,
)

from utils.wm_process import crop, MsgGenerator

# from My_model.modules import Encoder, Decoder, Discriminator
from models.old_modules.msstft import MultiScaleSTFTDiscriminator

# from model.My_model.privacy_wm import WMEmbedder
# from models.WMBuilder import WMEmbedder, WMExtractor
from models.WMbuilder import WMEmbedder,WMExtractor
from dataset.data import wav_dataset as used_dataset

from losses.loss import Loss
from losses.loss import AdversarialLoss
from losses.sisnr import SISNR
import torch.optim as optim

from pesq import pesq as pesq_fn, PesqError





snr_fn=SISNR()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    lambda_wav = args["optimize"]["lambda_wav"]
    lambda_loud = args["optimize"]["lambda_loud"]
    lambda_msmel = args["optimize"]["lambda_msmel"]
    lambda_wm_det = args["optimize"]["lambda_wm_det"]
    lambda_wm_rec_fix = args["optimize"]["lambda_wm_rec_fix"]
    lambda_wm_rec_temp = args["optimize"]["lambda_wm_rec_temp"]
    lambda_wm_rec_val = args["optimize"]["lambda_wm_rec_val"]
    lambda_adv_d = args["optimize"]["lambda_adv_d"] # modify weights of m and a for better convergence
    lambda_adv_g = args["optimize"]["lambda_adv_g"] # modify weights of m and a for better convergence
    lambda_adv_g_map = args["optimize"]["lambda_adv_g_map"] # modify weights of m and a for better convergence
    # lambda_penalty_det = args["optimize"]["lambda_penalty_det"] # modify weights of m and a for better convergence
    # lambda_penalty_rec = args["optimize"]["lambda_penalty_rec"] # modify weights of m and a for better convergence
    lambda_wav2vec= args["optimize"]["lambda_wav2vec"]

    msg_fix_length = args.wm_msg.fix.length
    msg_temp_length = args.wm_msg.temp.length
    msg_val_length = args.wm_msg.val.length
    lambda_boundary_rec = args["optimize"]["lambda_boundary_rec"]
    


    # ---------------- Init logger
    current_time = datetime.datetime.now().strftime("%y-%m-%d-%H:%M")
    logfile=os.path.join(args.path.log_dir, f"{args.experiment_name}#{current_time}.log")
    logger = get_logger(log_file=logfile)
    
    if args.print_args:
        logger.info(f"Arguments: {OmegaConf.to_yaml(args)}")

    # -------------- load dataset
    train_audios = used_dataset(config=args.dataset, flag='train')
    val_audios = used_dataset(config=args.dataset,flag='test',test_num=100)
    batch_size = args.optimize.batch_size
    assert batch_size < len(train_audios)
    train_audio_loader = DataLoader(train_audios, batch_size=batch_size, shuffle=True, num_workers=8)
    val_audios_loader = DataLoader(val_audios, batch_size=1, shuffle = False)

    # -------------- build model
    logger.info('building model')

    embedder = WMEmbedder(model_config=args.model, klass=args.embedder, msg_fix_nbits=msg_fix_length, msg_temp_nbits=msg_temp_length, msg_val_nbits=msg_val_length).to(device)
    detector = WMExtractor(model_config=args.model, klass=args.detector, output_dim=32, nbits=msg_fix_length+msg_temp_length+msg_val_length).to(device)

    embedder_params = sum(p.numel() for p in embedder.parameters() if p.requires_grad)
    detector_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)

    logger.info(f"Number of parameters in embedder: {embedder_params/1000/1000:.1f} M")
    logger.info(f"Number of parameters in detector: {detector_params/1000/1000:.1f} M")

    if getattr(args, "adv", None):
        discriminator = MultiScaleSTFTDiscriminator().to(device)
        d_op = Adam(
            params=chain(discriminator.parameters()),
            betas=args["optimize"]["betas"],
            eps=args["optimize"]["eps"],
            weight_decay=args["optimize"]["weight_decay"],
            lr = args["optimize"]["lr"]
        )
        adversary = AdversarialLoss(adversary=discriminator, optimizer=d_op)
        discriminator_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logger.info(f"Number of parameters in discriminator: {discriminator_params/1000/1000:.1f} M")

    # -------------- audio effects
    effects = get_audio_effects(args.augmentation)  # noqa

    aug_weights = {}
    for name in effects.keys():
        aug_weights[name] =args.augmentation.aug_weights.get(name, -1)
    augmentations = {**effects}  # noqa

    # -------------- optimizer
    if args.optimize.fix_layers:
        
        for name, param in embedder.named_parameters():
            frozen_layers = [0, 1, 2, 3]
            if any(f'decoder.{i}.' in name for i in frozen_layers):
                param.requires_grad = False
            if 'lstm' in name:
                param.requires_grad = False
            frozen_layers = [1, 2, 3, 4]
            if any(f'encoder.{i}.' in name for i in frozen_layers):
                param.requires_grad = False


    en_de_optim = Adam(
        params = chain(embedder.parameters(), detector.parameters()),
        betas = args["optimize"]["betas"],
        eps = args["optimize"]["eps"],
        weight_decay=args["optimize"]["weight_decay"],
        lr = args["optimize"]["lr"]
    )

#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     en_de_optim, 
#     mode='min',
#     factor=0.1,    # 每次降低为原来的0.1倍
#     patience=10,   # 10个epoch没改善就降低lr
#     min_lr=1e-6    # 最小lr
# )
    
    # -------------- continue from ckpt
    init_epoch = 0
    global_step = -1
    
    if getattr(args, "init_from_ckpt", False):
        logger.info(f"init from ckpt: {args.init_from_ckpt}")
        module_state_dict=load_ckpt(args["init_from_ckpt"])
        for name, param in embedder.state_dict().items():
            if name in module_state_dict['embedder']:
                if param.shape == module_state_dict['embedder'][name].shape:
                    param.copy_(module_state_dict['embedder'][name])
                else:
                    logger.warning(f"Shape mismatch for {name}, skipping this parameter.")
        for name, param in detector.state_dict().items():
            if name in module_state_dict['detector']:
                if param.shape == module_state_dict['detector'][name].shape:
                    param.copy_(module_state_dict['detector'][name])
                else:
                    logger.warning(f"Shape mismatch for {name}, skipping this parameter.")
                    
    if getattr(args, "continue_from_ckpt", False):
        logger.info(f"continue from ckpt: {args['continue_from_ckpt']}")
        module_state_dict=load_ckpt(args["continue_from_ckpt"])
        embedder.load_state_dict(module_state_dict['embedder'],strict=False)
        detector.load_state_dict(module_state_dict['detector'],strict=False)
        # discriminator.load_state_dict(module_state_dict['discriminator'],strict=False)
        en_de_optim.load_state_dict(module_state_dict['en_de_optim'])
        # d_op.load_state_dict(module_state_dict['d_optim'])
        # lr_sched.load_state_dict(module_state_dict['lr_sched'])
        # lr_sched_d.load_state_dict(module_state_dict['lr_sched_d'])
        global_step=module_state_dict['global_step']
        init_epoch=module_state_dict['epoch']+1

    # -------------- Loss 
    loss = Loss(args).to(device)

    # ---------------- train
    logger.info("Begin Training")



    adv_d_loss=0
    adv_g_loss=0

    msg_generator=MsgGenerator(args.wm_msg)

    # eval(args, embedder, detector, val_audios_loader, loss, logger, ep=init_epoch, global_step=global_step)

    for ep in range(init_epoch, args.iter.max_epoch):
        embedder.train()
        detector.train()
        discriminator.train()
        
        logger.info('Epoch {}/{}'.format(ep, args.iter.max_epoch))
        train_audio_loader = DataLoader(train_audios, batch_size=batch_size, shuffle=True)
        for step, sample in enumerate(train_audio_loader):

            if step >=args.iter.steps_per_epoch: break

            global_step += 1
            
            x = sample["matrix"].to(device)
            B,C,T=x.shape

            msg_total, msg_fix, msg_temp, msg_val = msg_generator.msg_generate(x)
            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)
            x, wm, mask, msg_total= crop(x, wm, args.crop, msg_total, clip=global_step>=args.crop.clip_starts_from)

            msg_fix = msg_total[:, : msg_fix_length,:]
            msg_temp = msg_total[:,msg_fix_length : msg_fix_length+msg_temp_length,:]
            msg_val = msg_total[:,msg_fix_length+msg_temp_length : ,:]

            x_wm = x+wm
            x_masked = x
            x_wm_masked = x_wm
            # if mask is not None:
            #     x_masked = torch.masked_select(x, mask == 1).reshape(x.shape[0], 1, -1)
            #     x_wm_masked = torch.masked_select(x_wm, mask == 1).reshape(x.shape[0], 1, -1)

            decoded_identity_det, decoded_identity_rec = detector(x_wm)
            decoded_identity_original_det, decoded_identity_original_rec = detector(x)

            decoded_identity_rec_fix=decoded_identity_rec[:,:msg_fix_length,:]
            decoded_identity_rec_temp=decoded_identity_rec[:,msg_fix_length:msg_fix_length+msg_temp_length,:]
            decoded_identity_rec_val=decoded_identity_rec[:,msg_fix_length+msg_temp_length:,:]

            # for easy training
            # n=int(global_step/200)
            # decoded_identity_rec_temp=decoded_identity_rec_temp[:,:4+n,:]
            # msg_temp=msg_temp[:,:4+n,:]

            # percep loss
            wav_loss = loss.wav_loss(x_masked, x_wm_masked)
            loudness_loss = loss.loud_loss(x_masked, x_wm_masked)
            mel_loss = loss.mel_loss(x_masked, x_wm_masked)
            wav2vec_loss = loss.wav2vec_loss(x_masked,x_wm_masked)

            # watermark loss
            wm_det_identity_loss = loss.wm_det_loss(decoded_identity_det,decoded_identity_original_det,mask)
            wm_rec_identity_loss_fix = loss.wm_rec_loss(decoded_identity_rec_fix,mask,msg_fix)
            wm_rec_identity_loss_temp = loss.wm_rec_loss(decoded_identity_rec_temp,mask,msg_temp)
            wm_rec_identity_loss_val = loss.wm_rec_loss(decoded_identity_rec_val,mask,msg_val)
            wm_rec_identity_loss = wm_rec_identity_loss_fix + wm_rec_identity_loss_temp + wm_rec_identity_loss_val
            
            wm_boundary_rec_loss=loss.boundary_loss_rec(decoded_identity_rec,msg_total,mask)
            wm_boundary_det_loss=loss.boundary_loss_det(decoded_identity_det,mask)

            snr = (-snr_fn(torch.masked_select(x_wm, mask == 1).reshape(B,C,-1),torch.masked_select(x, mask == 1).reshape(B,C,-1)))
            pesq=pesq_fn(16000,torch.masked_select(x, mask == 1).reshape(-1).detach().cpu().numpy()[:48000],torch.masked_select(x_wm, mask == 1).reshape(-1).detach().cpu().numpy()[:48000],on_error=PesqError.RETURN_VALUES)
            
            wm_identity_det_acc = compute_accuracy(decoded_identity_det, decoded_identity_original_det,mask).item()
            wm_identity_rec_acc = compute_bit_acc(decoded_identity_rec, msg_total,mask).item()
            wm_identity_rec_acc_fix = compute_bit_acc(decoded_identity_rec_fix, msg_fix,mask).item()
            wm_identity_rec_acc_temp = compute_bit_acc(decoded_identity_rec_temp, msg_temp,mask).item()
            wm_identity_rec_acc_val = compute_bit_acc(decoded_identity_rec_val, msg_val,mask).item()



            if args.augmentation.select_aug_mode!='none' and global_step>=args.augmentation.aug_starts_from:
                selected_augs = select_audio_effects(augmentations,
                    aug_weights,
                    mode=args.augmentation.select_aug_mode,
                    max_length=args.augmentation.n_max_aug,
                )
            else:
                selected_augs = {}
            N_augs = 0
            decoded_det_augs = []
            decoded_rec_augs = []
            wm_rec_aug_loss_fix=torch.tensor(0.0).to(device)
            wm_rec_aug_loss_temp=torch.tensor(0.0).to(device)
            wm_rec_aug_loss_val=torch.tensor(0.0).to(device)
            wm_rec_aug_loss=torch.tensor(0.0).to(device)
            wm_det_aug_loss=torch.tensor(0.0).to(device)
            wm_aug_det_acc=0
            wm_aug_rec_acc=0
            wm_aug_rec_acc_fix=0
            wm_aug_rec_acc_temp=0
            wm_aug_rec_acc_val=0

            t_f=''
            for augmentation_name, augmentation_method in selected_augs.items():
                aug_x_wm, mask_aug_wm=augmentation_method(x_wm, mask=mask)
                aug_x, mask_aug=augmentation_method(x, mask=mask)

                decoded_aug_det, decoded_aug_rec = detector(aug_x_wm)
                decoded_aug_original_det, decoded_aug_original_rec = detector(aug_x)

                decoded_aug_rec_fix=decoded_aug_rec[:,:msg_fix_length,:]
                decoded_aug_rec_temp=decoded_aug_rec[:,msg_fix_length:msg_fix_length+msg_temp_length,:]
                decoded_aug_rec_val=decoded_aug_rec[:,msg_fix_length+msg_temp_length:,:]

                # Adjust message size to match augmented audio length
                T_aug = mask_aug_wm.shape[-1]
                T_orig = msg_total.shape[-1]
                if T_aug <= T_orig:
                    # Augmentation shortened the audio, truncate message
                    msg_total_aug = msg_total[:, :, :T_aug]
                else:
                    # Augmentation lengthened the audio, pad message with zeros
                    pad_size = T_aug - T_orig
                    msg_total_aug = torch.nn.functional.pad(msg_total, (0, pad_size), mode='constant', value=0)
                msg_fix_aug = msg_total_aug[:, :msg_fix_length, :]
                msg_temp_aug = msg_total_aug[:, msg_fix_length:msg_fix_length+msg_temp_length, :]
                msg_val_aug = msg_total_aug[:, msg_fix_length+msg_temp_length:, :]

                # wm loss
                wm_det_aug_loss += loss.wm_det_loss(decoded_aug_det,decoded_aug_original_det,mask_aug_wm)
                wm_rec_aug_loss_fix += loss.wm_rec_loss(decoded_aug_rec_fix,mask_aug_wm,msg_fix_aug)
                wm_rec_aug_loss_temp += loss.wm_rec_loss(decoded_aug_rec_temp,mask_aug_wm,msg_temp_aug)
                wm_rec_aug_loss_val += loss.wm_rec_loss(decoded_aug_rec_val,mask_aug_wm,msg_val_aug)
                wm_rec_aug_loss += wm_rec_aug_loss_fix + wm_rec_aug_loss_temp + wm_rec_aug_loss_val

                wm_aug_det_acc += compute_accuracy(decoded_aug_det, decoded_identity_original_det,mask_aug_wm).item()
                wm_aug_rec_acc += compute_bit_acc(decoded_aug_rec, msg_total_aug,mask_aug_wm).item()
                wm_aug_rec_acc_fix += compute_bit_acc(decoded_aug_rec_fix, msg_fix_aug, mask_aug_wm).item()
                wm_aug_rec_acc_temp += compute_bit_acc(decoded_aug_rec_temp, msg_temp_aug,mask_aug_wm).item()
                wm_aug_rec_acc_val += compute_bit_acc(decoded_aug_rec_val,msg_val_aug, mask_aug_wm).item()

                decoded_det_augs.append(decoded_aug_det)
                decoded_rec_augs.append(decoded_aug_rec)

                t_f+=f'{augmentation_name} '
                N_augs+=1

            # logger.info(t_f)
            if N_augs!=0:

                wm_det_aug_loss /= N_augs
                wm_rec_aug_loss /= N_augs
                wm_rec_aug_loss_fix /= N_augs
                wm_rec_aug_loss_temp /= N_augs
                wm_rec_aug_loss_val /= N_augs

                wm_aug_det_acc /= N_augs
                wm_aug_rec_acc /= N_augs
                wm_aug_rec_acc_fix /= N_augs
                wm_aug_rec_acc_temp /= N_augs
                wm_aug_rec_acc_val /= N_augs

            sum_loss =  lambda_wav * wav_loss + \
                        lambda_wav2vec * wav2vec_loss + \
                        lambda_msmel * mel_loss + \
                        lambda_loud * loudness_loss + \
                        lambda_wm_det * (wm_det_identity_loss + wm_det_aug_loss) + \
                        lambda_wm_rec_fix * (wm_rec_identity_loss_fix + wm_rec_aug_loss_fix) + \
                        lambda_wm_rec_temp * (wm_rec_identity_loss_temp + wm_rec_aug_loss_temp) + \
                        lambda_wm_rec_val * (wm_rec_identity_loss_val + wm_rec_aug_loss_val) + \
                        lambda_boundary_rec * (wm_boundary_rec_loss + wm_boundary_det_loss)
                        
            
            adv_d_loss = adv_g_loss = adv_g_map_loss = torch.tensor(0.0).to(device)
            if global_step>= args.adv.adv_starts_from:
                if step%args.adv.update_d_every==0: #控制disc 优化次数
                    if lambda_adv_d>0:
                        adv_d_loss = adversary.train_adv(real=x, fake=x_wm.detach(), lambda_loss_d=lambda_adv_d)
                    if lambda_adv_g>0 or lambda_adv_g_map>0:
                        adv_g_loss, adv_g_map_loss=adversary(fake=x_wm, real=x)
            sum_loss += adv_g_loss * lambda_adv_g + adv_g_map_loss *lambda_adv_g_map

            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=5)
            
            en_de_optim.step()
            en_de_optim.zero_grad()
            # lr_sched.step()
  
            logger.info(  f"{global_step}/{args.iter.steps_per_epoch}({ep})[{N_augs}]"+
                    f"\n[me] " +
                    f"id_det_acc:{wm_identity_det_acc:.3f}, " +
                    f"id_rec_acc:{wm_identity_rec_acc:.3f}/{wm_identity_rec_acc_fix:.3f}/{wm_identity_rec_acc_temp:.3f}/{wm_identity_rec_acc_val:.3f}, " +
                    f"aug_det_acc:{wm_aug_det_acc:.3f}, " + 
                    f"aug_rec_acc:{wm_aug_rec_acc:.3f}/{wm_aug_rec_acc_fix:.3f}/{wm_aug_rec_acc_temp:.3f}/{wm_aug_rec_acc_val:.3f}, " +
                    f"\n[pe] " +
                    f"snr:{snr:.3f}, " + 
                    f"pesq:{pesq:.3f}, " +
                    f"l_wav2vec: {wav2vec_loss:.6f}," +
                    f"l_wav:{wav_loss:.6f}, " + 
                    f"l_msmel:{mel_loss:.5f}, " +
                    f"l_loud: {loudness_loss:.6f}, " + 
                    f"l_d:{adv_d_loss:.5f}, " + 
                    f"l_g:{adv_g_loss:.5f},"+
                    f"l_gmap:{adv_g_map_loss:.5f}" +
                    f"\n[wm] " +
                    f"l_id_det:{wm_det_identity_loss:.3f}, " + 
                    f"l_aug_det:{wm_det_aug_loss:.3f}, " + 
                    f"l_id_rec:{wm_rec_identity_loss:.3f}/{wm_rec_identity_loss_fix.item():.3f}/{wm_rec_identity_loss_temp.item():.3f}/{wm_rec_identity_loss_val.item():.3f}," +
                    f"l_aug_rec:{wm_rec_aug_loss:.3f}/{wm_rec_aug_loss_fix.item():.3f}/{wm_rec_aug_loss_temp.item():.3f}/{wm_rec_aug_loss_val.item():.3f}," +
                    f"l_wm_boundary_loss:{wm_boundary_det_loss:.6f}/{wm_boundary_rec_loss:6f}, " +
                    f"lambda_wav:{lambda_wav}, " +
                    f"lambda_wav2vec:{lambda_wav2vec}, " +
                    f"lambda_msmel:{lambda_msmel}, " +
                    f"lambda_loud:{lambda_loud}, " +
                    f"\n[sh] {list(x_wm.shape)}" +
                    f"\n[ch] "+t_f +
                    f"\n---"
                    )

            if global_step%args.iter.save_sample_by_step==0:
                save_audio_path = os.path.join(args.path.wm_speech, "{}_step{}.wav".format(args["attack_type"], global_step))
                torchaudio.save(save_audio_path, src = x_wm.detach().squeeze()[0].unsqueeze(0).to("cpu"), sample_rate = sample["trans_sr"][0])
                save_wm_path = os.path.join(args.path.wm_speech, "{}_step{}_wm.wav".format(args["attack_type"], global_step))
                torchaudio.save(save_wm_path, src = (wm.squeeze()[0]).detach().unsqueeze(0).to("cpu"), sample_rate = sample["trans_sr"][0])

            if global_step%args.iter.wandb_log_by_step==0:
                wandb.log({
                    f"train/pesq": pesq, 
                    f"train/snr": snr,
                    f"train/wm_identity_det_acc": wm_identity_det_acc,
                    f"train/wm_identity_rec_acc": wm_identity_rec_acc,
                    f"train/wm_identity_rec_acc_fix": wm_identity_rec_acc_fix,
                    f"train/wm_identity_rec_acc_temp": wm_identity_rec_acc_temp,
                    f"train/wm_identity_rec_acc_val": wm_identity_rec_acc_val,
                    f"train/wm_aug_det_acc": wm_aug_det_acc,
                    f"train/wm_aug_rec_acc": wm_aug_rec_acc,
                    f"train/wm_aug_rec_acc_fix": wm_aug_rec_acc_fix,
                    f"train/wm_aug_rec_acc_temp": wm_aug_rec_acc_temp,
                    f"train/wm_aug_rec_acc_val": wm_aug_rec_acc_val,
                    f"train/wm_boundary_rec_loss": wm_boundary_rec_loss.item(),
                    f"train/wm_boundary_det_loss": wm_boundary_det_loss.item(),
                    f"train/wav2vec_loss": wav2vec_loss.item(),
                    f"train/wav_loss": wav_loss.item(),
                    f"train/mel_loss": mel_loss.item(),
                    f"train/loudness_loss": loudness_loss.item(),
                    f"train/wm_det_identity_loss": wm_det_identity_loss.item(),
                    f"train/wm_det_aug_loss": wm_det_aug_loss.item(),
                    f"train/wm_rec_identity_loss": wm_rec_identity_loss.item(),
                    f"train/wm_rec_identity_loss_fix": wm_rec_identity_loss_fix.item(),
                    f"train/wm_rec_identity_loss_temp": wm_rec_identity_loss_temp.item(),
                    f"train/wm_rec_identity_loss_val": wm_rec_identity_loss_val.item(),
                    f"train/wm_rec_aug_loss": wm_rec_aug_loss.item(),
                    f"train/wm_rec_aug_loss_fix": wm_rec_aug_loss_fix.item(),
                    f"train/wm_rec_aug_loss_temp": wm_rec_aug_loss_temp.item(),
                    f"train/wm_rec_aug_loss_val": wm_rec_aug_loss_val.item(),
                    f"train/adv_d_loss": adv_d_loss.item(),
                    f"train/adv_g_loss": adv_g_loss.item(),
                    f"train/adv_g_map_loss": adv_g_map_loss.item()
                }, step=global_step)
                
        if ep % args.iter.save_skpt_by_epoch == 0:

            module_state_dict={
                f"embedder": embedder.state_dict(),
                f"detector": detector.state_dict(),
                # f"discriminator": discriminator.state_dict(),
                f"en_de_optim": en_de_optim.state_dict(),
                # f"d_optim":d_op.state_dict(),
                # f"lr_sched":lr_sched.state_dict(),
                # f"lr_sched_d":lr_sched_d.state_dict(),
                f"global_step":global_step,
                f"epoch":ep
            }

            save_ckpt(ckpt_dir=args.path.ckpt, base_name=args.experiment_name, module_state_dict=module_state_dict, epoch=ep)
            shutil.copyfile(os.path.realpath(__file__), os.path.join(args.path.ckpt, os.path.basename(os.path.realpath(__file__)))) # save training scripts

        # if ep % args.iter.eval_by_epoch == 0:
        #     eval(args, embedder, detector, val_audios_loader, loss, logger, ep, global_step=global_step)



def eval(args, embedder, detector, val_audios_loader, loss, logger, ep, global_step):
    with torch.no_grad():
        embedder.eval()
        detector.eval()
        msg_generator=MsgGenerator(args.wm_msg)

        count = 0.0
        msg_fix_length = args.wm_msg.fix.length
        msg_temp_length = args.wm_msg.temp.length
        msg_val_length = args.wm_msg.val.length

        wav_loss=0.0
        loudness_loss=0.0
        mel_loss=0.0
        wav2vec_loss=0.0

        wm_det_identity_loss = 0.0
        wm_rec_identity_loss_fix = 0.0
        wm_rec_identity_loss_temp = 0.0
        wm_rec_identity_loss_val = 0.0
        wm_rec_identity_loss = 0.0
        wm_boundary_rec_loss =0.0
        wm_boundary_det_loss =0.0

        wm_identity_det_acc=0.0
        wm_identity_rec_acc=0.0
        wm_identity_rec_acc_fix=0.0
        wm_identity_rec_acc_temp=0.0
        wm_identity_rec_acc_val=0.0
        pesq=0.0
                
        snr=0.0

        for i, sample in enumerate(val_audios_loader):
            count += 1

            x = sample["matrix"].to(device)

            B,C,T=x.shape

            msg_total, msg_fix, msg_temp, msg_val = msg_generator.msg_generate(x)

            # wm = embedder(model_config=args.model, klass=args.embedder, msg_fix_nbits=msg_fix_length, msg_temp_nbits=msg_temp_length, msg_val_nbits=msg_val_length)

            wm = embedder(x=x, sample_rate=16_000, message=msg_fix, temp_message=msg_total)

            x, wm, mask, msg_total= crop(x, wm, args.crop, msg_total, clip=global_step>=args.crop.clip_starts_from)
   
            # mask=None
            x_wm = x+wm
            x_wm = torch.clip(x_wm, -1, 1)

            decoded_identity_det, decoded_identity_rec  = detector(x_wm)
            decoded_identity_original_det, decoded_identity_original_rec = detector(x.detach())
            
            decoded_identity_rec_fix=decoded_identity_rec[:,:msg_fix_length,:]
            decoded_identity_rec_temp=decoded_identity_rec[:,msg_fix_length:msg_fix_length+msg_temp_length,:]
            decoded_identity_rec_val=decoded_identity_rec[:,msg_fix_length+msg_temp_length:,:]


            if torch.isnan(decoded_identity_det).any() or torch.isnan(decoded_identity_rec).any():
                logger.warning("NaN detected in decoded_identity_det or decoded_identity_rec, skipping this sample.")
                count-=1
                continue

            x_masked = x
            x_wm_masked = x_wm
            if mask is not None:
                x_masked = torch.masked_select(x, mask == 1).reshape(x.shape[0], 1, -1)
                x_wm_masked = torch.masked_select(x_wm, mask == 1).reshape(x.shape[0], 1, -1)


            # percep loss
            wav_loss = loss.wav_loss(x_masked, x_wm_masked)
            loudness_loss = loss.loud_loss(x_masked, x_wm_masked)
            mel_loss = loss.mel_loss(x_masked, x_wm_masked)
            wav2vec_loss = loss.wav2vec_loss(x_masked,x_wm_masked)

            # watermark loss
            wm_det_identity_loss += loss.wm_det_loss(decoded_identity_det,decoded_identity_original_det,mask)
            wm_rec_identity_loss_fix += loss.wm_rec_loss(decoded_identity_rec_fix,mask,msg_fix)
            wm_rec_identity_loss_temp += loss.wm_rec_loss(decoded_identity_rec_temp,mask,msg_temp)
            wm_rec_identity_loss_val += loss.wm_rec_loss(decoded_identity_rec_val,mask,msg_val)
            wm_rec_identity_loss += wm_rec_identity_loss_fix + wm_rec_identity_loss_temp + wm_rec_identity_loss_val
            wm_boundary_rec_loss += loss.boundary_loss_rec(decoded_identity_rec,msg_total,mask)
            wm_boundary_det_loss += loss.boundary_loss_det(decoded_identity_rec,mask)


            wm_identity_det_acc += compute_accuracy(decoded_identity_det, decoded_identity_original_det,mask).item()
            wm_identity_rec_acc += compute_bit_acc(decoded_identity_rec, msg_total,mask).item()
            wm_identity_rec_acc_fix += compute_bit_acc(decoded_identity_rec_fix, msg_fix,mask).item()
            wm_identity_rec_acc_temp += compute_bit_acc(decoded_identity_rec_temp, msg_temp,mask).item()
            wm_identity_rec_acc_val += compute_bit_acc(decoded_identity_rec_val, msg_val,mask).item()

            try:
                pesq+=pesq_fn(16000,torch.masked_select(x, mask == 1).reshape(-1).detach().cpu().numpy()[:48000],torch.masked_select(x_wm, mask == 1).reshape(-1).detach().cpu().numpy()[:48000])
            except:
                pesq+=0
            

            snr += (-snr_fn(torch.masked_select(x_wm, mask == 1).reshape(B,C,-1),torch.masked_select(x, mask == 1).reshape(B,C,-1)))

        wav_loss /= count
        loudness_loss /= count
        mel_loss /= count
        wav2vec_loss /=count

        wm_det_identity_loss /= count
        wm_rec_identity_loss_fix /= count
        wm_rec_identity_loss_temp /= count
        wm_rec_identity_loss_val /= count
        wm_rec_identity_loss /=count
        wm_boundary_rec_loss /=count
        wm_boundary_det_loss /=count
        wm_identity_det_acc /= count
        wm_identity_rec_acc /= count
        wm_identity_rec_acc_fix /= count
        wm_identity_rec_acc_temp /= count
        wm_identity_rec_acc_val /= count
    
        snr /= count
        pesq /= count

        wandb.log({
            f"eval/pesq": pesq, 
            f"eval/snr": snr,
            f"eval/wm_identity_det_acc": wm_identity_det_acc,
            f"eval/wm_identity_rec_acc": wm_identity_rec_acc,
            f"eval/wm_identity_rec_acc_fix": wm_identity_rec_acc_fix,
            f"eval/wm_identity_rec_acc_temp": wm_identity_rec_acc_temp,
            f"eval/wm_identity_rec_acc_val": wm_identity_rec_acc_val,
            f"eval/wm_boundary_rec_loss": wm_boundary_rec_loss.item(),
            f"eval/wm_boundary_det_loss": wm_boundary_det_loss.item(),
            f"eval/wav2vec_loss": wav2vec_loss.item(),
            f"eval/wav_loss": wav_loss.item(),
            f"eval/mel_loss": mel_loss.item(),
            f"eval/loudness_loss": loudness_loss.item(),
            f"eval/wm_det_identity_loss": wm_det_identity_loss.item(),
            f"eval/wm_rec_identity_loss": wm_rec_identity_loss.item(),
            f"eval/wm_rec_identity_loss_fix": wm_rec_identity_loss_fix.item(),
            f"eval/wm_rec_identity_loss_temp": wm_rec_identity_loss_temp.item(),
            f"eval/wm_rec_identity_loss_val": wm_rec_identity_loss_val.item()
        }, step=global_step)

        logger.info(f"==========eval epoch {ep}==========" +  
                f"\n[me] " +
                f"id_det_acc:{wm_identity_det_acc:.3f}, " +
                f"id_rec_acc:{wm_identity_rec_acc:.3f}/{wm_identity_rec_acc_fix:.3f}/{wm_identity_rec_acc_temp:.3f}/{wm_identity_rec_acc_val:.3f}, " +
                f"\n[pe] " +
                f"snr:{snr:.3f}, " + 
                f"pesq:{pesq:.3f}, " +
                f"l_wav2vec: {wav2vec_loss:.6f}," +
                f"l_wav:{wav_loss:.6f}, " + 
                f"l_msmel:{mel_loss:.5f}, " +
                f"l_loud: {loudness_loss:.6f}, " + 
                f"\n[wm] " +
                f"l_id_det:{wm_det_identity_loss:.3f}, " + 
                f"l_id_rec:{wm_rec_identity_loss:.3f}/{wm_rec_identity_loss_fix.item():.3f}/{wm_rec_identity_loss_temp.item():.3f}/{wm_rec_identity_loss_val.item():.3f}," +
                f"l_wm_boundary_loss:{wm_boundary_det_loss:.6f}/{wm_boundary_rec_loss:.6f}"
                )

# 创建水印，选择一个分布来生成水印
def generate_watermark(batch_size, length, micro, sigma):
    sigmoid = torch.nn.Sigmoid()
    eye_matrix = np.eye(length)
    mask_convariance_maxtix = eye_matrix * (sigma ** 2)
    center = np.ones(length) * micro

    w_bit = multivariate_normal.rvs(mean = center, cov = mask_convariance_maxtix, size = [batch_size, 1])
    w_bit = torch.from_numpy(w_bit).float()
    return w_bit


def compute_accuracy(positive, negative, mask=None):

    if positive.size(1)==0:
        return torch.tensor(0.0)

    if mask is not None:
        # cut last dim of positive to keep only where mask is 1
        # new_shape = [*positive.shape[:-1], -1]  # b nbits t -> b nbits -1
        # positive = torch.masked_select(positive, mask == 1).reshape(new_shape)
        acc = ((positive>0) == (mask==1)).float().mean()

        return acc
    else:
        N = (positive[:, 0, :] > 0).float().sum()/mask.sum() + (negative[:, 0, :] < 0).float().mean()
        
        acc = N / 2

        return acc


def compute_bit_acc(decoded, message, mask=None):
    """Compute bit accuracy.
    Args:
        positive: detector outputs [bsz, 2+nbits, time_steps]
        original: original message (0 or 1) [bsz, nbits]
        mask: mask of the watermark [bsz, 1, time_steps]
    """
    if decoded.size(1)==0:
        return torch.tensor(0.0)

    if mask is not None:
        # cut last dim of positive to keep only where mask is 1
        new_shape = [*decoded.shape[:-1], -1]  # b nbits t -> b nbits -1
        decoded = torch.masked_select(decoded, mask == 1).reshape(new_shape)
        message = torch.masked_select(message, mask == 1).reshape(new_shape)
    # average decision over time, then threshold
    # decoded = decoded.mean(dim=-1) > 0  # b nbits
    decoded = decoded > 0  # b nbits
    res = (decoded == message).float().mean()
    return res


@hydra.main(config_path="config", config_name="demucs_temp")
def run(args):

    cwd=os.getcwd()

    wandb_config={}
    wandb_config.update(OmegaConf.to_container(args, resolve=True))
    wandb.init(
            project=args.project_name,
            notes=args.notes,
            name=args.experiment_name+"_"+str(args.seed),
            group=args.scenario_name,
            job_type="training",
            config=wandb_config,
            reinit=True)
    

    args_save_path = os.path.join(args.path.ckpt, "hydra_config.yaml")
    # 将配置保存为YAML文件
    for p in args["path"].values():
         os.makedirs(p, exist_ok=True)

    with open(args_save_path, 'w') as file:
        # 使用OmegaConf.to_yaml方法将配置转换为YAML格式
        file.write(OmegaConf.to_yaml(args))
    

    # seet seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.chdir(cwd)

    main(args)

if __name__ == "__main__":
    run()


