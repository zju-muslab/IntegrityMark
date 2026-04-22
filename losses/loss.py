from statistics import mean
import torch
import torch.nn as nn
from .loudnessloss import TFLoudnessRatio
from torchaudio.transforms import MelSpectrogram
from torch.nn import functional as F
import math
import numpy as np
from typing import Literal

import typing as tp
from losses.wav2vec_perceploss import WeightedPerceptualLoss


class Loss(nn.Module):
    def __init__(self, args):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(Loss, self).__init__()
        self.msg_loss_fn = nn.MSELoss()
        # self.msg_loss = nn.CrossEntropyLoss()
        self.wm_det_loss_fn = nn.BCEWithLogitsLoss()
        self.embedding_loss = nn.MSELoss()
        # self.msg_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.fragile_target = torch.tensor(0.5)
        self.loudness_loss_fn=TFLoudnessRatio(sample_rate=16000, n_bands=args.optimize.loss.loudness.n_bands, temperature=args.optimize.loss.loudness.temperature)
        self.mel_loss_fn=MultiScaleMelSpectrogramLoss(sample_rate=16000)
        self.wm_rec_loss_fn = WMMbLoss(temperature=1, loss_type="bce")
        self.wm_det_loss_fn = WMDetectionLoss(temperature=1)


        self.wav_loss_fn = nn.MSELoss()
        # self.wav_loss_fn = EnhancedLogTimeLoss()

        self.boundary_loss_fn = BoundaryLoss()
        alpha=0.95
        self.max_length=args.dataset.audio.max_len
        self.ewma_weights = torch.tensor([alpha ** i for i in range(self.max_length)])
        self.ewma_weights = self.ewma_weights.flip(dims=(0,))  # 反转权重以匹配时间序列

        self.wav2vec_loss_fn = WeightedPerceptualLoss(selected_layers=[0,1,2,3])
   

    def boundary_loss_det(self, positive, mask):
        return self.boundary_loss_fn.det_forward(positive, mask)


    def boundary_loss_rec(self, positive, msg, mask):
        return self.boundary_loss_fn(positive, msg, mask)
     
    def wav2vec_loss(self, x, w_x):
        if len(x.shape)>2:
            x=x.squeeze(1)
        if len(w_x.shape)>2:
            w_x=w_x.squeeze(1)
        return self.wav2vec_loss_fn(x, w_x)


    def continuity_penalty_loss(self, x, threshold=0.55, window_size=64, alpha=0.85):
        x = torch.sigmoid(x)

        padded_x = F.pad(x, (window_size-1, 0), mode='reflect')

        unfolded_x = padded_x.unfold(2, window_size, 1)

        # 应用权重并求和
        avg = (unfolded_x * alpha).sum(dim=-1)

        # 计算每个窗口的平均值，并将结果展平回到原始形状
        # avg = weighted_unfolded_x.mean(dim=-1)

        # 计算差值并应用阈值mask
        diff = torch.abs(x[...,32:] - avg[...,:-32])
        diff = diff[...,:-30]
        mask = diff > threshold

        # 更新penalty张量
        penalty = torch.where(mask, diff, torch.zeros_like(diff))

        # 计算最终的penalty项，取平均值
        penalty = torch.mean(penalty)
        
        return penalty


    def wm_rec_loss(self, x, mask, msg):
        return self.wm_rec_loss_fn(x, mask, msg)

    def wm_det_loss(self, pos, neg, mask):
        return self.wm_det_loss_fn(pos, neg, mask)

    def wm_rec_loss_old(self,msg, rec_msg):
        rec_msg=rec_msg[...,1:]
        return self.msg_loss_fn(rec_msg, msg)

    def wm_det_loss_old(self, pos_msg, neg_msg):
        pos=pos_msg[...,:1]
        neg=neg_msg[...,:1]
        pos_correct_classes = torch.ones_like(pos).to(pos.device)
        neg_correct_classes = torch.zeros_like(neg).to(neg.device)
        loss_p = self.wm_det_loss_fn(pos, pos_correct_classes)
        loss_n = self.wm_det_loss_fn(neg, neg_correct_classes)
        return loss_p+loss_n

    def wav_loss(self,x,w_x):
        return self.wav_loss_fn(x,w_x)
        # return self.embedding_loss(x,w_x)

    def loud_loss(self, x, w_x):
        return self.loudness_loss_fn(x, w_x)
    
    def mel_loss(self,x, w_x):
        return self.mel_loss_fn(x,w_x)


class BoundaryLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(BoundaryLoss, self).__init__()
        self.epsilon = epsilon  # 防止除零的小量

    def det_forward(self, pred, mask):
        target = mask.expand_as(pred)

        # 计算梯度变化（差分）, 注意torch.diff会减少一个时间步长
        pred_diff = torch.diff(pred, dim=-1)
        target_diff = torch.diff(target, dim=-1)
        # # 对差异取绝对值，得到边界强度
        pred_boundaries = torch.abs(pred_diff)
        target_boundaries = torch.abs(target_diff)


        # 边界对齐损失：鼓励预测的边界强度与目标的边界强度一致
        boundary_alignment_loss = F.l1_loss(pred_boundaries, target_boundaries, reduction='none')
        boundary_loss = torch.mean(boundary_alignment_loss)

        return boundary_loss

    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, H, T) 模型预测的概率分布
            target: (B, H, T) 目标的真实标签（假设是0或1）
            mask: (B, 1, T) 掩码，用于忽略某些时间步长
        """
        # 扩展mask以匹配pred的形状



        pred = torch.nn.functional.sigmoid(pred)
        expanded_mask = mask.expand_as(pred)


        # 应用掩码到预测和目标
        masked_pred = pred * expanded_mask
        masked_target = target * expanded_mask

        # 计算梯度变化（差分）, 注意torch.diff会减少一个时间步长
        pred_diff = torch.diff(masked_pred, dim=-1)
        target_diff = torch.diff(masked_target, dim=-1)

        # # 对差异取绝对值，得到边界强度
        pred_boundaries = torch.abs(pred_diff)
        target_boundaries = torch.abs(target_diff)

        # 边界对齐损失：鼓励预测的边界强度与目标的边界强度一致
        boundary_alignment_loss = F.l1_loss(pred_boundaries, target_boundaries, reduction='none')

        boundary_loss = boundary_alignment_loss

        # 忽略那些因为torch.diff而没有对应的目标的时间步长
        valid_positions = mask[:, :, :-1]
        masked_boundary_loss = torch.masked_select(boundary_alignment_loss, valid_positions == 1)  

        # 平均边界误差，同时确保不会出现除以0的情况
        boundary_loss = torch.mean(masked_boundary_loss)

        return boundary_loss

class TemporalBinaryClassificationLoss(nn.Module):
    def __init__(self, lambda_boundary=1.0):
        super(TemporalBinaryClassificationLoss, self).__init__()
        self.lambda_boundary = lambda_boundary  # 边界惩罚项的权重
        self.cross_entropy = nn.BCEWithLogitsLoss()  # 二分类的交叉熵损失

    def forward(self, logits, targets):
        # logits: (batch_size, sequence_length)
        # targets: (batch_size, sequence_length)

        # 计算交叉熵损失
        ce_loss = self.cross_entropy(logits, targets)

        # 计算边界误差惩罚项
        boundary_loss = self.compute_boundary_loss(logits, targets)

        # 总损失
        total_loss = ce_loss + self.lambda_boundary * boundary_loss
        return total_loss

class MultiScaleMelSpectrogramLoss(nn.Module):
    """Multi-Scale spectrogram loss (msspec).

    Args:
        sample_rate (int): Sample rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        n_mels (int): Number of mel bins.
        f_min (float): Minimum frequency.
        f_max (float or None): Maximum frequency.
        normalized (bool): Whether to normalize the melspectrogram.
        alphas (bool): Whether to use alphas as coefficients or not.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 11,
                 n_mels: int = 64, f_min: float = 64, f_max: tp.Optional[float] = None,
                 normalized: bool = False, alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        l1s = list()
        l2s = list()
        self.alphas = list()
        self.total = 0
        self.normalized = normalized
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=False, normalized=normalized, floor_level=floor_level))
            l2s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=True, normalized=normalized, floor_level=floor_level))
            if alphas:
                self.alphas.append(np.sqrt(2 ** i - 1))
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.l1s = nn.ModuleList(l1s)
        self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        loss = 0.0
        self.l1s.to(x.device)
        self.l2s.to(x.device)
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            loss += F.l1_loss(s_x_1, s_y_1) + self.alphas[i] * F.mse_loss(s_x_2, s_y_2)
        if self.normalized:
            loss = loss / self.total
        return loss
    
class MelSpectrogramWrapper(nn.Module):
    """Wrapper around MelSpectrogram torchaudio transform providing proper padding
    and additional post-processing including log scaling.

    Args:
        n_mels (int): Number of mel bins.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        sample_rate (int): Sample rate.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level based on human perception (default=1e-5).
    """
    def __init__(self, n_fft = 1024, hop_length = 256, win_length = None,
                 n_mels = 80, sample_rate = 22050, f_min: float = 0.0, f_max = None,
                 log = True, normalized = False, floor_level = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, f_min=f_min, f_max=f_max, normalized=normalized,
                                            window_fn=torch.hann_window, center=False)
        self.floor_level = floor_level
        self.log = log

    def forward(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)
    

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


class AdversarialLoss(nn.Module):
    """Adversary training wrapper.

    Args:
        adversary (nn.Module): The adversary module will be used to estimate the logits given the fake and real samples.
            We assume here the adversary output is ``Tuple[List[torch.Tensor], List[List[torch.Tensor]]]``
            where the first item is a list of logits and the second item is a list of feature maps.
        optimizer (torch.optim.Optimizer): Optimizer used for training the given module.
        loss (AdvLossType): Loss function for generator training.
        loss_real (AdvLossType): Loss function for adversarial training on logits from real samples.
        loss_fake (AdvLossType): Loss function for adversarial training on logits from fake samples.
        loss_feat (FeatLossType): Feature matching loss function for generator training.
        normalize (bool): Whether to normalize by number of sub-discriminators.

    Example of usage:
        adv_loss = AdversarialLoss(adversaries, optimizer, loss, loss_real, loss_fake)
        for real in loader:
            noise = torch.randn(...)
            fake = model(noise)
            adv_loss.train_adv(fake, real)
            loss, _ = adv_loss(fake, real)
            loss.backward()
    """
    def __init__(self,
                 adversary: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 normalize: bool = True):
        super().__init__()
        self.adversary: nn.Module = adversary
        self.optimizer = optimizer
        self.loss = hinge_loss
        self.loss_real = hinge_real_loss
        self.loss_fake = hinge_fake_loss
        self.loss_feat = FeatureMatchingLoss()
        self.normalize = normalize

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Add the optimizer state dict inside our own.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'optimizer'] = self.optimizer.state_dict()
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Load optimizer state.
        self.optimizer.load_state_dict(state_dict.pop(prefix + 'optimizer'))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_adversary_pred(self, x):
        """Run adversary model, validating expected output format."""
        logits, fmaps = self.adversary(x)
        assert isinstance(logits, list) and all([isinstance(t, torch.Tensor) for t in logits]), \
            f'Expecting a list of tensors as logits but {type(logits)} found.'
        assert isinstance(fmaps, list), f'Expecting a list of features maps but {type(fmaps)} found.'
        for fmap in fmaps:
            assert isinstance(fmap, list) and all([isinstance(f, torch.Tensor) for f in fmap]), \
                f'Expecting a list of tensors as feature maps but {type(fmap)} found.'
        return logits, fmaps

    def train_adv(self, fake: torch.Tensor, real: torch.Tensor, lambda_loss_d=0) -> torch.Tensor:
        """Train the adversary with the given fake and real example.

        We assume the adversary output is the following format: Tuple[List[torch.Tensor], List[List[torch.Tensor]]].
        The first item being the logits and second item being a list of feature maps for each sub-discriminator.

        This will automatically synchronize gradients (with `flashy.distrib.eager_sync_model`)
        and call the optimizer.
        """
        loss = torch.tensor(0., device=fake.device)
        all_logits_fake_is_fake, _ = self.get_adversary_pred(fake.detach())
        all_logits_real_is_fake, _ = self.get_adversary_pred(real.detach())
        n_sub_adversaries = len(all_logits_fake_is_fake)
        for logit_fake_is_fake, logit_real_is_fake in zip(all_logits_fake_is_fake, all_logits_real_is_fake):
            loss += self.loss_fake(logit_fake_is_fake) + self.loss_real(logit_real_is_fake)

        loss = loss

        if self.normalize:
            loss /= n_sub_adversaries

        loss_=loss * lambda_loss_d

        self.optimizer.zero_grad()
        loss_.backward()
        torch.nn.utils.clip_grad_value_(self.adversary.parameters(), clip_value=100)
        self.optimizer.step()

        return loss

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Return the loss for the generator, i.e. trying to fool the adversary,
        and feature matching loss if provided.
        """
        adv = torch.tensor(0., device=fake.device)
        feat = torch.tensor(0., device=fake.device)
        all_logits_fake_is_fake, all_fmap_fake = self.get_adversary_pred(fake)
        all_logits_real_is_fake, all_fmap_real = self.get_adversary_pred(real)
        n_sub_adversaries = len(all_logits_fake_is_fake)
        for logit_fake_is_fake in all_logits_fake_is_fake:
            adv += self.loss(logit_fake_is_fake)
        if self.loss_feat:
            for fmap_fake, fmap_real in zip(all_fmap_fake, all_fmap_real):
                feat += self.loss_feat(fmap_fake, fmap_real)

        if self.normalize:
            adv /= n_sub_adversaries
            feat /= n_sub_adversaries

        return adv, feat

def hinge_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return -x.mean()

def hinge_real_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(x - 1, torch.tensor(0., device=x.device).expand_as(x)))


def hinge_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(-x - 1, torch.tensor(0., device=x.device).expand_as(x)))

class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for adversarial training.

    Args:
        loss (nn.Module): Loss to use for feature matching (default=torch.nn.L1).
        normalize (bool): Whether to normalize the loss.
            by number of feature maps.
    """
    def __init__(self, loss: nn.Module = torch.nn.L1Loss(), normalize: bool = True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, fmap_fake: tp.List[torch.Tensor], fmap_real: tp.List[torch.Tensor]) -> torch.Tensor:
        assert len(fmap_fake) == len(fmap_real) and len(fmap_fake) > 0
        feat_loss = torch.tensor(0., device=fmap_fake[0].device)
        feat_scale = torch.tensor(0., device=fmap_fake[0].device)
        n_fmaps = 0
        for (feat_fake, feat_real) in zip(fmap_fake, fmap_real):
            assert feat_fake.shape == feat_real.shape
            n_fmaps += 1
            feat_loss += self.loss(feat_fake, feat_real)
            feat_scale += torch.mean(torch.abs(feat_real))

        if self.normalize:
            feat_loss /= n_fmaps

        return feat_loss

class WMDetectionLoss(nn.Module):
    """Compute the detection loss"""
    def __init__(self, p_weight: float = 1.0, n_weight: float = 1.0, temperature=0.1) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.p_weight = p_weight
        self.n_weight = n_weight
        self.temperature = temperature

    def forward(self, positive, negative, mask, message=None):

        # positive = positive[:, :2, :]  # b 2+nbits t -> b 2 t
        # negative = negative[:, :2, :]  # b 2+nbits t -> b 2 t

        # print(positive.shape)
        if positive.size(1)==0:
            return torch.tensor(0.0).to(positive.device)

        # dimensionality of positive [bsz, classes=2, time_steps]
        # correct classes for pos = [bsz, time_steps] where all values = 1 for positive

        pos_correct_classes = torch.ones_like(positive, dtype=int)
        neg_correct_classes = torch.zeros_like(positive, dtype=int)

        # taking log because network outputs softmax
        # NLLLoss expects a logsoftmax input
        # positive = torch.log(positive)
        # negative = torch.log(negative)

        if not torch.all(mask == 1):
            # pos_correct_classes [bsz, timesteps] mask [bsz, 1, timesptes]
            # mask is applied to the watermark, this basically flips the tgt class from 1 (positive)
            # to 0 (negative) in the correct places

            
            pos_correct_classes = pos_correct_classes * mask.to(int)
            loss_p = self.p_weight * self.criterion(positive/self.temperature, pos_correct_classes.float())
            # no need for negative class loss here since some of the watermark
            # is masked to negative
            return loss_p

        else:
            loss_p = self.p_weight * self.criterion(positive/self.temperature, pos_correct_classes.float())
            loss_n = self.n_weight * self.criterion(negative/self.temperature, neg_correct_classes.float())
            return loss_p + loss_n

class WMMbLoss(nn.Module):
    def __init__(self, temperature: float, loss_type: Literal["bce", "mse"]) -> None:
        """
        Compute the masked sample-level detection loss
        (https://arxiv.org/pdf/2401.17264)

        Args:
            temperature: temperature for loss computation
            loss_type: bce or mse between outputs and original message
        """
        super().__init__()
        self.bce_with_logits = (
            nn.BCEWithLogitsLoss(reduction='mean')
        )  # same as Softmax + NLLLoss, but when only 1 output unit
        self.mse = nn.MSELoss(reduction='mean')
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, positive, mask, message, negative=None, penalty_alpha=0, penalty_threshold=0):
        if positive.size(1)==0:
            return torch.tensor(0.0).to(positive.device)
        
        new_shape = [*positive.shape[:-1], -1]  # b nbits -1
        positive = torch.masked_select(positive, mask == 1).reshape(new_shape)
        message = torch.masked_select(message, mask == 1).reshape(new_shape)
        # print(positive.shape,mask.shape, message.shape,positive.shape)

        # Accuracy-based channel weighting, 准确率低的channel权重高
        B, C, T = message.shape
        predictions = (positive > 0).float()
        accuracy_per_channel = (predictions == message).float().mean(dim=(0, 2))
        # alpha = 3  # 指数因子，值越大，权重差异越明显
        # weights = (1.0 / (accuracy_per_channel + 1e-9))**alpha
        w_temperature= 0.2
        weights = torch.exp((1.0 - accuracy_per_channel + 1e-9) / w_temperature) # softmax_weight
        weights = weights / weights.sum()  # Normalize weights to sum to 1
        weights = weights.view(1, C, 1).expand(B, C, T)  # Shape: (B, C, T)

        if self.loss_type == "bce":
            # in this case similar to temperature in softmax
            loss = self.bce_with_logits(positive / self.temperature, message.float())
        elif self.loss_type == "mse":
            loss = self.mse(positive / self.temperature, message.float())         # loss = (loss * weights).mean()
        # loss = loss.mean()
        # print(loss.shape,weights.shape)
        loss = (loss * weights).sum()/B/T
        # loss = (loss * weights).mean()
        # loss = loss.mean()
        return loss
    

class EnhancedLogTimeLoss(nn.Module):
    def __init__(self, eps=1e-9, alpha=1.0):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        
    def forward(self, marked, original):
        # 取绝对值差异
        abs_diff = torch.abs(marked - original)
        
        # log(1 + x)形式，保证loss总是正值
        # alpha用于调节loss的敏感度
        log_loss = torch.mean(torch.log(1 + self.alpha * abs_diff))
        return log_loss
