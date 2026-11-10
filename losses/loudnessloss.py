# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import typing as tp

import julius
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torchaudio.functional.filtering import highpass_biquad, treble_biquad


def basic_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")
    eps = torch.finfo(torch.float32).eps
    gate_duration = 0.4
    overlap = 0.75
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))


    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)


    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)


    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)

    loudness = -0.691 + 10 * torch.log10(energy_weighted + eps)
    return loudness


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "data should be contiguous"
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


class FLoudnessRatio(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: tp.Optional[float] = 20,
        overlap: float = 0.5,
        epsilon: float = torch.finfo(torch.float32).eps,
        n_bands: int = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.epsilon = epsilon
        if n_bands == 0:
            self.filter = None
        else:
            self.filter = julius.SplitBands(sample_rate=sample_rate, n_bands=n_bands)
        self.loudness = torchaudio.transforms.Loudness(sample_rate)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        assert ref_sig.shape == out_sig.shape
        assert self.filter is not None
        bands_ref = self.filter(ref_sig)
        bands_out = self.filter(out_sig)
        l_noise = self.loudness(bands_ref - bands_out)
        l_ref = self.loudness(bands_ref)
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = torch.nn.functional.softmax(l_ratio, dim=0) * l_ratio
        return loss.sum()


class TLoudnessRatio(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: float = 0.5,
        overlap: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.loudness = torchaudio.transforms.Loudness(sample_rate)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        assert ref_sig.shape == out_sig.shape
        assert C == 1

        frame = int(self.segment * self.sample_rate)
        stride = int(frame * (1 - self.overlap))
        gt = _unfold(ref_sig, frame, stride).view(-1, 1, frame)
        est = _unfold(out_sig, frame, stride).view(-1, 1, frame)
        l_noise = self.loudness(gt - est)
        l_ref = self.loudness(gt)
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = torch.nn.functional.softmax(l_ratio, dim=0) * l_ratio
        return loss.sum()


class TFLoudnessRatio(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: float = 0.5,
        overlap: float = 0.5,
        n_bands: int = 0,
        clip_min: float = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.clip_min = clip_min
        self.temperature = temperature
        if n_bands == 0:
            self.filter = None
        else:
            self.n_bands = n_bands
            self.filter = julius.SplitBands(sample_rate=sample_rate, n_bands=n_bands)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape

        assert ref_sig.shape == out_sig.shape
        assert C == 1
        assert self.filter is not None

        bands_ref = self.filter(ref_sig).view(B * self.n_bands, 1, -1)
        bands_out = self.filter(out_sig).view(B * self.n_bands, 1, -1)
        frame = int(self.segment * self.sample_rate)
        stride = int(frame * (1 - self.overlap))
        gt = _unfold(bands_ref, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        est = _unfold(bands_out, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        l_noise = basic_loudness(est - gt, sample_rate=self.sample_rate)
        l_ref = basic_loudness(gt, sample_rate=self.sample_rate)
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = torch.nn.functional.softmax(l_ratio / self.temperature, dim=0) * l_ratio
        return loss.mean()
