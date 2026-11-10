# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp
import torch
from torch import nn
import torchaudio


def db_to_scale(volume: tp.Union[float, torch.Tensor]):
    return 10 ** (volume / 20)


def scale_to_db(scale: torch.Tensor, min_volume: float = -120):
    min_scale = db_to_scale(min_volume)
    return 20 * torch.log10(scale.clamp(min=min_scale))


class RelativeVolumeMel(nn.Module):
    def __init__(self, sample_rate: int = 24000, n_mels: int = 80, n_fft: int = 512,
                 hop_length: int = 128, min_relative_volume: float = -25,
                 max_relative_volume: float = 25, max_initial_gain: float = 25,
                 min_activity_volume: float = -25,
                 num_aggregated_bands: int = 4) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            normalized=True, sample_rate=sample_rate, power=2)
        self.min_relative_volume = min_relative_volume
        self.max_relative_volume = max_relative_volume
        self.max_initial_gain = max_initial_gain
        self.min_activity_volume = min_activity_volume
        self.num_aggregated_bands = num_aggregated_bands

    def forward(self, estimate: torch.Tensor, ground_truth: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
        min_scale = db_to_scale(-self.max_initial_gain)
        std = ground_truth.pow(2).mean().sqrt().clamp(min=min_scale)
        z_gt = self.melspec(ground_truth / std).sqrt()
        z_est = self.melspec(estimate / std).sqrt()

        delta = z_gt - z_est
        ref_db = scale_to_db(z_gt, self.min_activity_volume)
        delta_db = scale_to_db(delta.abs(), min_volume=-120)
        relative_db = (delta_db - ref_db).clamp(self.min_relative_volume, self.max_relative_volume)
        dims = list(range(relative_db.dim()))
        dims.remove(dims[-2])
        losses_per_band = relative_db.mean(dim=dims)
        aggregated = [chunk.mean() for chunk in losses_per_band.chunk(self.num_aggregated_bands, dim=0)]
        metrics = {f'rvm_{index}': value for index, value in enumerate(aggregated)}
        metrics['rvm'] = losses_per_band.mean()
        return metrics
