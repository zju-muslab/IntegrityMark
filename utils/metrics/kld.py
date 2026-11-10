# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
from functools import partial
import logging
import os
import typing as tp

import torch
import torchmetrics

from ..data.audio_utils import convert_audio


logger = logging.getLogger(__name__)


class _patch_passt_stft:
    def __init__(self):
        self.old_stft = torch.stft

    def __enter__(self):


        torch.stft = partial(torch.stft, return_complex=False)

    def __exit__(self, *exc):
        torch.stft = self.old_stft


def kl_divergence(pred_probs: torch.Tensor, target_probs: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    kl_div = torch.nn.functional.kl_div((pred_probs + epsilon).log(), target_probs, reduction="none")
    return kl_div.sum(-1)


class KLDivergenceMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("kld_pq_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("kld_qp_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("kld_all_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0), dist_reduce_fx="sum")

    def _get_label_distribution(self, x: torch.Tensor, sizes: torch.Tensor,
                                sample_rates: torch.Tensor) -> tp.Optional[torch.Tensor]:
        raise NotImplementedError("implement method to extract label distributions from the model.")

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        assert preds.shape == targets.shape
        assert preds.size(0) > 0, "Cannot update the loss with empty tensors"
        preds_probs = self._get_label_distribution(preds, sizes, sample_rates)
        targets_probs = self._get_label_distribution(targets, sizes, sample_rates)
        if preds_probs is not None and targets_probs is not None:
            assert preds_probs.shape == targets_probs.shape
            kld_scores = kl_divergence(preds_probs, targets_probs)
            assert not torch.isnan(kld_scores).any(), "kld_scores contains NaN value(s)!"
            self.kld_pq_sum += torch.sum(kld_scores)
            kld_qp_scores = kl_divergence(targets_probs, preds_probs)
            self.kld_qp_sum += torch.sum(kld_qp_scores)
            self.weight += torch.tensor(kld_scores.size(0))

    def compute(self) -> dict:
        weight: float = float(self.weight.item())
        assert weight > 0, "Unable to compute with total number of comparisons <= 0"
        logger.info(f"Computing KL divergence on a total of {weight} samples")
        kld_pq = self.kld_pq_sum.item() / weight
        kld_qp = self.kld_qp_sum.item() / weight
        kld_both = kld_pq + kld_qp
        return {'kld': kld_pq, 'kld_pq': kld_pq, 'kld_qp': kld_qp, 'kld_both': kld_both}


class PasstKLDivergenceMetric(KLDivergenceMetric):
    def __init__(self, pretrained_length: tp.Optional[float] = None):
        super().__init__()
        self._initialize_model(pretrained_length)

    def _initialize_model(self, pretrained_length: tp.Optional[float] = None):
        model, sr, max_frames, min_frames = self._load_base_model(pretrained_length)
        self.min_input_frames = min_frames
        self.max_input_frames = max_frames
        self.model_sample_rate = sr
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def _load_base_model(self, pretrained_length: tp.Optional[float]):
        try:
            if pretrained_length == 30:
                from hear21passt.base30sec import get_basic_model
                max_duration = 30
            elif pretrained_length == 20:
                from hear21passt.base20sec import get_basic_model
                max_duration = 20
            else:
                from hear21passt.base import get_basic_model

                max_duration = 10
            min_duration = 0.15
            min_duration = 0.15
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install hear21passt to compute KL divergence: ",
                "pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt'"
            )
        model_sample_rate = 32_000
        max_input_frames = int(max_duration * model_sample_rate)
        min_input_frames = int(min_duration * model_sample_rate)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            model = get_basic_model(mode='logits')
        return model, model_sample_rate, max_input_frames, min_input_frames

    def _process_audio(self, wav: torch.Tensor, sample_rate: int, wav_len: int) -> tp.List[torch.Tensor]:
        wav = wav.unsqueeze(0)
        wav = wav[..., :wav_len]
        wav = convert_audio(wav, from_rate=sample_rate, to_rate=self.model_sample_rate, to_channels=1)
        wav = wav.squeeze(0)

        segments = torch.split(wav, self.max_input_frames, dim=-1)
        valid_segments = []
        for s in segments:

            if s.size(-1) > self.min_input_frames:
                valid_segments.append(s)
        return [s[None] for s in valid_segments]

    def _get_model_preds(self, wav: torch.Tensor) -> torch.Tensor:
        assert wav.dim() == 3, f"Unexpected number of dims for preprocessed wav: {wav.shape}"
        wav = wav.mean(dim=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            with torch.no_grad(), _patch_passt_stft():
                logits = self.model(wav.to(self.device))
                probs = torch.softmax(logits, dim=-1)
                return probs

    def _get_label_distribution(self, x: torch.Tensor, sizes: torch.Tensor,
                                sample_rates: torch.Tensor) -> tp.Optional[torch.Tensor]:
        all_probs: tp.List[torch.Tensor] = []
        for i, wav in enumerate(x):
            sample_rate = int(sample_rates[i].item())
            wav_len = int(sizes[i].item())
            wav_segments = self._process_audio(wav, sample_rate, wav_len)
            for segment in wav_segments:
                probs = self._get_model_preds(segment).mean(dim=0)
                all_probs.append(probs)
        if len(all_probs) > 0:
            return torch.stack(all_probs, dim=0)
        else:
            return None
