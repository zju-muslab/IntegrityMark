# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torchmetrics

from ..data.audio_utils import convert_audio
from ..modules.chroma import ChromaExtractor


class ChromaCosineSimilarityMetric(torchmetrics.Metric):
    def __init__(self, sample_rate: int, n_chroma: int, radix2_exp: int, argmax: bool, eps: float = 1e-8):
        super().__init__()
        self.chroma_sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.eps = eps
        self.chroma_extractor = ChromaExtractor(sample_rate=self.chroma_sample_rate, n_chroma=self.n_chroma,
                                                radix2_exp=radix2_exp, argmax=argmax)
        self.add_state("cosine_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        if preds.size(0) == 0:
            return

        assert preds.shape == targets.shape, (
            f"Preds and target shapes mismatch: preds={preds.shape}, targets={targets.shape}")
        assert preds.size(0) == sizes.size(0), (
            f"Number of items in preds ({preds.shape}) mismatch ",
            f"with sizes ({sizes.shape})")
        assert preds.size(0) == sample_rates.size(0), (
            f"Number of items in preds ({preds.shape}) mismatch ",
            f"with sample_rates ({sample_rates.shape})")
        assert torch.all(sample_rates == sample_rates[0].item()), "All sample rates are not the same in the batch"

        device = self.weight.device
        preds, targets = preds.to(device), targets.to(device)
        sample_rate = sample_rates[0].item()
        preds = convert_audio(preds, from_rate=sample_rate, to_rate=self.chroma_sample_rate, to_channels=1)
        targets = convert_audio(targets, from_rate=sample_rate, to_rate=self.chroma_sample_rate, to_channels=1)
        gt_chroma = self.chroma_extractor(targets)
        gen_chroma = self.chroma_extractor(preds)
        chroma_lens = (sizes / self.chroma_extractor.winhop).ceil().int()
        for i in range(len(gt_chroma)):
            t = int(chroma_lens[i].item())
            cosine_sim = torch.nn.functional.cosine_similarity(
                gt_chroma[i, :t], gen_chroma[i, :t], dim=1, eps=self.eps)
            self.cosine_sum += cosine_sim.sum(dim=0)
            self.weight += torch.tensor(t)

    def compute(self) -> float:
        assert self.weight.item() > 0, "Unable to compute with total number of comparisons <= 0"
        return (self.cosine_sum / self.weight).item()
