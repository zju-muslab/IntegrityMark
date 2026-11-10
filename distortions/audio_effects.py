# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import random
import typing as tp
from functools import partial

import julius
import omegaconf
import torch
import torchaudio
from julius import fft_conv1d, resample_frac

from .audio_utils import get_aac, get_mp3, get_opus










def select_audio_effects(
    audio_effects: tp.Dict,
    weights: tp.Optional[tp.Dict] = None,
    mode: str = "all",
    max_length: tp.Optional[int] = None,
):


    if mode == "all":
        out = audio_effects
    elif mode=='none':
        out = {}
    elif mode == "weighted":

        assert weights is not None
        out = {
            name: value
            for name, value in audio_effects.items()
            if random.random() < weights.get(name, 1.0)
        }
    else:
        raise ValueError(f"Unknown mode {mode}")


    if max_length is not None:

        if len(list(out.keys()))>=max_length:
            random_keys = random.sample(list(out.keys()), max_length)
            out = {key: out[key] for key in random_keys}


    if len(out) == 0:
        out = {"identity": AudioEffects.identity}
    return out

def get_audio_effects(cfg: omegaconf.DictConfig):


    assert hasattr(cfg, "audio_effects")
    cfg_audio_effects = dict(cfg["audio_effects"])



    return {
        name: partial(value, **cfg_audio_effects.get(name, {}))
        for name, value in inspect.getmembers(AudioEffects)
        if inspect.isfunction(value)
    }

def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    if torch.isnan(tensor).any():
        raise ValueError("NaN detected in tensor, terminating program.")


    if tensor.abs().max() >= 1:
        tensor=tensor/(tensor.abs().max()+ 1e-6)


    if mask is None:
        return tensor
    else:
        return tensor, mask


def generate_pink_noise(length: int) -> torch.Tensor:

    num_rows = 16


    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]


    pink_noise = reshaped_array / (torch.max(torch.abs(reshaped_array))+1e-6)
    return pink_noise


def compress_with_encodec(
    tensor: torch.Tensor,
    n_q: int,
    model: "CompressionModel",
    sample_rate: int,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    model.to(tensor.device)


    model.set_num_codebooks(n_q)


    codes, scale = model.encode(
        julius.resample_frac(tensor, old_sr=sample_rate, new_sr=model.sample_rate)
    )


    compressed = model.decode(codes=codes, scale=scale)


    return audio_effect_return(
        tensor=julius.resample_frac(
            compressed, old_sr=model.sample_rate, new_sr=sample_rate
        ),
        mask=mask,
    )


def apply_compression_skip_grad(tensor: torch.Tensor, compression_fn, **kwargs):

    compressed = compression_fn(tensor.detach(), **kwargs)


    compressed = compressed[:, :, : tensor.size(-1)]




    out = tensor + (compressed - tensor).detach()


    if out.requires_grad:
        assert (
            out.grad_fn
        ), "The computation graph might be broken due to compression augmentation."

    return out






class AudioEffects:





    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed_range: tuple = (0.5, 1.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        speed = torch.FloatTensor(1).uniform_(*speed_range)



        new_sr = int(sample_rate * 1 / speed)
        resampled_tensor = julius.resample.resample_frac(tensor, sample_rate, new_sr)


        if mask is None:
            return resampled_tensor
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )





    @staticmethod
    def updownresample(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        intermediate_freq: int = 32000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:


        orig_length = tensor.shape[-1]


        tensor = torchaudio.functional.resample(tensor, sample_rate, intermediate_freq)



        tensor = torchaudio.functional.resample(tensor, intermediate_freq, sample_rate)


        curr_length = tensor.shape[-1]
        if curr_length > orig_length:
            tensor = tensor[..., :orig_length]
        elif curr_length < orig_length:
            pad_size = orig_length - curr_length
            tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)

        return audio_effect_return(tensor=tensor, mask=mask)


    @staticmethod
    def quantize_audio(
        tensor: torch.Tensor,
        bits_range: tuple = (6,12),
        mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        max_val = torch.max(torch.abs(tensor))


        bits=torch.randint(bits_range[0],bits_range[1],(1,))[0].to(tensor.device)


        step = 2 * max_val / (2**bits - 1)


        quant = torch.round(tensor / step) * step


        out = tensor + (quant - tensor).detach()

        return audio_effect_return(tensor=out, mask=mask)





    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:



        duration = torch.FloatTensor(1).uniform_(*duration_range)
        volume = torch.FloatTensor(1).uniform_(*volume_range)

        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)


        impulse_response[0] = 1.0

        impulse_response[
            int(sample_rate * duration) - 1
        ] = volume


        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)


        reverbed_signal = fft_conv1d(tensor, impulse_response)


        reverbed_signal = (
            reverbed_signal
            / (torch.max(torch.abs(reverbed_signal))+1e-6)
            * torch.max(torch.abs(tensor))
        )


        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp

        return audio_effect_return(tensor=reverbed_signal, mask=mask)





    @staticmethod
    def random_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        noise = torch.randn_like(waveform) * noise_std


        noisy_waveform = waveform + noise
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def pink_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.01,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        noise = generate_pink_noise(waveform.shape[-1])


        current_std = torch.std(noise)


        scaled_noise = noise * (noise_std / (current_std + 1e-6))
        scaled_noise = scaled_noise.to(waveform.device)
        noise = scaled_noise.to(waveform.device)



        noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0).to(waveform.device)
        return audio_effect_return(tensor=noisy_waveform, mask=mask)





    @staticmethod
    def lowpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 5000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(
            tensor=julius.lowpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def highpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 500,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(
            tensor=julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def bandpass_filter(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        return audio_effect_return(
            tensor=julius.bandpass_filter(
                waveform,
                cutoff_low=cutoff_freq_low / sample_rate,
                cutoff_high=cutoff_freq_high / sample_rate,
            ),
            mask=mask,
        )





    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))


        kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
        kernel = kernel.to(tensor.device)


        smoothed = fft_conv1d(tensor, kernel)


        tmp = torch.zeros_like(tensor)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return audio_effect_return(tensor=smoothed, mask=mask)

    @staticmethod
    def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor * (1 + amount / 100), mask=mask)

    @staticmethod
    def duck_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)





    @staticmethod
    def identity(
        tensor: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor, mask=mask)





    @staticmethod
    def mp3_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        bitrate_range: tp.Optional[tp.List[str]] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        if bitrate_range is not None and len(bitrate_range) > 0:
            bitrate = random.choice(bitrate_range)



        out = apply_compression_skip_grad(
            tensor, get_mp3, sr=sample_rate, bitrate=bitrate
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def aac_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        bitrate_range: tp.Optional[tp.List[str]] = None,
        lowpass_freq: tp.Optional[int] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        if bitrate_range is not None and len(bitrate_range) > 0:
            bitrate = random.choice(bitrate_range)



        out = apply_compression_skip_grad(
            tensor, get_aac, sr=sample_rate, bitrate=bitrate, lowpass_freq=lowpass_freq
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def opus_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "64k",
        bitrate_range: tp.Optional[tp.List[str]] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        if bitrate_range is not None and len(bitrate_range) > 0:
            bitrate = random.choice(bitrate_range)

        out = apply_compression_skip_grad(
            tensor, get_opus, sr=sample_rate, bitrate=bitrate
        )
        return audio_effect_return(tensor=out, mask=mask)
