# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import io
import logging
import re
import sys
import typing as tp

import julius
import torch
import torchaudio
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:



        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:



        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:



        wav = wav[..., :channels, :]
    else:

        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()

    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)


def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav


def compress(wav: torch.Tensor, sr: int,
             target_format: tp.Literal["mp3", "ogg", "flac"] = "mp3",
             bitrate: str = "128k") -> tp.Tuple[torch.Tensor, int]:


    match = re.search(r"\d+(\.\d+)?", str(bitrate))
    parsed_bitrate = float(match.group()) if match else None
    assert parsed_bitrate, f"Invalid bitrate specified (got {parsed_bitrate})"
    try:

        buffer = io.BytesIO()
        torchaudio.save(
            buffer, wav, sr, format=target_format, bits_per_sample=parsed_bitrate,
        )

        buffer.seek(0)
        compressed_wav, sr = torchaudio.load(buffer)
        return compressed_wav, sr

    except RuntimeError:
        logger.warning(
            f"compression failed skipping compression: {format} {parsed_bitrate}"
        )
        return wav, sr


def MP3_compress_with_bitrate(wav, sr, bitrate="64k"):

    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, sr, format="wav")
    buffer.seek(0)


    audio = AudioSegment.from_file(buffer, format="wav")
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3", bitrate=bitrate)
    mp3_buffer.seek(0)



    compressed_wav, compressed_sr = torchaudio.load(mp3_buffer)

    return compressed_wav, compressed_sr


def get_mp3(wav_tensor: torch.Tensor, sr: int, bitrate: str = "128k") -> torch.Tensor:
    device = wav_tensor.device
    batch_size, channels, length = wav_tensor.shape

    wav_tensor_flat = wav_tensor.reshape(1, -1).cpu()

    original_length=length*batch_size

    wav_tensor_flat, _ = MP3_compress_with_bitrate(wav_tensor_flat, sr, bitrate=bitrate)


    compressed_length = wav_tensor_flat.shape[-1]
    if compressed_length > original_length:
        wav_tensor_flat = wav_tensor_flat[..., :original_length]
    elif compressed_length < original_length:
        padding = torch.zeros(
            0, original_length - compressed_length, device=device
        )
        wav_tensor_flat = torch.cat((wav_tensor_flat, padding), dim=-1)

    wav_tensor = wav_tensor_flat.reshape(batch_size, channels, -1)


    return wav_tensor.to(device)


def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: tp.Optional[int] = None,
    encoder_delay = 1024,
) -> torch.Tensor:
    import tempfile
    import subprocess

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape


    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = (
        match.group() if match else "128"
    )


    wav_tensor_flat = wav_tensor.reshape(1, -1).cpu()

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name


        torchaudio.save(input_path, wav_tensor_flat, sr, backend="ffmpeg")


        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        try:

            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


            aac_tensor, _ = torchaudio.load(output_path, backend="ffmpeg")
        except Exception as exc:
            raise RuntimeError(
                "Failed to run command "  ".join(command)} "
                "(Often this means ffmpeg is not installed or the encoder is not supported, "
                "make sure you installed an older version ffmpeg<5)"
            ) from exc

    original_length_flat = batch_size * channels * original_length

    compressed_length_flat = aac_tensor.shape[-1]


    if compressed_length_flat > original_length_flat:
        aac_tensor = aac_tensor[:, :original_length_flat]


    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)


    wav_tensor = aac_tensor.reshape(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)


def get_opus(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "64k"
    "",
    encoder_delay: int = 312,
) -> torch.Tensor:
    import tempfile
    import subprocess
    import os

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape


    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = (
        match.group() if match else "64"
    )


    wav_tensor_flat = wav_tensor.reshape(1, -1).cpu()


    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f_out:
        input_path, output_path = f_in.name, f_out.name

    try:

        torchaudio.save(input_path, wav_tensor_flat, sr)


        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "libopus",
            "-f",
            "ogg",
            output_path,
        ]


        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed with return code {result.returncode}. "
                f"stderr: {result.stderr.decode('utf-8', errors='ignore')}"
            )


        opus_tensor, _ = torchaudio.load(output_path)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to run Opus compression. "
            f"Make sure ffmpeg is installed with libopus support. "
            f"Check with: ffmpeg -encoders | grep opus"
        ) from exc
    finally:

        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

    original_length_flat = batch_size * channels * original_length

    compressed_length_flat = opus_tensor.shape[-1]


    if compressed_length_flat > original_length_flat:
        opus_tensor = opus_tensor[:, :original_length_flat]


    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        opus_tensor = torch.cat((opus_tensor, padding), dim=-1)


    wav_tensor = opus_tensor.reshape(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "Opus-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not installed and used as proper backend "
        "for torchaudio, or the Opus encoder is not correct."
    )
    return wav_tensor.to(device)
