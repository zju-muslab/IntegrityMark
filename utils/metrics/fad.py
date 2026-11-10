# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
import os
import subprocess
import tempfile
import typing as tp

from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
import flashy
import torch
import torchmetrics

from ..environment import AudioCraftEnvironment


logger = logging.getLogger(__name__)

VGGISH_SAMPLE_RATE = 16_000
VGGISH_CHANNELS = 1


class FrechetAudioDistanceMetric(torchmetrics.Metric):
    def __init__(self, bin: tp.Union[Path, str], model_path: tp.Union[Path, str],
                 format: str = "wav", batch_size: tp.Optional[int] = None,
                 log_folder: tp.Optional[tp.Union[Path, str]] = None):
        super().__init__()
        self.model_sample_rate = VGGISH_SAMPLE_RATE
        self.model_channels = VGGISH_CHANNELS
        self.model_path = AudioCraftEnvironment.resolve_reference_path(model_path)
        assert Path(self.model_path).exists(), f"Could not find provided model checkpoint path at: {self.model_path}"
        self.format = format
        self.batch_size = batch_size
        self.bin = bin
        self.tf_env = {"PYTHONPATH": str(self.bin)}
        self.python_path = os.environ.get('TF_PYTHON_EXE') or 'python'
        logger.info("Python exe for TF is  %s", self.python_path)
        if 'TF_LIBRARY_PATH' in os.environ:
            self.tf_env['LD_LIBRARY_PATH'] = os.environ['TF_LIBRARY_PATH']
        if 'TF_FORCE_GPU_ALLOW_GROWTH' in os.environ:
            self.tf_env['TF_FORCE_GPU_ALLOW_GROWTH'] = os.environ['TF_FORCE_GPU_ALLOW_GROWTH']
        logger.info("Env for TF is %r", self.tf_env)
        self.reset(log_folder)
        self.add_state("total_files", default=torch.tensor(0.), dist_reduce_fx="sum")

    def reset(self, log_folder: tp.Optional[tp.Union[Path, str]] = None):
        log_folder = Path(log_folder or tempfile.mkdtemp())
        self.tmp_dir = log_folder / 'fad'
        self.tmp_dir.mkdir(exist_ok=True)
        self.samples_tests_dir = self.tmp_dir / 'tests'
        self.samples_tests_dir.mkdir(exist_ok=True)
        self.samples_background_dir = self.tmp_dir / 'background'
        self.samples_background_dir.mkdir(exist_ok=True)
        self.manifest_tests = self.tmp_dir / 'files_tests.cvs'
        self.manifest_background = self.tmp_dir / 'files_background.cvs'
        self.stats_tests_dir = self.tmp_dir / 'stats_tests'
        self.stats_background_dir = self.tmp_dir / 'stats_background'
        self.counter = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               sizes: torch.Tensor, sample_rates: torch.Tensor,
               stems: tp.Optional[tp.List[str]] = None):
        assert preds.shape == targets.shape, f"preds={preds.shape} != targets={targets.shape}"
        num_samples = preds.shape[0]
        assert num_samples == sizes.size(0) and num_samples == sample_rates.size(0)
        assert stems is None or num_samples == len(set(stems))
        for i in range(num_samples):
            self.total_files += 1
            self.counter += 1
            wav_len = int(sizes[i].item())
            sample_rate = int(sample_rates[i].item())
            pred_wav = preds[i]
            target_wav = targets[i]
            pred_wav = pred_wav[..., :wav_len]
            target_wav = target_wav[..., :wav_len]
            stem_name = stems[i] if stems is not None else f'sample_{self.counter}_{flashy.distrib.rank()}'

            try:
                pred_wav = convert_audio(
                    pred_wav.unsqueeze(0), from_rate=sample_rate,
                    to_rate=self.model_sample_rate, to_channels=1).squeeze(0)
                audio_write(
                    self.samples_tests_dir / stem_name, pred_wav, sample_rate=self.model_sample_rate,
                    format=self.format, strategy="peak")
            except Exception as e:
                logger.error(f"Exception occured when saving tests files for FAD computation: {repr(e)} - {e}")
            try:


                target_wav = convert_audio(
                    target_wav.unsqueeze(0), from_rate=sample_rate,
                    to_rate=self.model_sample_rate, to_channels=1).squeeze(0)
                audio_write(
                    self.samples_background_dir / stem_name, target_wav, sample_rate=self.model_sample_rate,
                    format=self.format, strategy="peak")
            except Exception as e:
                logger.error(f"Exception occured when saving background files for FAD computation: {repr(e)} - {e}")

    def _get_samples_name(self, is_background: bool):
        return 'background' if is_background else 'tests'

    def _create_embedding_beams(self, is_background: bool, gpu_index: tp.Optional[int] = None):
        if is_background:
            input_samples_dir = self.samples_background_dir
            input_filename = self.manifest_background
            stats_name = self.stats_background_dir
        else:
            input_samples_dir = self.samples_tests_dir
            input_filename = self.manifest_tests
            stats_name = self.stats_tests_dir
        beams_name = self._get_samples_name(is_background)
        log_file = self.tmp_dir / f'fad_logs_create_beams_{beams_name}.log'

        logger.info(f"Scanning samples folder to fetch list of files: {input_samples_dir}")
        with open(input_filename, "w") as fout:
            for path in Path(input_samples_dir).glob(f"*.{self.format}"):
                fout.write(f"{str(path)}\n")

        cmd = [
            self.python_path, "-m",
            "frechet_audio_distance.create_embeddings_main",
            "--model_ckpt", f"{self.model_path}",
            "--input_files", f"{str(input_filename)}",
            "--stats", f"{str(stats_name)}",
        ]
        if self.batch_size is not None:
            cmd += ["--batch_size", str(self.batch_size)]
        logger.info(f"Launching frechet_audio_distance embeddings main method: {' '.join(cmd)} on {beams_name}")
        env = os.environ
        if gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        process = subprocess.Popen(
            cmd, stdout=open(log_file, "w"), env={**env, **self.tf_env}, stderr=subprocess.STDOUT)
        return process, log_file

    def _compute_fad_score(self, gpu_index: tp.Optional[int] = None):
        cmd = [
            self.python_path, "-m", "frechet_audio_distance.compute_fad",
            "--test_stats", f"{str(self.stats_tests_dir)}",
            "--background_stats", f"{str(self.stats_background_dir)}",
        ]
        logger.info(f"Launching frechet_audio_distance compute fad method: {' '.join(cmd)}")
        env = os.environ
        if gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        result = subprocess.run(cmd, env={**env, **self.tf_env}, capture_output=True)
        if result.returncode:
            logger.error(
                "Error with FAD computation from stats: \n %s \n %s",
                result.stdout.decode(), result.stderr.decode()
            )
            raise RuntimeError("Error while executing FAD computation from stats")
        try:

            fad_score = float(result.stdout[4:])
            return fad_score
        except Exception as e:
            raise RuntimeError(f"Error parsing FAD score from command stdout: {e}")

    def _log_process_result(self, returncode: int, log_file: tp.Union[Path, str], is_background: bool) -> None:
        beams_name = self._get_samples_name(is_background)
        if returncode:
            with open(log_file, "r") as f:
                error_log = f.read()
                logger.error(error_log)
            os._exit(1)
        else:
            logger.info(f"Successfully computed embedding beams on {beams_name} samples.")

    def _parallel_create_embedding_beams(self, num_of_gpus: int):
        assert num_of_gpus > 0
        logger.info("Creating embeddings beams in a parallel manner on different GPUs")
        tests_beams_process, tests_beams_log_file = self._create_embedding_beams(is_background=False, gpu_index=0)
        bg_beams_process, bg_beams_log_file = self._create_embedding_beams(is_background=True, gpu_index=1)
        tests_beams_code = tests_beams_process.wait()
        bg_beams_code = bg_beams_process.wait()
        self._log_process_result(tests_beams_code, tests_beams_log_file, is_background=False)
        self._log_process_result(bg_beams_code, bg_beams_log_file, is_background=True)

    def _sequential_create_embedding_beams(self):
        logger.info("Creating embeddings beams in a sequential manner")
        tests_beams_process, tests_beams_log_file = self._create_embedding_beams(is_background=False)
        tests_beams_code = tests_beams_process.wait()
        self._log_process_result(tests_beams_code, tests_beams_log_file, is_background=False)
        bg_beams_process, bg_beams_log_file = self._create_embedding_beams(is_background=True)
        bg_beams_code = bg_beams_process.wait()
        self._log_process_result(bg_beams_code, bg_beams_log_file, is_background=True)

    @flashy.distrib.rank_zero_only
    def _local_compute_frechet_audio_distance(self):
        num_of_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_of_gpus > 1:
            self._parallel_create_embedding_beams(num_of_gpus)
        else:
            self._sequential_create_embedding_beams()
        fad_score = self._compute_fad_score(gpu_index=0)
        return fad_score

    def compute(self) -> float:
        assert self.total_files.item() > 0, "No files dumped for FAD computation!"
        fad_score = self._local_compute_frechet_audio_distance()
        logger.warning(f"FAD score = {fad_score}")
        fad_score = flashy.distrib.broadcast_object(fad_score, src=0)
        return fad_score
