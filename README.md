# IntegrityMark

This is an official implementation of the paper **"IntegrityMark: Tamper-Resistant Watermarking for Real-Time Audio Integrity Protection"**

IntegrityMark is a research codebase for robust speech watermarking, watermark detection, and low-latency streaming watermark injection. The repository contains training, evaluation, detection, and client/server streaming utilities built around the project's `WMEmbedder` and `WMExtractor` models.

## Repository Layout

```text
.
├── config/                  Hydra configs for training and evaluation
├── dataset/                 Dataset loaders and helpers
├── distortions/             Audio distortions and augmentations
├── eval/                    Comprehensive evaluation scripts
├── losses/                  Training losses
├── models/                  Watermark model definitions
├── utils/                   Shared utilities
├── train.py                 Main training entrypoint
├── pyproject.toml           Project metadata and dependencies (pip install uses this)
├── requirements.txt         Convenience shortcut to install the project
├── checkpoints/             (Create manually) Directory for saving trained models
└── pretrained/              (Create manually) Directory for pretrained models
```

## Environment Setup

Follow these 4 steps to set up the environment and download required data.

### Step 1: Clone Repository

```bash
git clone https://github.com/<your-org>/IntegrityMark.git
cd IntegrityMark
```

### Step 2: Install Dependencies

All dependencies are defined in `pyproject.toml`. Install the base package:

```bash
pip install -U pip
pip install -e .
```

Optional: For streaming/real-time features (WebSocket client/server):
```bash
pip install -e .[stream]
```

**Core dependencies** (automatically installed):
- PyTorch 2.1+, torchaudio 2.1+
- Hydra, OmegaConf (configuration management)
- librosa, julius (audio processing)
- numpy, scipy, pandas (data processing)
- PESQ, PySTOI, matplotlib (evaluation and visualization)
- wandb (experiment tracking)

**Streaming optional dependencies**:
- sounddevice, websockets (real-time audio I/O)

### Step 3: Download Pretrained Models

Download the required pretrained models from Hugging Face:

```bash
mkdir -p pretrained

# Download PASE+ (~170 MB)
git clone https://huggingface.co/asappresearch/pase-plus pretrained/pase+

# Download Wav2Vec2 (~360 MB)
git clone https://huggingface.co/facebook/wav2vec2-base-960h pretrained/wav2vec2-base-960h
```

### Step 4: Download Training Dataset

Choose one of the recommended datasets and download it:

**Option A: LibriSpeech (Recommended)**
```bash
# Download from http://www.openslr.org/12
# Extract to:
mkdir -p dataset/LibriSpeech
# Place downloaded files in dataset/LibriSpeech/
```

**Option B: Common Voice**
```bash
# Download from https://commonvoice.mozilla.org
# Extract to:
mkdir -p dataset/CommonVoice
# Place downloaded files in dataset/CommonVoice/
```

**Option C: Your Own Audio**
```bash
# Create directory with your audio files
mkdir -p data/speech
# Add your WAV files (16kHz sample rate)
```

---

## Implementation Notes

This implementation follows the IntegrityMark paper ("Tamper-Resistant Watermarking for Real-Time Audio Integrity Protection") with the following key parameters:

**Paper vs. Implementation**:
- Dataset: LibriSpeech (clean, 360 hours) - ✓ matches paper
- Training epochs: 200 - ✓ matches paper
- Watermark message: 4-bit validation (V=4) - ✓ matches paper
- Segment lengths: 512-2064 samples, increment 32 - ✓ matches paper
- **Batch size**: 6 (paper specifies 12, optimized for GPU memory)
- **Learning rate**: 1×10⁻⁴ (paper specifies 1×10⁻³, empirically tuned)

All loss functions, model architectures (Demucs + SeaNet), and curriculum learning strategy are implemented exactly as specified in the paper.

---

# PART 1: Steps to Train the Model

IntegrityMark uses **curriculum learning** strategy for robust watermarking. The training is organized in three progressive stages:

- **Stage 1**: Basic watermark embedding (0-10k steps) - focus on learning to embed watermarks without augmentation
- **Stage 2**: Robustness training (10k+ steps) - add data augmentation and progressive difficulty
- **Stage 3**: Final optimization - maximize audio quality with full augmentation

## Quick Start: Train with Default Configuration

Train using the main configuration:

```bash
python train.py --config-name 0+0+4
```

## Curriculum Training Strategy

For best results, follow the three-stage curriculum approach:

### Stage 1: Basic Watermark Embedding (No Augmentation)

Start with stage 1 config to learn basic watermarking patterns without data augmentation:

```bash
# Stage 1: Basic embedding, disable augmentation, lower perceptual loss
python train.py --config-name 0+0+4_stage1_basic

# After training, save the checkpoint
mkdir -p checkpoints
cp outputs/ckpt/0+0+4_stage1/0+0+4_stage1_*.pth checkpoints/stage1_checkpoint.pth
```

**Duration**: ~10,000 steps (about 10-20 epochs depending on dataset size)
**Data augmentation**: DISABLED


### Stage 2: Robustness Training (Augmentation Enabled)

Continue from stage 1 with augmentation enabled for robustness:

```bash
# Stage 2: Enable augmentation, progressive loss increase
python train.py --config-name 0+0+4_stage2_augment \
  continue_from_ckpt=checkpoints/stage1_checkpoint.pth

# After training, save the checkpoint
cp outputs/ckpt/0+0+4_stage2/0+0+4_stage2_*.pth checkpoints/stage2_checkpoint.pth
```

### Stage 3: Final Optimization

Fine-tune with maximum audio quality constraints:

```bash
# Stage 3: Full optimization, maximum perceptual loss
python train.py --config-name 0+0+4_stage3_final \
  continue_from_ckpt=checkpoints/stage2_checkpoint.pth

```


<!-- 
## Training with Custom Dataset

You can override the dataset path at any stage:

```bash
python train.py --config-name 0+0+4_stage1_basic \
  dataset.path.raw_path=/path/to/your/LibriSpeech
```

Supported dataset formats:
- **LibriSpeech**: `dataset/LibriSpeech` (recommended)
- **CommonVoice**: `dataset/CommonVoice`
- **Custom audio**: `data/speech` (WAV files, 16kHz sample rate)

## Disable W&B (Optional, Recommended for Quick Testing)

```bash
export WANDB_DISABLED=true
python train.py --config-name 0+0+4_stage1_basic
```

## Available Training Configurations

| Config Name | Purpose | Use Case |
|------------|---------|----------|
| `0+0+4_stage1_basic.yaml` | Stage 1: Basic embedding | Initial watermark learning (no augmentation) |
| `0+0+4_stage2_augment.yaml` | Stage 2: Robustness | Add robustness to distortions |
| `0+0+4_stage3_final.yaml` | Stage 3: Optimization | Final quality optimization |
| `0+0+4.yaml` | Full training | One-stage training (for quick testing) |
| `basic.yaml` | Minimal config | Quick tests and debugging |
| `sweep.yaml` | Hyperparameter sweep | Grid/random search over parameters |

View all configurations:
```bash
python train.py --help
``` -->

## Training Outputs

Training generates outputs in `outputs/` directory:

```
outputs/
├── ckpt/          # Model checkpoints (saved every epoch)
├── logs/          # Training logs and metrics
└── wav/           # Generated audio samples during training
```
<!-- 
### Save Important Checkpoints

```bash
# After each stage, save the checkpoint for next stage
mkdir -p checkpoints
cp outputs/ckpt/0+0+4_stage1/0+0+4_stage1_*.pth checkpoints/stage1_model.pth
cp outputs/ckpt/0+0+4_stage2/0+0+4_stage2_*.pth checkpoints/stage2_model.pth
cp outputs/ckpt/0+0+4_stage3/0+0+4_stage3_*.pth checkpoints/final_model.pth
``` -->

---

# PART 2: Steps to Evaluate the Model

## Quick Evaluation

```bash
# Basic evaluation
python eval/eval.py ckpt_path=checkpoints/model.pth
```

## Detailed Evaluations

```bash
# 1. Basic quality evaluation
python eval/eval.py ckpt_path=checkpoints/model.pth

# 2. Distortion robustness
python eval/eval_distortion.py ckpt_path=checkpoints/model.pth

# 3. Cross-source attack detection
python eval/eval_cross_source_attack.py ckpt_path=checkpoints/model.pth

# 4. In-source attack detection
python eval/eval_in_source_attack.py ckpt_path=checkpoints/model.pth

# 5. Cross-domain evaluation
python eval/eval_cross_domain.py ckpt_path=checkpoints/model.pth

# 6. Language transfer evaluation
python eval/eval_language.py ckpt_path=checkpoints/model.pth

# 7. Detection speed benchmarking
python eval/eval_detection_speed.py ckpt_path=checkpoints/model.pth
```
<!-- 
## Evaluation Scripts Overview

| Script | Purpose | Key Metrics |
|--------|---------|------------|
| `eval.py` | Basic quality & watermark | SNR, PESQ, STOI, detection accuracy |
| `eval_distortion.py` | Robustness to distortions | Accuracy under noise, compression, filtering |
| `eval_cross_source_attack.py` | Cross-source tampering | TPR, FPR, TIOU for insertion/replacement |
| `eval_in_source_attack.py` | In-source tampering | TPR, FPR, TIOU for deletion/copy |
| `eval_cross_domain.py` | Domain generalization | Performance across audio domains |
| `eval_language.py` | Language transfer | Performance across languages |
| `eval_detection_speed.py` | Detection latency | Inference time, throughput, memory |

## Specialized Evaluations

### Distortion Robustness

Test watermark resilience to audio distortions:

```bash
python eval/eval_distortion.py ckpt_path=checkpoints/model.pth
``` -->
