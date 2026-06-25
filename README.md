# FALCON — Forced Alignment through Contrastive Optimization Networks

[![arXiv](https://img.shields.io/badge/arXiv-2606.25460-b31b1b.svg)](https://arxiv.org/abs/2606.25460) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**FALCON** (Forced Alignment through Contrastive Optimization Networks) is an end-to-end, fully differentiable neural system for **phoneme-level and word-level (and also multilingual) forced alignment** — given a speech waveform and a known phoneme/words transcript sequence, it predicts precise phoneme boundary timestamps.

> Rotem Rousso, Eyal Cohen, Joseph Keshet  
> *"Fully Differentiable Neural Forced Alignment via Soft Dynamic Programming"*  
> Preprint, 2026 — [arXiv:2606.25460](https://arxiv.org/abs/2606.25460)

---

## Citation

If you use FALCON in your research, please cite:

```bibtex
@article{rousso2026falcon,
  title     = {Fully Differentiable Neural Forced Alignment via Soft Dynamic Programming},
  author    = {Rousso, Rotem and Cohen, Eyal and Keshet, Joseph},
  journal   = {arXiv preprint arXiv:2606.25460},
  year      = {2026}
}
```

## Highlights

- 🎯 **Phoneme-level precision** — operates at ~10 ms frame resolution, the finest granularity among neural forced aligners.
- 🔁 **Fully differentiable** — a Soft Dynamic Programming decoder enables gradient flow through the entire alignment pipeline.
- 🌍 **Zero-shot multilingual** — trained on English, generalizes to unseen languages at inference (tested on Dutch, German, and Hebrew can be others too) **without any additional training**.
- 📝 **Word-level generalization** — word boundaries derived from phoneme predictions at test time, competitive with word-level neural aligners.
---

## How It Works

The system has three jointly trained components:

```
Raw waveform (16 kHz)
        │
        ▼
 ┌──────────────────┐
 │ CNN Encoder f_θ  │  learns boundary-sensitive representations
 │ (5 strided conv) │  via MNCE contrastive loss
 └───────┬──────────┘
         │ Z  (frame-level latent features)
         ├──────────────────────────────────────┐
         ▼                                      │ Z
 ┌────────────────────┐                         │
 │ BiLSTM Context g_ψ │  frame-wise phoneme     │
 │ (5 layers, 512 dim)│  posterior probs U      │
 └───────┬────────────┘                         │
         │ U                                    │
         ▼                                      ▼
 ┌───────────────────────────────────────────────────┐
 │        Soft-DP Decoder h_W                        │
 │  φ₁(Z): acoustic boundary sharpness              │
 │  φ₂(U,p): phoneme posterior consistency          │
 │  LogSumExp forward + soft-argmax backtrack       │
 └───────────────────────────────────────────────────┘
         │
         ▼
  Predicted boundary timestamps  ŷ = (ŷ₁, …, ŷₙ)
```

---

## Example

FALCON aligns the example [`fasw0sa2.wav`](assets/fasw0sa2.wav) — a TIMIT sentence, *"Don't ask me to carry an oily rag like that."* — using the included English checkpoint. The same utterance is provided in every supported input format so you can try them in the web demo: [`.phn`](assets/fasw0sa2.phn) (phonemes), [`.wrd`](assets/fasw0sa2.wrd) (words), and [`.txt`](assets/fasw0sa2.txt) (plain transcript).

**Waveform with predicted phoneme boundaries:**

![FALCON phoneme alignment example](assets/example_alignment.png)

**Under the hood — every representation on one shared time axis.** Waveform, log-mel spectrogram, the phoneme posteriors, the Soft-DP cost matrix with its backtracked alignment path, and the contrastive boundary score with its derivative. The *same* predicted boundaries (crimson dashed) and ground-truth boundaries (charcoal dotted) line up across all panels, with the input phonemes written under each axis, so every component's contribution is visible at a glance:

![FALCON aligned representations](assets/example_panels.png)

**Output** (first rows of the `phones` tier; the full alignment is written to a Praat `.TextGrid`):

| Phoneme | Start (s) | End (s) |
|:--|--:|--:|
| h#  | 0.000 | 0.121 |
| d   | 0.121 | 0.141 |
| ow  | 0.141 | 0.313 |
| n   | 0.313 | 0.333 |
| q   | 0.333 | 0.434 |
| ae  | 0.434 | 0.575 |
| s   | 0.575 | 0.645 |
| epi | 0.645 | 0.726 |
| m   | 0.726 | 0.776 |
| iy  | 0.776 | 0.897 |
| …   |       |       |

Reproduce:

```bash
# alignment + TextGrid
python generate_textgrids.py --wav assets/fasw0sa2.wav --mode phoneme --lang english --annotation phn

# the multi-panel "under the hood" figure
python falcon_viz.py
```

---

## Clone Repository

> **Note:** the official public release will be hosted at **[github.com/MLSpeech/FALCON](https://github.com/MLSpeech/FALCON)** (as referenced in the paper). This repository (`github.com/RotemRousso/FDNFA`) is the active development version.

```bash
git clone https://github.com/RotemRousso/FDNFA.git
cd FDNFA
```

---

## Setup Environment

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)

### Option 1 — Conda (recommended)
```bash
conda create -n falcon python=3.8
conda activate falcon
pip install -r requirements.txt
```

### Option 2 — pip only
```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `torchaudio` are pinned to 2.4.1. For a different CUDA version, install them separately first following [PyTorch's official instructions](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Data Format

FALCON expects data in **TIMIT-style** format:

```
<dataset_root>/
├── train/
│   ├── speaker1_utt1.wav
│   ├── speaker1_utt1.phn     ← required
│   └── ...
├── val/                       ← only for Buckeye-style splits
└── test/
    ├── speaker2_utt1.wav
    ├── speaker2_utt1.phn
    └── ...
```

Each `.phn` file contains one phoneme segment per line:
```
<start_sample>  <end_sample>  <phoneme_label>
0     3050  h#
3050  4559  sh
4559  5723  ix
...
```

Audio must be **16 kHz mono WAV**. Phoneme labels should use the TIMIT 61-phoneme set (automatically mapped to Lee-Hon 39 internally).

Supported datasets:
- **TIMIT** — `data: timit` in config (standard train/test split, 10% of train used for validation)
- **Buckeye** — `data: buckeye` in config (explicit `train/`, `val/`, `test/` dirs)

---

## Training

Edit `conf/config.yaml` to set your dataset paths and output directory:

```yaml
# Data
data: timit                                   # or 'buckeye'
timit_path: /path/to/your/timit/
buckeye_path: /path/to/your/buckeye/

# Output run directory (Hydra timestamped)
hydra:
  run:
    dir: /path/to/your/runs/my_run/${now:%Y-%m-%d_%H-%M-%S}-${exp_name}
```

Then run:

```bash
conda activate falcon
python main.py
```

Checkpoints are saved every epoch as `{epoch}_best_model.pt` in the Hydra run directory. Training progress is logged to [Weights & Biases](https://wandb.ai) (project `FALCON`; override with `WANDB_PROJECT`).

To disable W&B logging:
```bash
WANDB_MODE=disabled python main.py
```

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 200 | Total training epochs |
| `batch_size` | 8 | Per-GPU batch size |
| `lr` | 3e-4 | Learning rate (AdamW) |
| `devices` | [7] |
| `num_classes` | 39 | Phoneme set size (Lee-Hon 39) |
| `z_dim` | 256 | CNN encoder output dimension |
| `z_proj` | 64 | Projection head output dimension |

---

## Pretrained Checkpoints

Two checkpoints are used by FALCON (under `pretrained_models/`):

| File | Trained on | Best for |
|------|------------|----------|
| `falcon_timit_english.pt`      | TIMIT (read English)            | English phoneme alignment |
| `falcon_joint_multilingual.pt` | Joint TIMIT+Buckeye             | **Best for cross-lingual / multilingual zero-shot alignment** (Dutch, German, Hebrew, ...) at both phoneme and word level — the joint model generalizes better to unseen languages than either single-corpus model. |

The CLI tools (`predict.py`, `generate_textgrids.py`, `test_results.py`) and the
web demo (`app.py`) all auto-pick the right one from the `--lang` flag — pass
`--lang english` for the TIMIT checkpoint, `--lang multilingual` for the joint
checkpoint. Override with `--ckpt /path/to/your.pt` (or upload one in the demo).

For HuggingFace Spaces deployment, set `HF_MODEL_REPO` (and `HF_TOKEN` if private)
as Space Secrets — `app.py` will download both files from that repo on first use.

---

## Web Demo (Interactive Inference)

FALCON includes an interactive Gradio web interface for zero-shot forced alignment.
Upload audio of any sample rate along with a plain-text transcript (or standard
annotation file), and instantly visualize predicted boundaries, the Soft-DP path,
and download a TextGrid.

```bash
conda activate falcon
python app.py
```

This launches a local server (default `http://localhost:7860`). Inputs (audio, transcript, options) are on the left; the **Alignment Data** and **Visualizations** tabs are on the right.

![FALCON web demo — alignment data (boundary tables + TextGrid download)](assets/app_screen.jpeg)

![FALCON web demo — time-aligned visualizations](assets/app_screen2.jpeg)

**How to use it** — upload audio (any sample rate; resampled to 16 kHz internally) and a transcript, set the options, and click **Run Alignment**:

| Option | What it does |
|---|---|
| **Mode** | `phoneme` — align the phonemes; `word` — align words (boundaries derived from the predicted phonemes). |
| **Language** | `english` — transcript is already TIMIT-39 phonemes (no G2P); `multilingual` — any other language (runs the G2P path). Auto-selects the matching checkpoint. |
| **Pretrained checkpoint** | *Read English* (TIMIT), *Spontaneous English* (Buckeye), or *Multilingual* (joint). Follows **Language** by default; override it, or upload your own `.pt`. |
| **Word G2P** *(word mode)* | *Auto* (recommended — best backend for the language), *espeak*, *MFA-like*, or *none* (romanization — only for languages with no G2P model). If the chosen backend lacks the language it falls back automatically. |
| **Input language** *(word mode)* | Optional but recommended — lets *Auto* pick the right G2P and set the voice/dictionary behind the scenes. |
| **Annotation** | `.phn` (phoneme timestamps), `.wrd` (word timestamps), or `.txt` (plain token sequence, no timestamps). |
| **φ weight** | Acoustic ↔ linguistic balance in the Soft-DP (0.5 default). |

**Outputs:** the **Alignment Data** tab gives the boundary tables and a downloadable Praat `.TextGrid`; the **Visualizations** tab shows the time-aligned panels (waveform · spectrogram · phoneme posteriors · Soft-DP path · contrastive score).

---

## TextGrid Generation (CLI Inference)

The main entry point for generating alignments is `generate_textgrids.py`. This script processes your audio and annotations, and outputs standard Praat `.TextGrid` files ready for linguistic analysis (similar to tools like Montreal Forced Aligner).

`--ckpt` is optional — if omitted, the bundled TIMIT checkpoint is used for
`--lang english` and the joint TIMIT+Buckeye checkpoint for `--lang multilingual`.

### Option 1: Single File

```bash
conda activate falcon
python generate_textgrids.py \
  --wav  /path/to/audio.wav \
  --mode phoneme \
  --lang english \
  --annotation phn
```

### Option 2: Full Dataset Directory

```bash
conda activate falcon
python generate_textgrids.py \
  --wav_dir /path/to/dataset/test/ \
  --mode    word \
  --lang    english \
  --annotation wrd
```

**Flags:**

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--wav` | file path | — | Path to a single input `.wav` file (16 kHz mono) |
| `--wav_dir` | directory path | — | Path to a directory containing `.wav` files to process |
| `--ckpt` | file path | bundled | Path to a trained checkpoint (`.pt`). Defaults to `pretrained_models/falcon_timit_english.pt` (or `falcon_joint_multilingual.pt` when `--lang multilingual`). |
| `--mode` | `phoneme`, `word` | `phoneme` | Alignment granularity. `phoneme` = phoneme-level (default). `word` = word-level alignment (zero-shot). |
| `--lang` | `english`, `multilingual` | `english` | Language setting. `english` = trained English phoneme alignment. `multilingual` = any non-English language (zero-shot cross-lingual). |
| `--annotation` | any string | `phn` | Annotation file extension to look for. Use `txt` for plain text transcripts, or standard extensions (e.g. `phn`, `wrd`). |

The `.wav` file must have a paired annotation file in the same directory, providing the phoneme or word sequence to align to.

**Examples:**

```bash
# English phoneme-level (default — uses TIMIT checkpoint)
python generate_textgrids.py --wav_dir my_dataset/

# English word-level (zero-shot)
python generate_textgrids.py --wav_dir my_dataset/ --mode word --annotation wrd

# Multilingual phoneme-level (zero-shot — uses joint TIMIT+Buckeye checkpoint)
python generate_textgrids.py --wav_dir my_dataset/ --lang multilingual --annotation phn

# Plain-text transcript (no timestamps needed)
python generate_textgrids.py --wav_dir my_dataset/ --mode word --annotation txt
```

**Output:**
- A standard `.TextGrid` file saved next to each input `.wav` file.
- If `--mode word` is used, the `.TextGrid` will contain two tiers: `words` and `phones`.
- Visualizations saved next to the input WAV: `<basename>_probs.png`, `<basename>_logits.png`, `<basename>_boundaries.png`

---

## Evaluation — Batch

Evaluate a checkpoint over a directory of `.wav` files and report precision at multiple time thresholds:

```bash
conda activate falcon
python test_results.py \
  --wav  /path/to/test/wavs/ \
  --mode phoneme \
  --lang english \
  --annotation phn
```

**Flags:**

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--wav` | directory path | — | Directory containing `.wav` files to evaluate |
| `--ckpt` | path | bundled | A single `.pt` file or a directory (sweeps `{idx}_best_model.pt`). Defaults to the bundled TIMIT/Buckeye checkpoint matching `--lang`. |
| `--mode` | `phoneme`, `word` | `phoneme` | Alignment granularity — same meaning as in `predict.py` |
| `--lang` | `english`, `multilingual` | `english` | Language setting — same meaning as in `predict.py` |
| `--annotation` | any string | `phn` | Annotation file extension — same meaning as in `predict.py` |
| `--w-phi` | float | `0.5` | Acoustic ↔ linguistic feature weight |
| `--out-plot` | file path | — | If set (and sweeping a directory), save a precision-vs-checkpoint plot |
| `--no-plots` | flag | off | Skip per-file diagnostic plots — much faster on large datasets |

Reports precision at 10, 15, 20, 25, 50, and 100 ms tolerances.

---

## Multilingual / Cross-Lingual Alignment

FALCON can align speech in **unseen languages** (Dutch, German, Hebrew, and others) without any additional training. Phonemes from the target language are automatically mapped to the Lee-Hon 39 set used during English training via articulatory feature distance ([PanPhon](https://github.com/dmort27/panphon)).

Just pass `--lang multilingual` — this both selects the joint TIMIT+Buckeye
checkpoint (which generalizes better cross-lingually) and routes the input
through the G2P/articulatory-mapping pipeline:

```bash
python generate_textgrids.py \
  --wav_dir /path/to/dutch_audio_dir/ \
  --lang multilingual \
  --annotation phn
```

`.phn` / `.wrd` files can use any standard phoneme/word notation — `dutch_preprocess.aligner_pipeline()` handles IFA → IPA → LH39 conversion.

---

## Latent Space Visualization

Visualize how the CNN encoder learns phoneme-boundary-sensitive representations:

```bash
conda activate falcon
python visualize_latent_representation.py \
  --wav     /path/to/audio.wav \
  --run-dir /path/to/training/run/directory/
```

Saves four heatmap plots to `<run_dir>/latent_representations/`:
- `latent_epoch0_untrained.png` — CNN features before training
- `latent_best_trained.png` — CNN features at the best epoch
- `latent_comparison_epoch0_vs_best.png` — side-by-side comparison
- `latent_difference_best_minus_epoch0.png` — difference map

---

## Project Structure

```
FALCON/
├── app.py                           # Gradio web demo
├── generate_textgrids.py            # MFA-style CLI for TextGrid output
├── main.py                          # Training entry point
├── predict.py                       # Low-level single-file inference
├── test_results.py                  # Batch evaluation
├── solver.py                        # Train/val/test loop
├── next_frame_classifier.py         # Model + loss functions
├── dataloader.py                    # Dataset classes
├── utils.py                         # Soft-DP, metrics, phoneme maps
├── dutch_preprocess.py              # Cross-lingual phoneme mapping
├── visualize_latent_representation.py
├── pretrained_models/               # Bundled checkpoints (or HF Hub at runtime)
│   ├── falcon_timit_english.pt
│   └── falcon_joint_multilingual.pt
├── conf/
│   └── config.yaml                  # All hyperparameters (Hydra)
├── scripts/                         # Data preparation & utilities
├── assets/                          # README screenshots & example figures
├── requirements.txt
```

---



## License

This project is released under the [MIT License](LICENSE).
