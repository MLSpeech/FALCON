"""
falcon_viz.py — Time-aligned alignment visualizations for FALCON.

Two entry points, both sharing the same per-panel drawers so the styling stays
identical:

    make_alignment_panels(wav, ckpt, out_path, ...)
        -> ONE tall figure, 5 panels stacked on a shared time axis (for the
           README / a downloadable overview).

    make_alignment_separate(wav, ckpt, out_dir, ...)
        -> a LIST of (path, caption), one wide, full-size figure per panel (for
           the web app, where a single stacked figure renders too small to read).

Panels: waveform, log-mel spectrogram, phoneme posteriors, Soft-DP cost matrix +
alignment path, and the contrastive boundary score with its detected peaks. The
same predicted boundaries (crimson) and truth boundaries (charcoal) are drawn on
every panel so the time alignment is visually verifiable.

This module does NOT modify any model/training file. It re-runs the model forward
(replicating predict.py:main_predict) and reproduces utils.phoneme_alignment's DP
matrix locally (re-using utils.compute_phi_1 / compute_phi_2 unchanged).
"""

import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dataloader import spectral_size
from predict import _load_model
from utils import (
    max_min_norm,
    get_timit_61_phoneme_mappings,
    compute_phi_1,
    compute_phi_2,
    phoneme_to_idx_MACRO,
    timit_to_leehon_map_MACRO,
    timit_leehon_39_phonemes,
)

# Hard-coded in the model forward (next_frame_classifier.py) — CNN total stride.
# One latent frame ~= len_ratio audio samples ~= 10 ms at 16 kHz.
LEN_RATIO = 161.34011627906978
SR = 16000

# Light / pastel palette so the crimson boundary overlays stand out.
PRED_COLOR = "crimson"
TRUTH_COLOR = "#37474f"        # charcoal (distinct from the blue/purple cmaps)
WAVE_COLOR = "#3b6fb6"         # blue (matches the example waveform figure)
SCORE_COLOR = "#3a7ca5"
SPEC_CMAP = "viridis"
POST_CMAP = "viridis"
DP_CMAP = "viridis"


# --------------------------------------------------------------------------- #
#  DP matrix reconstruction (mirrors utils.phoneme_alignment, returns matrix)  #
# --------------------------------------------------------------------------- #
def _build_dp_matrix(p_seq, w_phi, original_lengths, derivative_preds_np, probs_real):
    """Re-run the exact Soft-DP forward + soft-argmax backtrack from
    utils.phoneme_alignment, additionally returning the DP matrix so it can be
    plotted (the stock function only returns the path)."""
    gamma = 1e-20
    T = int(original_lengths[0])
    n = len(p_seq)
    device = derivative_preds_np.device

    if isinstance(probs_real, np.ndarray):
        probs_real = torch.tensor(probs_real, device=device)
    cumsum_probs = torch.cumsum(probs_real, dim=0)

    phoneme_mappings = {
        p.lower(): timit_to_leehon_map_MACRO.get(p.lower(), "sil")
        if p.lower() not in timit_leehon_39_phonemes
        else p.lower()
        for p in p_seq
    }
    derivatives = torch.cat(
        [torch.tensor([0], device=device), torch.diff(derivative_preds_np, dim=0)]
    )

    dp_mat = torch.full((n, T, T), float(-1e9), device=device)

    t_e = torch.arange(T, device=device)
    dp_mat[0, 0, :] = (
        w_phi[0] * compute_phi_1(derivatives, 0, t_e)
        + w_phi[1] * compute_phi_1(derivatives, 0, t_e)
    )

    for i in range(1, n):
        p_idx = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[i].lower()]]
        t_start = torch.arange(T, device=device)
        t_end = torch.arange(T, device=device)
        t_start_grid, t_end_grid = torch.meshgrid(t_start, t_end, indexing="ij")
        valid_mask = t_start_grid < t_end_grid

        phi1_dev = compute_phi_1(derivatives, t_start_grid, t_end_grid)
        phi2 = compute_phi_2(cumsum_probs, p_idx, t_start_grid, t_end_grid)
        total_phi = w_phi[0] * phi1_dev + w_phi[1] * phi2

        col_lse = torch.logsumexp(dp_mat[i - 1] / gamma, dim=0) * gamma
        prev_scores = torch.where(
            valid_mask,
            col_lse.unsqueeze(1).expand(T, T),
            torch.full((T, T), float(-1e9), device=device),
        )
        dp_mat[i] = torch.where(
            valid_mask, total_phi + prev_scores, torch.full_like(total_phi, float(-1e9))
        )

    best_start_times = torch.zeros((n), dtype=derivative_preds_np.dtype, device=device)
    best_prev_t_end = T - 1
    for i in range(n):
        cur_ph = n - 1 - i
        scores = dp_mat[cur_ph, :, best_prev_t_end]
        soft_weights = torch.softmax(scores / gamma, dim=0)
        expected_idx = (
            soft_weights
            * torch.arange(T, device=device, dtype=derivative_preds_np.dtype)
        ).sum()
        best_start_times[cur_ph] = expected_idx
        best_prev_t_end = int(expected_idx.round().item())

    dp_to_plot = dp_mat.detach().cpu().max(dim=1)[0].numpy()  # (n, T)
    best_start_frames = best_start_times.detach().cpu().numpy()
    return dp_to_plot, best_start_frames


# --------------------------------------------------------------------------- #
#  Forward pass + array extraction (replicates predict.py:main_predict)         #
# --------------------------------------------------------------------------- #
def _extract_arrays(wav, ckpt, annotation):
    model, _peak_params = _load_model(ckpt)
    model.eval()

    audio, sr = torchaudio.load(wav)
    assert sr == SR, "model expects 16 kHz audio"
    audio = audio[0]
    audio_len = len(audio)
    spectral_len = spectral_size(audio_len)
    len_ratio = audio_len / spectral_len  # ~= LEN_RATIO

    base_dir = os.path.dirname(wav)
    base_name = os.path.basename(wav).split(".")[0]
    matches = glob(os.path.join(base_dir, f"{base_name}*.{annotation}"))
    phn_path = matches[0] if matches else wav.replace("wav", "phn")

    with open(phn_path, "r") as f:
        lines = [ln.split() for ln in f.readlines()]
    truth_secs = [float(ln[1]) / SR for ln in lines][:-1]
    phonemes = [ln[2].strip() for ln in lines]
    truth_labels = list(phonemes)

    length = [audio_len / len_ratio]
    with torch.no_grad():
        preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi = model(
            audio.unsqueeze(0), None, [phonemes], length
        )

    # Contrastive / latent boundary score (predict.py 168-172).
    p = preds[1][0]
    p = max_min_norm(p)
    p_np = p.detach().numpy()[0]
    p_np = p_np - np.median(p_np)

    # Phoneme posteriors (predict.py 143).
    probs_real = F.softmax(probs, dim=-1).squeeze(0).detach().numpy()  # (T, 39)
    _, idx_to_phoneme = get_timit_61_phoneme_mappings()
    phoneme_labels = [idx_to_phoneme[i] for i in range(39)]

    pred_secs = list(total_peaks[0])

    # Phoneme labels at predicted-segment midpoints (segment i = [bound[i],
    # bound[i+1]] with bounds = predicted boundaries + the utterance end), matching
    # how the TextGrid assigns each phoneme to a predicted interval.
    bounds = [float(x) for x in pred_secs] + [audio_len / SR]
    seg_mids = [(bounds[i] + bounds[i + 1]) / 2.0 for i in range(len(bounds) - 1)]
    n_lab = min(len(seg_mids), len(phonemes))
    label_mids = seg_mids[:n_lab]
    label_text = phonemes[:n_lab]

    # DP matrix (reproduce phoneme_alignment locally).
    w_phi_vec = w_phi.detach()
    deriv_arg = torch.tensor(p_np, dtype=torch.float32)
    dp_to_plot, dp_path_frames = _build_dp_matrix(
        phonemes, w_phi_vec, [int(original_lengths[0])], deriv_arg, probs_real
    )

    return dict(
        audio=audio.numpy(),
        sr=sr,
        len_ratio=len_ratio,
        duration=audio_len / SR,
        latent_score=p_np,
        probs_real=probs_real,
        phoneme_labels=phoneme_labels,
        pred_secs=pred_secs,
        truth_secs=truth_secs,
        truth_labels=truth_labels,
        dp_to_plot=dp_to_plot,
        dp_path_frames=dp_path_frames,
        phonemes=phonemes,
        label_mids=label_mids,
        label_text=label_text,
    )


# --------------------------------------------------------------------------- #
#  Shared per-panel drawers                                                     #
# --------------------------------------------------------------------------- #
def _overlay_boundaries(ax, pred_secs, truth_secs, show_truth, label_first=False):
    """Predicted (crimson dashed) + optional truth (charcoal dotted) lines."""
    for j, t in enumerate(truth_secs if show_truth else []):
        ax.axvline(t, color=TRUTH_COLOR, linestyle=":", linewidth=0.9, alpha=0.6,
                   label="Truth boundary" if (label_first and j == 0) else None, zorder=2)
    for j, t in enumerate(pred_secs):
        ax.axvline(t, color=PRED_COLOR, linestyle="--", linewidth=1.1, alpha=0.9,
                   label="Predicted boundary" if (label_first and j == 0) else None, zorder=3)


def _annotate_phoneme_tier(ax, d):
    """Write the input phoneme labels at their predicted-segment midpoints, just
    below the x-axis (a phoneme tier)."""
    mids = d.get("label_mids") or []
    labels = d.get("label_text") or []
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for m, lab in zip(mids, labels):
        ax.text(m, -0.07, str(lab), transform=trans, rotation=90, ha="center",
                va="top", fontsize=5.5, color="#333333", clip_on=False)


def _panel_waveform(ax, d, show_truth, label_first=True):
    audio = d["audio"]
    dur = d["duration"]
    t = np.linspace(0, dur, num=len(audio))
    ax.plot(t, audio, color=WAVE_COLOR, linewidth=0.5)
    ax.set_ylabel("Amplitude")
    ax.margins(x=0)
    ymax = (float(np.abs(audio).max()) or 1.0) * 1.15
    ax.set_ylim(-ymax, ymax)
    _overlay_boundaries(ax, d["pred_secs"], d["truth_secs"], show_truth, label_first=label_first)
    if label_first:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)


def _panel_spectrogram(ax, d, show_truth):
    audio = d["audio"]
    dur = d["duration"]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=400, hop_length=160, n_mels=80
    )(torch.tensor(audio).float().unsqueeze(0))
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel).squeeze(0).numpy()
    ax.imshow(mel_db, aspect="auto", origin="lower",
              extent=[0, dur, 0, SR / 2 / 1000.0], cmap=SPEC_CMAP)
    ax.set_ylabel("Freq (kHz)")
    _overlay_boundaries(ax, d["pred_secs"], d["truth_secs"], show_truth)


def _panel_posteriors(ax, d, show_truth, colorbar=True):
    probs_real = d["probs_real"]
    dur = d["duration"]
    im = ax.imshow(probs_real.T, aspect="auto", origin="lower",
                   extent=[0, dur, -0.5, 38.5], cmap=POST_CMAP, interpolation="nearest")
    ax.set_yticks(range(39))
    ax.set_yticklabels(d["phoneme_labels"], fontsize=5.5)
    ax.set_ylabel("Phoneme (LH-39)")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="P(phoneme)", pad=0.01, fraction=0.025)
    _overlay_boundaries(ax, d["pred_secs"], d["truth_secs"], show_truth)


def _panel_dp(ax, d, show_truth, colorbar=True):
    dp_to_plot = d["dp_to_plot"]
    dur = d["duration"]
    len_ratio = d["len_ratio"]
    masked = np.ma.masked_where(dp_to_plot <= -1e8, dp_to_plot)
    cmap = getattr(plt.cm, DP_CMAP).copy()
    cmap.set_bad(color="white")
    n_ph = dp_to_plot.shape[0]
    im = ax.imshow(masked, aspect="auto", origin="lower",
                   extent=[0, dur, -0.5, n_ph - 0.5], cmap=cmap, interpolation="nearest")
    path_secs = np.asarray(d["dp_path_frames"]) * len_ratio / SR
    ax.plot(path_secs, np.arange(n_ph), color=PRED_COLOR, marker=".", markersize=4,
            linewidth=1.2, label="Optimal alignment path")
    ax.set_ylabel("Phoneme position")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="DP score", pad=0.01, fraction=0.025)
    _overlay_boundaries(ax, d["pred_secs"], d["truth_secs"], show_truth)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def _panel_contrastive(ax, d, show_truth):
    # Boundary score (red) + its derivative (magenta), as in predict.py's run plot,
    # but (1) auto-scaled robustly so the per-boundary structure is visible instead
    # of being squashed by the silence->speech onset spike, and (2) without the
    # redundant predicted-boundary markers — the predicted boundaries are the red
    # dashed lines shared across every panel. Ground truth omitted (not available
    # at inference). x-axis is time (s).
    s = np.asarray(d["latent_score"], dtype=float)
    len_ratio = d["len_ratio"]
    n = len(s)
    t = np.arange(n) * len_ratio / SR
    deriv = np.concatenate([[0.0], np.diff(s)])

    ax.plot(t, deriv, marker="o", markersize=1.6, linewidth=0.7, alpha=0.8,
            color="magenta", label="Derivative of latent score")
    ax.plot(t, s, marker="*", markersize=2.2, linewidth=0.8, color="red",
            label="Latent score")

    # Robust symmetric y-limit: zoom to the 96th-percentile magnitude so the small
    # per-boundary structure fills the panel; the rare large onset spike clips off.
    mag = np.concatenate([np.abs(s), np.abs(deriv)])
    A = max(float(np.percentile(mag, 96)) * 1.5, 0.03) if mag.size else 0.1
    ax.set_ylim(-A, A)
    ax.set_ylabel("Score")
    ax.margins(x=0)
    _overlay_boundaries(ax, d["pred_secs"], [], show_truth=False)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)


# Order shared by both makers: (key, caption, drawer, separate-figure height).
_PANELS = [
    ("waveform",    "1. Waveform",                                  _panel_waveform,     2.6),
    ("spectrogram", "2. Log-mel spectrogram",                       _panel_spectrogram,  2.8),
    ("posteriors",  "3. Phoneme posteriors",                        _panel_posteriors,   3.6),
    ("dp",          "4. Soft-DP cost matrix + alignment path",      _panel_dp,           3.4),
    ("contrastive", "5. Contrastive boundary score", _panel_contrastive, 2.6),
]


# --------------------------------------------------------------------------- #
#  Combined figure (README)                                                    #
# --------------------------------------------------------------------------- #
def make_alignment_panels(wav, ckpt, out_path, w_phi=0.5, language="english",
                          annotation="phn", show_truth=True):
    """Build the stacked, time-aligned multi-panel figure and save it to out_path."""
    if language != "english":
        print(f"[falcon_viz] language='{language}' not supported; using english path.")
    d = _extract_arrays(wav, ckpt, annotation)
    dur = d["duration"]

    fig = plt.figure(figsize=(12, 17), dpi=150, constrained_layout=True)
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1.0, 1.3, 1.7, 1.6, 1.0])
    ax_wave = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1], sharex=ax_wave)
    ax_post = fig.add_subplot(gs[2], sharex=ax_wave)
    ax_dp = fig.add_subplot(gs[3], sharex=ax_wave)
    ax_score = fig.add_subplot(gs[4], sharex=ax_wave)
    axes = [ax_wave, ax_spec, ax_post, ax_dp, ax_score]

    _panel_waveform(ax_wave, d, show_truth, label_first=True)
    _panel_spectrogram(ax_spec, d, show_truth)
    _panel_posteriors(ax_post, d, show_truth, colorbar=True)
    _panel_dp(ax_dp, d, show_truth, colorbar=True)
    _panel_contrastive(ax_score, d, show_truth)

    for ax, (_key, caption, _drawer, _h) in zip(axes, _PANELS):
        ax.set_title(caption, loc="left", fontweight="bold", fontsize=11)
        ax.tick_params(labelbottom=True)
        _annotate_phoneme_tier(ax, d)
    ax_score.set_xlabel("Time (s)", fontsize=12)
    ax_wave.set_xlim(0, dur)

    fig.suptitle("FALCON forced-alignment — time-aligned representations",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[falcon_viz] saved {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
#  Separate per-panel figures (web app)                                        #
# --------------------------------------------------------------------------- #
def make_alignment_separate(wav, ckpt, out_dir, w_phi=0.5, language="english",
                            annotation="phn", show_truth=True):
    """Build one wide, full-size figure per panel; return [(path, caption), ...]."""
    if language != "english":
        print(f"[falcon_viz] language='{language}' not supported; using english path.")
    d = _extract_arrays(wav, ckpt, annotation)
    dur = d["duration"]
    os.makedirs(out_dir, exist_ok=True)

    out = []
    for key, caption, drawer, h in _PANELS:
        fig, ax = plt.subplots(figsize=(12, h), dpi=130)
        if key == "waveform":
            drawer(ax, d, show_truth, label_first=True)
        else:
            drawer(ax, d, show_truth)
        ax.set_title(caption, loc="left", fontweight="bold", fontsize=12)
        ax.set_xlim(0, dur)
        _annotate_phoneme_tier(ax, d)
        ax.set_xlabel("Time (s)", labelpad=26)
        p = os.path.join(out_dir, f"panel_{key}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out.append((p, caption))
    return out


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    make_alignment_panels(
        wav=os.path.join(_here, "assets", "fasw0sa2.wav"),
        ckpt=os.path.join(_here, "pretrained_models", "falcon_timit_english.pt"),
        out_path=os.path.join(_here, "assets", "example_panels.png"),
    )
