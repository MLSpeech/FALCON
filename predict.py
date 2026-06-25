import argparse
from glob import glob
from unittest import case
import dill
from argparse import Namespace
import torch
import torchaudio
import torch.nn.functional as F
from utils import (max_min_norm,
                   get_timit_61_phoneme_mappings)
from next_frame_classifier import NextFrameClassifier

from dataloader import spectral_size
import matplotlib.pyplot as plt
import numpy as np
import os

import dutch_preprocess
from utils import timit_to_leehon_map_MACRO, timit_leehon_39_phonemes, timit_61_phonemes

# Cache the loaded model+peak-params by checkpoint path so batch runs (many files,
# one checkpoint) don't reload ~330MB from disk per file. Output is unchanged.
_MODEL_CACHE = {}

def _load_model(ckpt_path):
    cached = _MODEL_CACHE.get(ckpt_path)
    if cached is not None:
        return cached
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    hp = ckpt["hparams"]
    model = NextFrameClassifier(hp)
    try:
        weights = ckpt["state_dict"]
    except Exception:
        weights = ckpt["model_state_dict"]
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']
    _MODEL_CACHE.clear()  # keep only the most-recent checkpoint (bounded memory)
    _MODEL_CACHE[ckpt_path] = (model, peak_detection_params)
    return model, peak_detection_params

def main_predict(wav, ckpt, w_phi, language="english", annotation="phn", no_plots=False):
    print(f"running inference on: {wav}")
    print(f"running inferece using ckpt: {ckpt}")
    print("\n\n", 90 * "-")

    # Optional plot suppression for fast batch runs. Patches plt.savefig for
    # the duration of this call so plots inside utils.phoneme_alignment are
    # skipped too, without having to modify utils.py.
    _orig_savefig = plt.savefig if no_plots else None
    if no_plots:
        plt.savefig = lambda *a, **k: None

    model, peak_detection_params = _load_model(ckpt)
    # peak_detection_params["prominence"] = prominence # Unused
    # load data
    audio, sr = torchaudio.load(wav)
    assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
    audio = audio[0]
    # audio = audio.unsqueeze(0)
    
    base_dir = os.path.dirname(wav)
    base_name = os.path.basename(wav).split('.')[0]
    # search_pattern = os.path.join(base_dir, f"{base_name}*.phn")
    # search_pattern = os.path.join(base_dir, f"{base_name}*.wrd")
    # search_pattern = os.path.join(base_dir, f"{base_name}*.word")
    search_pattern = os.path.join(base_dir, f"{base_name}*.{annotation}")
    matching_files = glob(search_pattern)
    if matching_files:
        phn_path = matching_files[0]
    else:
        print("No matching .phn file found. Using default naming convention.")
        phn_path = wav.replace("wav", "phn")

    # load audio
    audio_len = len(audio)
    spectral_len = spectral_size(audio_len)
    len_ratio = (audio_len / spectral_len)

    # load labels -- segmentation and phonemes
    with open(phn_path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.split(), lines))

        # get segment times
        times = torch.FloatTensor(list(map(lambda line: int(float(line[1]) / len_ratio), lines)))[:-1]  # don't count end time as boundary
        # times = torch.FloatTensor(list(map(lambda line: int(int(line[1]) / len_ratio), lines)))[:-1]  # don't count end time as boundary
        times_sec = torch.FloatTensor(list(map(lambda line: (float(line[1]) / sr), lines)))[:-1]  # don't count end #sr = 16000 in TIMIT
        # times_sec = torch.FloatTensor(list(map(lambda line: (int(line[1]) / sr), lines)))[:-1]  # don't count end #sr = 16000 in TIMIT
        
        # get phonemes in each segment (for K times there should be K+1 phonemes)
        phonemes = list(map(lambda line: line[2].strip(), lines))

    # Original input labels (one per .phn/.wrd/.txt line) — kept around for the
    # truth-boundary text on plots. After G2P the `phonemes` variable is
    # flattened into LH39 phonemes whose count differs from len(times), so we
    # cache the per-line labels here before that transformation.
    truth_labels = list(phonemes)
    
    if language == "dutch":
        lh39_ph = []
        for IFA_ph in phonemes:
            print(f"\nINPUT: {IFA_ph}")
            # output = dutch_preprocess.aligner_pipeline(timit_to_leehon_map_MACRO[IFA_ph.lower()])
            output = dutch_preprocess.aligner_pipeline(IFA_ph if IFA_ph.lower() not in timit_61_phonemes else timit_to_leehon_map_MACRO[IFA_ph.lower()])
            # output = dutch_preprocess.aligner_pipeline(IFA_ph)
            
            # # FOR WORDS: #not good
            # lh39_ph.append(output[0]["lh39"])
            
            # FOR PHONEMES:
            lh39_ph.append([x["lh39"] for x in output])
        if not output:
            print("Results: None")
        print(phonemes)
        print(f"Dutch IPA to LH39 mapping: {lh39_ph}")
        
        # try
        phonemes = np.hstack(lh39_ph).tolist()
        
    audio, seg, phonemes, length = audio.unsqueeze(0), [times.tolist()], [phonemes], [audio_len/len_ratio] #[spectral_size(len(audio))]

    
    
    with torch.no_grad():
        model.eval()
        
        # preds,original_lengths, probs, frame_labels = model(audio,None,phonemes,length)
        # ------- Sept 10 - check with truth preds no truth for nce ---------------
        preds,original_lengths, probs, frame_labels, _,preds_peaks, w_phi = model(audio,None,phonemes,length)
        # preds,original_lengths, probs, frame_labels, _,preds_peaks, w_phi = model(audio,seg,phonemes,length)
        # ------- ------------------------------------------------- ---------------
    
    phoneme_to_idx, idx_to_phoneme = get_timit_61_phoneme_mappings()
    
    
    phoneme_labels = [idx_to_phoneme[i] for i in range(39)]
    # phoneme_labels = [idx_to_phoneme[i] for i in range(61)]
    # phoneme_labels = [idx_to_phoneme[i] for i in range(41)]
    # probs_real = probs #F.softmax(probs, dim=-1) 
    probs_real = F.softmax(probs, dim=-1).squeeze(0).detach().numpy()
    
    out_dir = os.path.dirname(wav)
    base_name = os.path.basename(wav).replace('.wav', '')

    plt.figure(figsize=(15, 5))
    plt.imshow(probs_real.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.xlabel('Frame Index')
    plt.ylabel('Phoneme')
    plt.yticks(ticks=range(39), labels=phoneme_labels)
    plt.title('Frame-wise Label Probability Map')
    # Truth-boundary axvlines + per-segment label text. truth_labels is the
    # original (pre-G2P) per-line input labels, so its length matches `times`.
    for i, s in enumerate(times):
        s_val = float(s)
        plt.axvline(x=s_val, color='red', linestyle='--', linewidth=1,
                    label='Truth boundary' if i == 0 else "")
        if i < len(truth_labels):
            plt.text(s_val, probs_real.shape[1] + 1, truth_labels[i],
                     color='red', rotation=90, va='top', ha='center', fontsize=8)
    plt.savefig(os.path.join(out_dir, f"{base_name}_probs.png"))
    plt.close()
    
    # Latent boundary scores (CNN output) — used for the boundaries plot.
    preds = preds[1][0]
    preds = max_min_norm(preds)
    preds_np = preds.detach().numpy()[0]
    median_h = np.median(preds_np)
    preds_np = preds_np - median_h

    # Predicted boundary timestamps (in seconds).
    preds = torch.tensor(preds_peaks[0], dtype=torch.float32)
    print(f"predicted boundaries (s): {preds}")

    # Marker height: scale to the *typical* peak of the latent score so the
    # triangles sit roughly on top of peaks instead of dwarfing them or
    # vanishing into the noise floor. The signal is sparse, so we filter out
    # near-zero values and take the median of what's left — robust to both
    # outlier spikes and the long tail of zero-ish frames.
    abs_p = np.abs(preds_np) if preds_np.size else np.array([])
    if abs_p.size:
        noise_thresh = 0.05 * float(np.max(abs_p))
        peaks = abs_p[abs_p > noise_thresh]
        marker_h = float(np.median(peaks)) if peaks.size else float(np.max(abs_p))
    else:
        marker_h = 0.1
    if marker_h <= 0:
        marker_h = 0.1

    # Truth-boundary marker signal (peak per truth boundary, zero elsewhere).
    signal = np.zeros(int(original_lengths[0]))
    signal_max_idx = signal.shape[0] - 1
    for t in times:
        idx = min(int(t), signal_max_idx)
        signal[idx] = marker_h
    times_clipped = [t for t in times if float(t) <= signal_max_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(signal, marker='*', linestyle='-', label='Truth boundary marker')
    derivative_preds_np = np.diff(preds_np)
    derivative_preds_np = np.concatenate([[0], derivative_preds_np])
    plt.plot(range(len(derivative_preds_np)), derivative_preds_np,
             marker='o', label='Derivative of latent score', color='magenta')
    plt.plot(range(len(preds_np)), preds_np,
             marker='*', label='Latent score', color='red')

    preds_plot = np.zeros(int(original_lengths[0]))
    for pred in preds:
        idx = int(pred * sr / len_ratio)
        if idx <= len(preds_plot) - 1:
            preds_plot[idx] = marker_h
    plt.plot(range(len(preds_plot)), preds_plot,
             marker='^', label='Predicted boundary', linestyle='None')

    y_top = plt.ylim()[1]
    for i, s in enumerate(times_clipped):
        s_val = float(s)
        plt.axvline(x=s_val, color='red', linestyle='--', linewidth=1,
                    label='Truth boundary' if i == 0 else "")
        if i < len(truth_labels):
            plt.text(s_val, y_top, truth_labels[i], color='red',
                     rotation=90, va='top', ha='center', fontsize=8)

    plt.xlabel('Frame Index')
    plt.ylabel('Score')
    plt.title('Predicted Boundaries')
    plt.legend(loc='upper right')
    # Clip the y-axis to a few × typical-peak height so single outlier spikes
    # don't visually squash the rest of the signal.
    y_half = max(marker_h * 4.0, 0.1)
    plt.ylim(-y_half, y_half)
    plt.savefig(os.path.join(out_dir, f"{base_name}_boundaries.png"))
    plt.close()

    pred_bound, truth_bound = preds, times_sec
    mapped_ph = lh39_ph if language == "dutch" else None

    if no_plots:
        plt.savefig = _orig_savefig

    return pred_bound, truth_bound, mapped_ph

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT_ENGLISH      = os.path.join(_SCRIPT_DIR, "pretrained_models", "falcon_timit_english.pt")
# The joint TIMIT+Buckeye model generalizes best to unseen languages — it is the
# strongest checkpoint on every multilingual test set (Dutch/German/Hebrew,
# phoneme and word) — so it is the default for the multilingual / cross-lingual path.
DEFAULT_CKPT_MULTILINGUAL = os.path.join(_SCRIPT_DIR, "pretrained_models", "falcon_joint_multilingual.pt")

def resolve_internal_language(lang: str, mode: str, annotation: str) -> str:
    """
    Map user-facing (--lang, --mode, --annotation) to the internal `language`
    flag main_predict() understands.

      'english' = no G2P; assumes labels are already TIMIT-39 phonemes.
      'dutch'   = G2P pipeline (panphon-based mapping). Used for any non-English
                  language, word-level mode, or plain-text input.
    """
    if lang == "english" and mode == "phoneme" and annotation.lower() == "phn":
        return "english"
    return "dutch"

def resolve_default_ckpt(lang: str) -> str:
    return DEFAULT_CKPT_MULTILINGUAL if lang == "multilingual" else DEFAULT_CKPT_ENGLISH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised segmentation inference script')
    parser.add_argument('--wav', help='path to wav file')
    parser.add_argument('--ckpt', default=None,
                        help='Path to checkpoint file. If omitted, uses '
                             'pretrained_models/falcon_timit_english.pt for --lang english '
                             'or falcon_joint_multilingual.pt for --lang multilingual '
                             '(the joint TIMIT+Buckeye model is best for cross-lingual zero-shot).')

    parser.add_argument('--mode', type=str, default='phoneme', choices=['phoneme', 'word'],
                        help='Alignment granularity: "phoneme" = phoneme-level alignment (default). '
                             '"word" = word-level alignment (zero-shot, no additional training).')
    parser.add_argument('--lang', type=str, default='english', choices=['english', 'multilingual'],
                        help='Language setting: "english" = trained English phoneme alignment (default). '
                             '"multilingual" = any non-English language (zero-shot cross-lingual).')
    parser.add_argument('--annotation', type=str, default='phn',
                        help='Annotation file extension to search for (e.g. phn, wrd, word, txt). Default: phn')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip per-file diagnostic plots (probs/logits/boundaries/dp_matrix) for faster runs.')
    args = parser.parse_args()

    ckpt = args.ckpt or resolve_default_ckpt(args.lang)
    language = resolve_internal_language(args.lang, args.mode, args.annotation)
    main_predict(args.wav, ckpt, w_phi=0.5, language=language,
                 annotation=args.annotation, no_plots=args.no_plots)
