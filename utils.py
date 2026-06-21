import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import wandb
from tqdm import tqdm
import concurrent.futures
from typing import List, Sequence, Union
import time
from memory_profiler import profile

# Optionally redirect the dp_matrix plot to a specific directory (used by demo).
# Set via set_dp_matrix_out_dir() before calling inference; reset to None after.
_dp_matrix_out_dir = None

def set_dp_matrix_out_dir(path):
    global _dp_matrix_out_dir
    _dp_matrix_out_dir = path

timit_leehon_39_phonemes = [
    'ao', 'ae', 'ah','aw', 'er', 'ay', 
    'b', 'sil', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'm', 'en', 'ng', 'ey',
    'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'v', 'w', 'y', 'z', 'sh', 't', 'r', 's', 'th','uh', 'uw', 'oy', 'ow','p'
]

timit_61_phonemes = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
        'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
        'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'm', 'n', 'ng', 'l',
        'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw','ux',
        'v', 'w', 'y', 'z', 'zh'
    ]

# Create mappings
# phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(timit_61_phonemes)}
phoneme_to_idx_MACRO = {phoneme: idx for idx, phoneme in enumerate(timit_leehon_39_phonemes)}

idx_to_phoneme_MACRO = {idx: phoneme for phoneme, idx in phoneme_to_idx_MACRO.items()}
timit_to_leehon_map_MACRO = {
        'aa': 'ao', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'ay': 'ay',
        'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 'dx', 'eh': 'eh', 'el': 'el',
        'em': 'm', 'en': 'en', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g', 'gcl': 'sil',
        'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k', 'kcl': 'sil',
        'l': 'el', 'm': 'm', 'n': 'en', 'ng': 'ng', 'nx': 'en', 'ow': 'ow', 'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil',
        'q': 't', 'qcl': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw',
        'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'sh',
}

def create_truth_probs_real(segments, phonemes, phoneme_to_index, num_frames):
    segments = [0] + list(segments)
    num_phonemes = len(phoneme_to_index)
    probs_real = torch.zeros((num_frames, num_phonemes), dtype=torch.float32)
    for seg_idx in range(len(phonemes)):
        start = int(segments[seg_idx])
        end = int(segments[seg_idx + 1]) if seg_idx + 1 < len(segments) else num_frames
        if end > start:
            ph_label = phonemes[seg_idx].lower()
            ph_index = phoneme_to_index.get(ph_label, phoneme_to_index.get('sil', 0))
            probs_real[start:end, ph_index] = 1.0
    return probs_real


# ------------------------------- ablations -------------------------------

# phoneme alignment with classic (hard) DP
def phoneme_alignment_Hard_DP(p_seq, w_phi, original_lengths, len_ratio, derivative_preds_np, probs_real):
    # Gamma is kept for signature consistency but not used in hard DP
    gamma = 1e-20 
    T = int(original_lengths[0])
    n = len(p_seq)
    device = derivative_preds_np.device

    if isinstance(probs_real, np.ndarray):
        probs_real = torch.tensor(probs_real, device=device)
    cumsum_probs = torch.cumsum(probs_real, dim=0)
    
    phoneme_mappings = {p.lower(): timit_to_leehon_map_MACRO.get(p.lower(), 'sil') if p.lower() not in timit_leehon_39_phonemes else p.lower() for p in p_seq}
    derivatives = torch.cat([torch.tensor([0], device=device), torch.diff(derivative_preds_np, dim=0)])

    # Initialize DP matrix with very low value
    dp_mat = torch.full((n, T, T), float(-1e9), device=device)
    p_idx0 = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[0].lower()]]

    # Initial state for first phoneme
    t_e = torch.arange(T, device=device)
    dp_mat[0, 0, :] = (
        w_phi[0] * compute_phi_1(derivatives, 0, t_e)
    )

    # Forward Pass
    for i in tqdm(range(1, n)):
        p_idx = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[i].lower()]]
        t_start = torch.arange(T, device=device)
        t_end = torch.arange(T, device=device)
        t_start_grid, t_end_grid = torch.meshgrid(t_start, t_end, indexing='ij')
        valid_mask = t_start_grid < t_end_grid
        
        phi1_dev = compute_phi_1(derivatives, t_start_grid, t_end_grid)
        phi2 = compute_phi_2(cumsum_probs, p_idx, t_start_grid, t_end_grid)
        total_phi = w_phi[0] * phi1_dev
        
        prev_scores = torch.full((T, T), float(-1e9), device=device)
        
        for t_end_val in range(T):
            valid_starts = t_start[t_start < t_end_val]
            if valid_starts.numel() == 0:
                continue
            
            # --- CLASSIC DP CHANGE ---
            # Instead of LogSumExp (Soft-Max), use Hard Max
            prev = dp_mat[i-1, :valid_starts[-1]+1, valid_starts]
            max_prev, _ = torch.max(prev, dim=0) 
            prev_scores[valid_starts, t_end_val] = max_prev

        dp_mat[i] = torch.where(valid_mask, total_phi + prev_scores, torch.full_like(total_phi, float(-1e9)))

    # Backtracking (Classic Argmax)
    best_start_times = torch.zeros((n), dtype=derivative_preds_np.dtype, device=device)
    best_prev_t_end = T - 1
    
    for i in range(n):
        cur_ph = n - 1 - i
        # Find the exact index that gave the maximum score
        scores = dp_mat[cur_ph, :, best_prev_t_end]
        
        # --- CLASSIC DP CHANGE ---
        # Instead of expected_idx (Soft-Argmax), use Hard Argmax
        best_t_start = torch.argmax(scores)
        
        best_start_times[cur_ph] = best_t_start.to(derivative_preds_np.dtype)
        best_prev_t_end = int(best_t_start.item())
        
    # Visualization Code (unchanged logic, updated labels)
    dp_mat_cpu = dp_mat.detach().cpu()
    best_start_times_cpu = best_start_times.detach().cpu().numpy()
    dp_to_plot = dp_mat_cpu.max(dim=1)[0].numpy()
    masked_dp = np.ma.masked_where(dp_to_plot <= -1e8, dp_to_plot)

    plt.figure(figsize=(12, 6))
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    plt.imshow(masked_dp, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label='Hard DP Score')
    plt.xlabel('End time (frame)')
    plt.ylabel('Phoneme index')
    plt.title('Classic (Hard) DP Matrix with Best Path')
    plt.plot(best_start_times_cpu, range(len(best_start_times_cpu)), 'r.-', label='Argmax path')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dp_matrix_hard_classic.png')
    plt.close()
        
    return best_start_times

# ------------------second ablations - naive peak detection ------------------

from scipy.signal import find_peaks

def phoneme_alignment_naive_peak_detection(p_seq, w_phi, original_lengths, len_ratio, derivative_preds_np, probs_real):
    """
    Ablation version: Replaces DP with Naive Scipy Peak Detection.
    """
    gamma = 1e-20 
    T = int(original_lengths[0])
    n = len(p_seq)
    device = derivative_preds_np.device

    # --- Keep identical preprocessing to ensure 'plug & play' ---
    if isinstance(probs_real, np.ndarray):
        probs_real = torch.tensor(probs_real, device=device)
    
    # We don't actually need cumsum_probs or phoneme_mappings for naive peak detection,
    # but we keep them defined to avoid any potential scope issues if you add code back.
    cumsum_probs = torch.cumsum(probs_real, dim=0)
    signal = derivative_preds_np.detach().cpu().numpy().flatten()

    # --- Naive Peak Detection ---
    # To get exactly 'n' boundaries for 'n' phonemes, we pick the top n most prominent peaks.
    peaks, properties = find_peaks(signal, prominence=0.05) 
    
    peak_heights = signal[peaks]
    
    # Sort peaks by height and take the top 'n'
    top_indices = np.argsort(peak_heights)[-n:]
    best_peaks = np.sort(peaks[top_indices])
    if len(best_peaks) < n:
        filler = np.linspace(0, T-1, n)
        best_peaks = filler # Fallback
    best_start_times = torch.tensor(best_peaks, dtype=derivative_preds_np.dtype, device=device)

    # --- Mock DP Matrix for Plotting ---
    dp_mat = torch.full((n, T, T), float(-1e9), device=device)
    for i, peak_time in enumerate(best_peaks):
        dp_mat[i, :, int(peak_time)] = 1.0 

    # --- Identical Plotting Logic ---
    dp_mat_cpu = dp_mat.detach().cpu()
    best_start_times_cpu = best_start_times.detach().cpu().numpy()
    dp_to_plot = dp_mat_cpu.max(dim=1)[0].numpy()
    
    masked_dp = np.ma.masked_where(dp_to_plot <= -1e8, dp_to_plot)

    plt.figure(figsize=(12, 6))
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    
    plt.imshow(masked_dp, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label='Peak Detection (Naive)')
    plt.xlabel('End time (frame)')
    plt.ylabel('Phoneme index')
    plt.title('Naive Peak Detection (Ablation)')
    plt.plot(best_start_times_cpu, range(len(best_start_times_cpu)), 'r.-', label='Detected Peaks')
    plt.legend()
    plt.tight_layout()
    save_path = 'peak_detection_ablation.png'
    plt.savefig(save_path)
    plt.close()
    
    print(f"Ablation plot saved as {save_path}")

    return best_start_times

# ------------------------- phoneme alignment main ------------------------
def phoneme_alignment(p_seq, w_phi, original_lengths, len_ratio, derivative_preds_np, probs_real):
    gamma = 1e-20
    T = int(original_lengths[0])
    n = len(p_seq)
    device = derivative_preds_np.device

    if isinstance(probs_real, np.ndarray):
        probs_real = torch.tensor(probs_real, device=device)
    cumsum_probs = torch.cumsum(probs_real, dim=0)
    
    
    
    phoneme_mappings = {p.lower(): timit_to_leehon_map_MACRO.get(p.lower(), 'sil') if p.lower() not in timit_leehon_39_phonemes else p.lower() for p in p_seq}
    derivatives = torch.cat([torch.tensor([0], device=derivative_preds_np.device), torch.diff(derivative_preds_np, dim=0)])

    dp_mat = torch.full((n, T, T), float(-1e9), device=device)

    p_idx0 = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[0].lower()]]

    # Vectorized init for first phoneme
    t_e = torch.arange(T, device=device)
    dp_mat[0, 0, :] = (
        w_phi[0] * compute_phi_1(derivatives, 0, t_e)
        + w_phi[1] * compute_phi_1(derivatives, 0, t_e)

    )

    for i in tqdm(range(1, n)):
        p_idx = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[i].lower()]]
        # Vectorized over t_start and t_end
        t_start = torch.arange(T, device=device)
        t_end = torch.arange(T, device=device)
        t_start_grid, t_end_grid = torch.meshgrid(t_start, t_end, indexing='ij')
        valid_mask = t_start_grid < t_end_grid

        phi1_dev = compute_phi_1(derivatives, t_start_grid, t_end_grid)
        phi1 = compute_phi_1(derivative_preds_np, t_start_grid, t_end_grid)
        phi2 = compute_phi_2(cumsum_probs, p_idx, t_start_grid, t_end_grid)
        total_phi = w_phi[0] * phi1_dev + w_phi[1] * phi2

        # Max over all possible previous end times.
        # Vectorized equivalent of the original per-t_end loop: for each previous
        # end time s, logsumexp over dp_mat[i-1]'s start-rows is the SAME regardless
        # of t_end (invalid spans start>=end are -1e9 and never contribute), so the
        # O(T) inner loop over t_end collapses to one column-wise logsumexp + mask.
        # Bit-identical output; ~T x faster on long utterances.
        col_lse = torch.logsumexp(dp_mat[i-1] / gamma, dim=0) * gamma  # (T,) over start rows
        prev_scores = torch.where(
            valid_mask,
            col_lse.unsqueeze(1).expand(T, T),
            torch.full((T, T), float(-1e9), device=device),
        )
        dp_mat[i] = torch.where(valid_mask, total_phi + prev_scores, torch.full_like(total_phi, float(-1e9)))

    # Backtracking
    best_start_times = torch.zeros((n), dtype=derivative_preds_np.dtype, device=device)
    best_prev_t_end = T-1
    for i in range(n):
        cur_ph = n-1-i
        scores = dp_mat[cur_ph, :, best_prev_t_end]
        soft_weights = torch.softmax(scores / gamma, dim=0)
        expected_idx = (soft_weights * torch.arange(T, device=device, dtype=derivative_preds_np.dtype)).sum()
        best_start_times[cur_ph] = expected_idx      
        best_prev_t_end = int(expected_idx.round().item())
        
    # DP-matrix figure is a debug artifact only; best_start_times (the return
    # value) is already computed above. Skip entirely unless an output dir is set
    # (the web demo sets it). Saves ~570ms/utterance in training/eval.
    if _dp_matrix_out_dir is not None:
        dp_mat_cpu = dp_mat.detach().cpu()
        best_start_times_cpu = best_start_times.detach().cpu().numpy()
        dp_to_plot = dp_mat_cpu.max(dim=1)[0].numpy()
        masked_dp = np.ma.masked_where(dp_to_plot <= -1e8, dp_to_plot)  # mask all values <= -1e8

        plt.figure(figsize=(12, 6))
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')
        real_min = masked_dp.min()
        real_max = masked_dp.max()
        # Plot DP matrix (max over start times)
        plt.imshow(masked_dp, aspect='auto', origin='lower', cmap=cmap, vmin=real_min, vmax=real_max)
        plt.colorbar(label='DP Score (max over start)')
        plt.xlabel('End time (frame)')
        plt.ylabel('Phoneme index')
        plt.title('DP Matrix with Best Path')
        # Overlay best_start_times as a red line
        plt.plot(best_start_times_cpu, range(len(best_start_times_cpu)), 'r.-', label='Best start times')
        plt.legend()
        plt.tight_layout()
        _save_path = os.path.join(_dp_matrix_out_dir or '.', 'dp_matrix_with_path.png')
        plt.savefig(_save_path)
        print(f"DP matrix with path plot saved as {_save_path}")
        
    return best_start_times

def compute_phi_1(derivative_preds_np: torch.Tensor, t_start: Union[torch.Tensor, int], t_end: Union[torch.Tensor, int]) -> torch.Tensor:
    """
    Computes phi_1 for dynamic programming.
    t_start and t_end can be scalars or tensors of the same shape.
    Returns a tensor of scores.
    """
    # Ensure t_start and t_end are tensors
    t_start = torch.as_tensor(t_start, device=derivative_preds_np.device)
    t_end = torch.as_tensor(t_end, device=derivative_preds_np.device)
    # Broadcast to same shape
    t_start, t_end = torch.broadcast_tensors(t_start, t_end)
    # Valid indices
    valid = (t_end < derivative_preds_np.shape[0]-1) & (t_start < derivative_preds_np.shape[0]-1) & (t_end > 0) & (t_start > 0)
    score = torch.zeros_like(t_start, dtype=derivative_preds_np.dtype, device=derivative_preds_np.device)

    eps = 1e-6
    tanh_scale = 1e-3 #1e-2 #0.5
    if valid.any():
        
        # start_pos - 
        idx_s = t_start[valid].long()
        s_center = torch.tanh(tanh_scale * derivative_preds_np[idx_s])
        s_prev = torch.tanh(tanh_scale * derivative_preds_np[idx_s -1])
        s_next = torch.tanh(tanh_scale * derivative_preds_np[idx_s +1])
        delta_prev_s = s_center - s_prev
        delta_next_s = s_center - s_next
        scores_zerocross_s = (1-torch.sqrt(s_center**2)) +  torch.sqrt(delta_prev_s **2 + eps) + torch.sqrt(delta_next_s**2 + eps)
        # orig - 
        score[valid] += scores_zerocross_s

        # end_pos - 
        idx_e = t_end[valid].long()
        e_center = torch.tanh(tanh_scale * derivative_preds_np[idx_e]) #do i need this? not sure
        e_prev = torch.tanh(tanh_scale * derivative_preds_np[idx_e -1])
        e_next = torch.tanh(tanh_scale * derivative_preds_np[idx_e +1])
        delta_prev_e = e_center - e_prev
        delta_next_e = e_center - e_next
        scores_zerocross_e = (1-torch.sqrt(e_center**2)) + torch.sqrt(delta_prev_e **2 + eps) + torch.sqrt(delta_next_e**2 + eps)
        # orig - 
        score[valid] += scores_zerocross_e
    return score

def compute_phi_2(cumsum_probs: torch.Tensor, p: int, t_start: Union[torch.Tensor, int], t_end: Union[torch.Tensor, int]) -> torch.Tensor:
    """
    Computes phi_2 for dynamic programming.
    t_start and t_end can be scalars or tensors of the same shape.
    Returns a tensor of scores.
    """
    t_start = torch.as_tensor(t_start, device=cumsum_probs.device)
    t_end = torch.as_tensor(t_end, device=cumsum_probs.device)
    t_start, t_end = torch.broadcast_tensors(t_start, t_end)
    # Valid indices
    valid = (t_end < cumsum_probs.shape[0]) & (t_start < cumsum_probs.shape[0]) & (t_end > 0) & (t_start >= 0)
    probs_score = torch.zeros_like(t_start, dtype=cumsum_probs.dtype, device=cumsum_probs.device)
    # Only assign where valid
    probs_score[valid] = cumsum_probs[t_end[valid], p] - torch.where(
        t_start[valid] > 0,
        cumsum_probs[t_start[valid], p],
        torch.zeros_like(t_start[valid], dtype=cumsum_probs.dtype, device=cumsum_probs.device)
    )
    lengths = (t_end - t_start).clamp(min=1)
    probs_score[valid] = probs_score[valid] / lengths[valid]
    return (probs_score)

def best_phoneme_for_segments(cumsum_probs: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor):
    """
    For each (t_start, t_end) pair (tensors broadcasted to same shape),
    compute the average probability per phoneme over the segment and return:
      - max_vals: tensor of shape (pairs,) with the max average prob per pair
      - max_idx:  LongTensor of shape (pairs,) with the argmax phoneme index per pair
    """
    device = cumsum_probs.device
    t_start = torch.as_tensor(t_start, device=device)
    t_end = torch.as_tensor(t_end, device=device)
    t_start, t_end = torch.broadcast_tensors(t_start, t_end)

    valid = (t_end < cumsum_probs.shape[0]) & (t_start < cumsum_probs.shape[0]) & (t_end > 0) & (t_start >= 0)
    max_vals = torch.zeros_like(t_start, dtype=cumsum_probs.dtype, device=device)
    max_idx = torch.full_like(t_start, -1, dtype=torch.long, device=device)

    if not valid.any():
        return max_vals, max_idx

    idx_end = t_end[valid].long()
    idx_start = t_start[valid].long()
    probs_end = cumsum_probs[idx_end]  # (k, P)
    probs_start = torch.zeros_like(probs_end)
    nonzero_mask = idx_start > 0
    if nonzero_mask.any():
        probs_start[nonzero_mask] = cumsum_probs[idx_start[nonzero_mask]]

    segment_sum = probs_end - probs_start  # (k, P)
    lengths = (t_end[valid] - t_start[valid]).clamp(min=1).unsqueeze(1).to(segment_sum.dtype)
    segment_mean = segment_sum / lengths  # (k, P)

    vals, idxs = segment_mean.max(dim=1)  # per-row max and argmax
    max_vals[valid] = vals
    max_idx[valid] = idxs.long()
    return max_vals, max_idx

def get_timit_61_phoneme_mappings():
    """
    Returns the TIMIT 61 phoneme-to-index mapping and the reverse index-to-phoneme mapping.

    Returns:
        phoneme_to_idx (dict): Dictionary mapping phonemes to unique indices.
        idx_to_phoneme (dict): Dictionary mapping indices to their corresponding phonemes.
    """
    # this is actually including the leehon 39 phonemes!!!!!
    timit_61_phonemes = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
        'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
        'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng',
        'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
        'v', 'w', 'y', 'z', 'zh'
    ]
    timit_leehon_39_phonemes = [
        'ao', 'ae', 'ah','aw', 'er', 'ay', 
        'b', 'sil', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'm', 'en', 'ng', 'ey',
        'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'v', 'w', 'y', 'z', 'sh', 't', 'r', 's', 'th','uh', 'uw', 'oy', 'ow','p'
    ]
    # Create mappings
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(timit_leehon_39_phonemes)}
    idx_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_idx.items()}

    return phoneme_to_idx, idx_to_phoneme

# --------------------------------


def timit_to_leehon(timit_label):
    # Mapping of TIMIT 61 phonemes to Leehon 39 phonemes
    timit_to_leehon_map = {
        'aa': 'ao', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'ay': 'ay',
        'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 'dx', 'eh': 'eh', 'el': 'el',
        'em': 'm', 'en': 'en', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g', 'gcl': 'sil',
        'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k', 'kcl': 'sil',
        'l': 'el', 'm': 'm', 'n': 'en', 'ng': 'ng', 'nx': 'en', 'ow': 'ow', 'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil',
        'q': 't', 'qcl': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw',
        'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'sh', '':'sil'
    }
    
    # Return the corresponding Leehon 39 label, or None if the label is not found
    return timit_to_leehon_map.get(timit_label.lower(), None)

def load_phoneme_stats():
    phonemes_path = "phonemes_39"
    stats_path = "phoneme_stats_39.out"
    
    # Load phoneme names
    with open(phonemes_path, "r") as f:
        phonemes = [line.strip() for line in f]

    # Load mu values (second row of stats file)
    with open(stats_path, "r") as f:
        lines = f.readlines()
        mu_values = list(map(float, lines[1].strip().split()))  # Convert to float
        sigma_values = list(map(float, lines[2].strip().split()))  # Convert to float
    
    # Create phoneme-to-mu dictionary
    phoneme_mu_dict = dict(zip(phonemes, mu_values))
    phoneme_sigma_dict = dict(zip(phonemes, sigma_values))
    
    return phoneme_mu_dict, phoneme_sigma_dict

# Load phoneme stats once


def get_mu_stats(p):
    phoneme_mu_dict, _ = load_phoneme_stats()
    """Return the mu value for the given phoneme p."""
    return phoneme_mu_dict.get(p, None)  # Return None if phoneme is not found

def get_sigma_stats(p):
    _, phoneme_sigma_dict = load_phoneme_stats()
    """Return the mu value for the given phoneme p."""
    return phoneme_sigma_dict.get(p, None)  # Return None if phoneme is not found

def replicate_first_k_frames(x, k, dim):
    return torch.cat([x.index_select(dim=dim, index=torch.LongTensor([0] * k).to(x.device)), x], dim=dim)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class PrintShapeLayer(nn.Module):
    def __init__(self):
        super(PrintShapeLayer, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

def detect_peaks_worker(xi,w_phi, p_seq, original_lengths, probs_real, len_ratio, width, distance):
    print(f"num peaks = {len(p_seq)}")
    print(f"xi type: {type(xi)}")
    preds_np = xi.requires_grad_(True)
    median_h = preds_np.median()
    preds_np = preds_np - median_h
    derivative_preds_np = preds_np
    xmin, xmax = xi.min(), xi.max()
    xi = (xi - xmin) / (xmax - xmin)
    xi = xi.flatten()
    
    peaks = phoneme_alignment(p_seq,w_phi, original_lengths, len_ratio, derivative_preds_np, probs_real)
    
    if len(peaks) == 0:
        peaks = torch.tensor([xi.shape[0] - 1], device=xi.device)
    
    return peaks

def detect_peaks(x,w_phi, original_lengths_all, phonemes, len_ratio, probs_real_all):
    """Detect peaks of next_frame_classifier using multithreading."""
    
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        xi=x
        p_seq = phonemes
        original_lengths = original_lengths_all
        probs_real = probs_real_all
        if len(xi)!=0:
            result = detect_peaks_worker(xi, w_phi, p_seq, [original_lengths], probs_real, len_ratio, width=None, distance=None)
        out.append(result)
    
    return out

class PrecisionRecallMetric:
    def __init__(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.eps = 1e-5
        self.data = []
        self.tolerance = 2
        self.width_range = [1]
        self.distance_range = [1]

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        EPS = 1e-7
        
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        os = recall / (precision + EPS) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

        return precision, recall, f1, rval

    def zero(self):
        self.data = []
        
    def update(self, seg, pos_pred, length,original_lengths_all, probs_all,phonemes_all):
        for seg_i, pos_pred_i, length_i , original_length, probs,phonemes in zip(seg, pos_pred, length,original_lengths_all,probs_all,phonemes_all):
            self.data.append((seg_i, pos_pred_i, length_i.item(),[original_length.item()], probs, phonemes))


    def get_stats(self, width=None, distance=None):
        print(f"calculating metrics using {len(self.data)} entries")
        max_rval = -float("inf")
        min_l1_dist = float("inf")
        best_params = None
        segs = list(map(lambda x: x[0], self.data))
        length = list(map(lambda x: x[2], self.data))
        yhats = list(map(lambda x: x[1], self.data))
        original_lengths_all = list(map(lambda x: x[3], self.data))
        probs = list(map(lambda x: x[4], self.data))
        phonemes = list(map(lambda x: x[5], self.data))

        width_range = self.width_range
        distance_range = self.distance_range

        if width is not None:
            width_range = [width]
            distance_range = [distance]
        sr = 16000
        len_ratio = 161.34011627906978

        for width in width_range:
            for distance in distance_range:
                    for (y, yhat,original_len, phoneme, prob) in zip(segs, yhats, original_lengths_all, phonemes, probs):
                        if isinstance(y,list):
                            y = torch.tensor(y, device=yhat.device, dtype=yhat.dtype)
                        peaks = detect_peaks(x=yhat,w_phi= [0.5,0.5],
                                             original_lengths_all = original_len[0],
                                             phonemes = phoneme,
                                             len_ratio = 161.34011627906978 , 
                                             probs_real_all = prob)
                        peaks = peaks[0]* len_ratio/sr
                        yhat = peaks
                        yhat = yhat[1:]

                        if isinstance(y,list):
                            y = torch.tensor(y, device=yhat.device, dtype=yhat.dtype)
                            y = y*len_ratio/sr
                        l1_dist = torch.mean(torch.abs(y - yhat)).item()
                        l2_dist = torch.mean((y - yhat)**2).item()
                        if l1_dist<min_l1_dist:
                            min_l1_dist = l1_dist
                            out = (l1_dist,l2_dist)
                            best_params = width, distance
        self.zero()
        print(f"best peak detection params: {best_params} (width, distance)")
        print(f"best peak detection L1_DIST: {l1_dist}")
        print(f"best peak detection L2_DIST: {l2_dist}")
        return out, best_params


class StatsMeter:
    def __init__(self):
        self.data = []

    def update(self, item):
        if type(item) == list:
            self.data.extend(item)
        else:
            self.data.append(item)

    def get_stats(self):
        data = np.array(self.data)
        if len(data)==0:
            return float('nan')
        mean = data.mean()
        return mean

    def zero(self):
        self.data.clear()
        assert len(self.data) == 0, "StatsMeter didn't clear"


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.msg} -- started")

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"{self.msg} -- done in {(time.time() - self.start_time)} secs")


def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x


def line():
    print(90 * "-")