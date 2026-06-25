import os
import argparse
import torchaudio
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

# Import main_predict from predict.py without modifying it
from predict import main_predict, resolve_internal_language, resolve_default_ckpt
import dutch_preprocess
from utils import timit_to_leehon_map_MACRO, timit_61_phonemes

def create_textgrid(duration, tiers):
    """
    tiers is a list of dictionaries:
    [
        {"name": "words", "intervals": [(xmin, xmax, text), ...]},
        {"name": "phones", "intervals": [(xmin, xmax, text), ...]}
    ]
    """
    tg = 'File type = "ooTextFile"\n'
    tg += 'Object class = "TextGrid"\n\n'
    tg += 'xmin = 0 \n'
    tg += f'xmax = {duration:.6f} \n'
    tg += 'tiers? <exists> \n'
    tg += f'size = {len(tiers)} \n'
    tg += 'item []: \n'
    
    for tier_idx, tier in enumerate(tiers, 1):
        tg += f'    item [{tier_idx}]:\n'
        tg += '        class = "IntervalTier" \n'
        tg += f'        name = "{tier["name"]}" \n'
        tg += '        xmin = 0 \n'
        tg += f'        xmax = {duration:.6f} \n'
        intervals = tier["intervals"]
        tg += f'        intervals: size = {len(intervals)} \n'
        for i, (xmin, xmax, text) in enumerate(intervals, 1):
            tg += f'        intervals [{i}]:\n'
            tg += f'            xmin = {xmin:.6f} \n'
            tg += f'            xmax = {xmax:.6f} \n'
            tg += f'            text = "{text}" \n'
            
    return tg

def process_single_file(wav_path, ckpt, mode, lang, annotation):
    language = resolve_internal_language(lang, mode, annotation)
    
    base_dir = os.path.dirname(wav_path)
    base_name = os.path.basename(wav_path).replace('.wav', '')
    
    actual_annotation = annotation
    if annotation.lower() == "txt":
        txt_path = os.path.join(base_dir, f"{base_name}.txt")
        with open(txt_path, "r") as f:
            import re
            raw_text = f.read().strip()
            clean_text = re.sub(r'[^\w\s]', '', raw_text)
            labels = clean_text.split()
            
        audio, sr = torchaudio.load(wav_path)
        audio_len = audio.shape[1]
        interval = audio_len / len(labels) if len(labels) > 0 else 0
        
        actual_annotation = "dummy_phn"
        dummy_path = os.path.join(base_dir, f"{base_name}.{actual_annotation}")
        with open(dummy_path, "w") as f:
            for i, label in enumerate(labels):
                end_frame = int((i + 1) * interval)
                f.write(f"0 {end_frame} {label}\n")

    # Run prediction
    pred_bound, truth_bound, mapped_ph = main_predict(wav_path, ckpt, w_phi=0.5, language=language, annotation=actual_annotation)
    
    # Read original labels
    ann_path = os.path.join(base_dir, f"{base_name}.{actual_annotation}")
    
    with open(ann_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(" ") for line in lines if line.strip()]
        labels = [line[2] for line in lines]
        
    if annotation.lower() == "txt" and os.path.exists(dummy_path):
        os.remove(dummy_path)
    
    audio, sr = torchaudio.load(wav_path)
    duration = audio.shape[1] / sr
    
    # Convert predictions to a list of boundaries
    if isinstance(pred_bound, torch.Tensor):
        pred_bound = pred_bound.detach().cpu().tolist()
    elif isinstance(pred_bound, np.ndarray):
        pred_bound = pred_bound.tolist()
        
    bounds = pred_bound + [duration]
    
    tiers = []
    
    if language == "dutch":
        # We need two tiers
        # mapped_ph is returned by main_predict directly
        
        flat_lh39 = [ph for seq in mapped_ph for ph in seq]
        
        if len(flat_lh39) + 1 != len(bounds):
            print(f"Warning: Boundary count mismatch in {wav_path}. Expected {len(flat_lh39) + 1}, got {len(bounds)}.")
            
        # Tier 2: LH39 English Phonemes (or just phones)
        phone_intervals = []
        for k in range(len(flat_lh39)):
            xmin = bounds[k]
            xmax = bounds[k+1] if k+1 < len(bounds) else duration
            text = flat_lh39[k]
            phone_intervals.append((xmin, xmax, text))
            
        # Tier 1: Original Labels (words or non-English phonemes)
        orig_intervals = []
        k = 0
        for i, l_i in enumerate(labels):
            seq_len = len(mapped_ph[i])
            xmin = bounds[k]
            k += seq_len
            xmax = bounds[k] if k < len(bounds) else duration
            text = l_i
            orig_intervals.append((xmin, xmax, text))
            
        tier1_name = "words" if mode == "word" else "original_phones"
        tiers.append({"name": tier1_name, "intervals": orig_intervals})
        tiers.append({"name": "phones", "intervals": phone_intervals})
        
    else:
        # Language == english, original labels are the phones
        phone_intervals = []
        for k in range(min(len(labels), len(bounds)-1)):
            xmin = bounds[k]
            xmax = bounds[k+1]
            text = labels[k]
            phone_intervals.append((xmin, xmax, text))
            
        tiers.append({"name": "phones", "intervals": phone_intervals})
        
    tg_str = create_textgrid(duration, tiers)
    out_path = os.path.join(base_dir, f"{base_name}.TextGrid")
    with open(out_path, "w") as f:
        f.write(tg_str)
    print(f"Saved TextGrid to {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate TextGrid files from FALCON predictions')
    parser.add_argument('--wav', help='path to a single wav file')
    parser.add_argument('--wav_dir', help='path to a directory of wav files')
    parser.add_argument('--ckpt', default=None,
                        help='Path to checkpoint. If omitted, uses the bundled '
                             'TIMIT-trained checkpoint for --lang english or the '
                             'Buckeye-trained one for --lang multilingual.')

    parser.add_argument('--mode', type=str, default='phoneme', choices=['phoneme', 'word'])
    parser.add_argument('--lang', type=str, default='english', choices=['english', 'multilingual'])
    parser.add_argument('--annotation', type=str, default='phn')
    args = parser.parse_args()

    ckpt = args.ckpt or resolve_default_ckpt(args.lang)

    if args.wav:
        process_single_file(args.wav, ckpt, args.mode, args.lang, args.annotation)
    elif args.wav_dir:
        wavs = [f for f in os.listdir(args.wav_dir) if f.lower().endswith(".wav")]
        for wav_file in tqdm(wavs, desc="Processing WAV files"):
            wav_path = os.path.join(args.wav_dir, wav_file)
            try:
                process_single_file(wav_path, ckpt, args.mode, args.lang, args.annotation)
            except Exception as e:
                print(f"Failed to process {wav_path}: {e}")
    else:
        print("Please provide either --wav or --wav_dir")

if __name__ == "__main__":
    main()
