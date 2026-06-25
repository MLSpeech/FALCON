"""
FALCON web demo — interactive forced alignment in the browser.

Two pretrained checkpoints are used by FALCON (under pretrained_models/):
  - falcon_timit_english.pt       — TIMIT-trained, best for English phoneme alignment
  - falcon_joint_multilingual.pt  — joint TIMIT+Buckeye model; best for cross-lingual /
                                   multilingual zero-shot alignment (Dutch, German,
                                   Hebrew, ...) at both phoneme and word level.

The app picks one automatically from the `Language` radio (english → TIMIT,
multilingual → joint); a custom .pt upload overrides both.

For HuggingFace Spaces deployment, set Space Secrets:
  HF_MODEL_REPO   — e.g. "MLSpeech/FALCON-weights"
  HF_TOKEN        — only needed for private repos
The app will download `falcon_timit_english.pt` and `falcon_joint_multilingual.pt`
from that repo on first use.
"""
import os
import re
import shutil
import tempfile
import threading
import time

import gradio as gr
import textgrid
import torchaudio

import utils
from predict import main_predict

# ── Checkpoint configuration ──────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained_models")

CKPT_FILES = {
    "english":      "falcon_timit_english.pt",
    "buckeye":      "falcon_buckeye_english.pt",
    "multilingual": "falcon_joint_multilingual.pt",
}
CKPT_LABELS = {
    "english":      "Read English (recommended) — TIMIT model",
    "buckeye":      "Spontaneous English (recommended) — Buckeye model",
    "multilingual": "Multilingual — joint TIMIT+Buckeye model",
}

_ckpt_cache = {}

def _resolve_ckpt(key: str):
    """Resolve checkpoint path: cache → local file → HF Hub. Returns None if all fail."""
    if key in _ckpt_cache and os.path.exists(_ckpt_cache[key]):
        return _ckpt_cache[key]

    filename = CKPT_FILES[key]
    local_path = os.path.join(PRETRAINED_DIR, filename)
    if os.path.exists(local_path):
        _ckpt_cache[key] = local_path
        return local_path

    repo = os.environ.get("HF_MODEL_REPO", "")
    if repo:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                token=os.environ.get("HF_TOKEN"),
            )
            _ckpt_cache[key] = path
            return path
        except Exception as exc:
            print(f"[FALCON] HF Hub fetch failed for {filename}: {exc}")

    return None

_inference_lock = threading.Lock()

# ── Heartbeat-based auto-shutdown (local runs only) ───────────────────────────
# The open browser tab pings _heartbeat() every few seconds. A watchdog thread
# exits the process when the pings stop (tab closed / browser crashed / unload
# event dropped). _last_ping stays None until the first browser connects, so the
# server never self-exits before anyone opens it.
_last_ping = [None]
_HEARTBEAT_TIMEOUT = 90  # secs of silence before the local server self-exits

def _heartbeat():
    _last_ping[0] = time.time()

def _start_shutdown_watchdog():
    def _watch():
        while True:
            time.sleep(5)
            last = _last_ping[0]
            if last is not None and (time.time() - last) > _HEARTBEAT_TIMEOUT:
                os._exit(0)
    threading.Thread(target=_watch, daemon=True).start()

# ── Internal language routing ────────────────────────────────────────────────

def _internal_language(lang: str, mode: str, ann_ext: str) -> str:
    """
    Map UI choices to the internal `language` flag understood by main_predict.

      'english' = no G2P; assumes labels are already TIMIT-39 phonemes.
      'dutch'   = G2P pipeline (panphon-based articulatory mapping). Used for:
                    • any non-English language
                    • word-level alignment (.wrd, words need phoneme decomposition)
                    • plain text input (could be words or arbitrary phonemes)

    NOTE: `ann_ext` here must be the *original* extension supplied by the user —
    not the post-rewrite extension after a .txt → dummy .phn synthesis.
    """
    if lang == "english" and mode == "phoneme" and ann_ext.lower() == "phn":
        return "english"
    return "dutch"

# ── Word-level G2P selection ──────────────────────────────────────────────────

# Optional input-language hint -> (espeak voice, MFA voice or None). MFA ships
# pronunciation models only for en/de/nl; everything else uses espeak, or "none"
# (romanized characters -> LH39) when even espeak has no voice.
G2P_LANG_CHOICES = [
    "English (default)", "German", "Dutch", "Hebrew", "French",
    "Spanish", "Italian", "Russian", "Portuguese", "Other / unknown",
]
_G2P_LANG_MAP = {
    "English (default)": ("en-us", "en-us"),
    "German":            ("de", "de"),
    "Dutch":             ("nl", "nl"),
    "Hebrew":            ("he", None),
    "French":            ("fr", None),
    "Spanish":           ("es", None),
    "Italian":           ("it", None),
    "Russian":           ("ru", None),
    "Portuguese":        ("pt", None),
    "Other / unknown":   ("en-us", None),
}

def _resolve_g2p(g2p_choice, lang_choice):
    """Map the (G2P option, input-language) UI choices to a concrete backend.

    Returns (backend, voice, note). backend in {"mfa", "espeak", "char"}. Honors
    an explicit espeak / MFA-like / none choice but auto-falls-back when the
    chosen backend has no model for the language; "Auto" picks the best available.
    """
    espeak_voice, mfa_voice = _G2P_LANG_MAP.get(lang_choice, ("en-us", "en-us"))
    try:
        import mfa_g2p
        mfa_ok = mfa_voice is not None and mfa_g2p.mfa_available(mfa_voice)
    except Exception:
        mfa_ok = False
    choice = (g2p_choice or "Auto").lower()

    if choice.startswith("none"):
        return "char", espeak_voice or "en-us", "none (romanization)"
    if choice.startswith("mfa"):
        if mfa_ok:
            return "mfa", mfa_voice, "MFA-like"
        if espeak_voice:
            return "espeak", espeak_voice, "espeak (no MFA model for this language)"
        return "char", "en-us", "none (no MFA/espeak model)"
    if choice.startswith("espeak"):
        if espeak_voice:
            return "espeak", espeak_voice, "espeak"
        if mfa_ok:
            return "mfa", mfa_voice, "MFA-like (no espeak voice for this language)"
        return "char", "en-us", "none"
    # Auto (recommended)
    if mfa_ok:
        return "mfa", mfa_voice, "MFA-like (auto)"
    if espeak_voice:
        return "espeak", espeak_voice, "espeak (auto)"
    return "char", "en-us", "none (auto)"

# ── Core handler ──────────────────────────────────────────────────────────────

OUTPUTS_NONE = (None, None, None, None, None)  # 5 None for the non-status outputs

def run_alignment(audio_file, annotation_file, ckpt_upload, mode, lang,
                  pretrained_choice, w_phi, g2p_choice="Auto (recommended)",
                  lang_choice="English (default)",
                  progress=gr.Progress(track_tqdm=True)):
    if not audio_file or not annotation_file:
        return ("Please upload both an audio file and an annotation file.", *OUTPUTS_NONE)

    ckpt_to_use = ckpt_upload if ckpt_upload else _resolve_ckpt(pretrained_choice)
    if not ckpt_to_use or not os.path.exists(ckpt_to_use):
        return (
            f"No checkpoint found. Expected {CKPT_FILES[pretrained_choice]} "
            f"in {PRETRAINED_DIR}, or HF_MODEL_REPO set, or upload a .pt file.",
            *OUTPUTS_NONE,
        )

    progress(0.1, desc="Preparing workspace...")
    workspace = tempfile.mkdtemp(prefix="falcon_")
    base = "input"
    wav_path = os.path.join(workspace, f"{base}.wav")

    original_ext = os.path.basename(annotation_file).split(".")[-1].lower()
    ann_ext = original_ext
    ann_path = os.path.join(workspace, f"{base}.{ann_ext}")

    # Resample audio to 16 kHz mono
    try:
        audio, sr = torchaudio.load(audio_file)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        torchaudio.save(wav_path, audio, 16000)
    except Exception as exc:
        return (f"Audio error: {exc}", *OUTPUTS_NONE)

    shutil.copy(annotation_file, ann_path)

    # Capture the original input tokens (whatever the user supplied per line):
    #   .phn → phoneme labels
    #   .wrd → word labels
    #   .txt → space-separated tokens (words or phonemes)
    if original_ext == "txt":
        with open(ann_path) as f:
            orig_tokens = re.sub(r"[^\w\s]", "", f.read().strip()).split()
        # TIMIT .txt files are "<start_sample> <end_sample> <sentence>" — drop the
        # leading sample indices so they aren't mistaken for words.
        if len(orig_tokens) >= 3 and orig_tokens[0].isdigit() and orig_tokens[1].isdigit():
            orig_tokens = orig_tokens[2:]
        if not orig_tokens:
            return ("Text annotation is empty after stripping punctuation.", *OUTPUTS_NONE)
        # Synthesize a uniform-segments dummy .phn so downstream code has timestamps.
        audio_len = audio.shape[1]
        interval = audio_len / len(orig_tokens)
        ann_ext = "phn"
        ann_path = os.path.join(workspace, f"{base}.{ann_ext}")
        with open(ann_path, "w") as f:
            for i, tok in enumerate(orig_tokens):
                f.write(f"{int(i * interval)} {int((i + 1) * interval)} {tok}\n")
    else:
        with open(ann_path) as f:
            orig_tokens = [ln.strip().split()[-1] for ln in f if ln.strip()]

    # Route by ORIGINAL extension (post-rewrite ann_ext is "phn" for txt inputs).
    language = _internal_language(lang, mode, original_ext)

    # Word-level G2P: convert orthographic words -> LH39 phonemes with the chosen
    # front-end (espeak, or the MFA english_us_arpa G2P used in the paper), then
    # align via the stock phoneme path. Replaces the legacy letter-by-letter
    # mapping. Only applies to real word input (.wrd / .txt / .word).
    # Word/text inputs (.wrd / .txt) always go through the proper word -> LH39 G2P,
    # regardless of the phoneme/word toggle — otherwise phoneme mode would fall back
    # to a crude character mapping and misalign. (.phn is the phoneme-input path.)
    app_mapped_ph = None
    word_g2p_note = ""
    if original_ext in ("wrd", "txt", "word") and orig_tokens:
        import word_g2p
        backend, voice, g2p_note = _resolve_g2p(g2p_choice, lang_choice)
        try:
            app_mapped_ph = [word_g2p.word_to_lh39(tok, voice=voice, backend=backend)
                             for tok in orig_tokens]
        except Exception as g2p_exc:
            # A missing model must never break the whole run — fall back.
            print(f"[FALCON] G2P backend '{backend}' failed ({g2p_exc}); falling back.")
            try:
                app_mapped_ph = [word_g2p.word_to_lh39(tok, voice=voice or "en-us",
                                                       backend="espeak")
                                 for tok in orig_tokens]
                g2p_note += " → espeak fallback"
            except Exception:
                app_mapped_ph = [word_g2p.word_to_lh39(tok, backend="char")
                                 for tok in orig_tokens]
                g2p_note += " → none fallback"
        word_g2p_note = f" Word G2P: {g2p_note}."
        phons = [ph for seq in app_mapped_ph for ph in seq] or ["sil"]
        # Rewrite the annotation as a dummy uniform-time .phn of LH39 phonemes so
        # the stock English/phoneme aligner runs on them (no further G2P).
        ann_ext = "phn"
        ann_path = os.path.join(workspace, f"{base}.{ann_ext}")
        audio_len = audio.shape[1]
        interval = audio_len / max(1, len(phons))
        with open(ann_path, "w") as f:
            for i, ph in enumerate(phons):
                f.write(f"{int(i * interval)} {int((i + 1) * interval)} {ph}\n")
        language = "english"   # phonemes are already LH39; skip the internal G2P

    progress(0.3, desc="Running alignment...")
    try:
        with _inference_lock:
            utils.set_dp_matrix_out_dir(workspace)
            pred_bound, _truth_bound, mapped_ph = main_predict(
                wav=wav_path,
                ckpt=ckpt_to_use,
                w_phi=w_phi,
                language=language,
                annotation=ann_ext,
            )
            utils.set_dp_matrix_out_dir(None)
            progress(0.6, desc="Rendering aligned visualization...")
            # Time-aligned representations as one stacked figure (waveform,
            # spectrogram, phoneme posteriors, Soft-DP matrix + path, contrastive
            # score) — all on the same time axis with predicted boundaries overlaid.
            panels_path = os.path.join(workspace, "panels.png")
            try:
                import falcon_viz
                falcon_viz.make_alignment_panels(
                    wav=wav_path, ckpt=ckpt_to_use, out_path=panels_path,
                    language=language, annotation=ann_ext,
                    show_truth=(original_ext == "phn" and mode == "phoneme"),
                )
            except Exception as viz_exc:
                print(f"[FALCON] panel viz failed: {viz_exc}")
                panels_path = None
    except Exception as exc:
        utils.set_dp_matrix_out_dir(None)
        return (f"Inference error: {exc}", *OUTPUTS_NONE)

    # When the app did the word-level G2P itself, use its per-word LH39 phoneme
    # lists for the two-table / TextGrid word tier (main_predict's English path
    # returns mapped_ph=None).
    if app_mapped_ph is not None:
        mapped_ph = app_mapped_ph

    progress(0.8, desc="Building outputs...")

    pred_bound_list = [float(t) for t in pred_bound]

    # ── Build two tables ─────────────────────────────────────────────────────
    # 1) LH39 phonemes  — the aligner's direct output, one row per pred_bound.
    # 2) Original tokens — words (.wrd / .txt) or non-LH39 phonemes (multilingual
    #    .phn). Only populated when the G2P path was used (mapped_ph != None);
    #    for english+phoneme+.phn the LH39 phonemes ARE the original, so the
    #    second table is left empty.
    # Every input token gets a row in the table even if its predicted interval
    # is degenerate; degenerate intervals are still kept out of the TextGrid
    # (Praat rejects zero-length).

    if mapped_ph is not None:
        phn_labels = [ph for seq in mapped_ph for ph in seq]
    else:
        phn_labels = orig_tokens

    table_phonemes, phn_intervals = [], []
    t0 = 0.0
    for i, t1 in enumerate(pred_bound_list):
        lbl = phn_labels[i] if i < len(phn_labels) else ""
        table_phonemes.append([round(t0, 3), round(t1, 3), lbl])
        if t1 > t0:
            phn_intervals.append((t0, t1, lbl))
        t0 = t1

    table_original, orig_intervals = [], []
    if mapped_ph is not None:
        counts_per_token = [len(seq) for seq in mapped_ph]
        cumulative = 0
        t0 = 0.0
        for i, count in enumerate(counts_per_token):
            cumulative += count
            if cumulative - 1 >= len(pred_bound_list):
                break
            t1 = pred_bound_list[cumulative - 1]
            lbl = orig_tokens[i] if i < len(orig_tokens) else ""
            table_original.append([round(t0, 3), round(t1, 3), lbl])
            if t1 > t0:
                orig_intervals.append((t0, t1, lbl))
            t0 = t1

    # ── TextGrid: phones tier always, original tier when applicable ──────────
    max_time = phn_intervals[-1][1] if phn_intervals else 0.0
    tg = textgrid.TextGrid(minTime=0, maxTime=max_time)
    tier_phn = textgrid.IntervalTier(name="phones", minTime=0, maxTime=max_time)
    for t0_iv, t1_iv, lbl_iv in phn_intervals:
        tier_phn.add(minTime=t0_iv, maxTime=t1_iv, mark=lbl_iv)
    tg.append(tier_phn)
    if orig_intervals:
        # Tier name reflects what the original layer represents.
        if mode == "word":
            orig_tier_name = "words"
        elif original_ext == "phn":
            orig_tier_name = "phones_original"
        else:
            orig_tier_name = "tokens"
        tier_orig = textgrid.IntervalTier(name=orig_tier_name, minTime=0, maxTime=max_time)
        for t0_iv, t1_iv, lbl_iv in orig_intervals:
            tier_orig.add(minTime=t0_iv, maxTime=t1_iv, mark=lbl_iv)
        tg.append(tier_orig)
    tg_path = os.path.join(workspace, f"{base}.TextGrid")
    tg.write(tg_path)

    # Status note: the .phn-as-word case produces a second "words" table that
    # actually contains phonemes — surface this in the status so it's not
    # mistaken for a bug.
    status_note = word_g2p_note
    if mode == "word" and original_ext == "phn":
        status_note += (" Note: input was phoneme-level (.phn) but mode=word "
                        "— the 'original' table shows input phonemes since no "
                        "word annotations were provided.")

    return (
        "Done." + status_note,
        audio_file,
        panels_path if panels_path and os.path.exists(panels_path) else None,
        tg_path,
        table_phonemes,
        table_original,
    )

# ── UI ────────────────────────────────────────────────────────────────────────

PRETRAINED_RADIO_CHOICES = [
    (CKPT_LABELS["english"],      "english"),
    (CKPT_LABELS["buckeye"],      "buckeye"),
    (CKPT_LABELS["multilingual"], "multilingual"),
]

with gr.Blocks(title="FALCON Forced Aligner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FALCON: Forced Alignment through Contrastive Optimization Networks")
    gr.Markdown(
        "Upload a speech file and a transcript to predict precise phoneme or word boundaries "
        "using Soft Dynamic Programming."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            audio_in  = gr.Audio(label="Audio file (any sample rate)", type="filepath")
            ann_in    = gr.File(label="Annotation (.phn / .wrd / .txt)")
            mode_in   = gr.Radio(["phoneme", "word"], value="phoneme", label="Mode")
            lang_in   = gr.Radio(["english", "multilingual"], value="english", label="Language")
            # Word-level G2P front-end (word mode only). The espeak voice / MFA
            # dictionary is chosen automatically from the optional input-language
            # hint below; "Auto" also picks the best available backend.
            g2p_in    = gr.Radio(
                ["Auto (recommended)", "espeak", "MFA-like",
                 "none — romanization (not recommended; only for languages with no G2P model)"],
                value="Auto (recommended)",
                label="Word G2P (used only in word mode)",
            )
            lang_g2p_in = gr.Dropdown(
                G2P_LANG_CHOICES,
                value="English (default)",
                label="Input language (optional, recommended — improves G2P choice)",
            )
            pretrained_in = gr.Radio(
                choices=PRETRAINED_RADIO_CHOICES,
                value="english",
                label="Pretrained checkpoint (auto-follows Language; override here if you want)",
            )
            ckpt_in   = gr.File(label="Or upload a custom checkpoint (.pt) — overrides pretrained",
                                file_types=[".pt"], type="filepath")
            wphi_in   = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                  label="φ weight (acoustic ↔ linguistic)")
            btn       = gr.Button("Run Alignment", variant="primary")
            status    = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### Outputs")
            with gr.Tabs():
                with gr.Tab("Alignment Data"):
                    audio_out  = gr.Audio(label="Playback", interactive=False)
                    table_phn_out = gr.Dataframe(
                        headers=["Start (s)", "End (s)", "Phoneme (LH39)"],
                        label="LH39 phonemes — aligner output",
                    )
                    table_orig_out = gr.Dataframe(
                        headers=["Start (s)", "End (s)", "Label"],
                        label="Original input layer (words / non-LH39 phonemes) — empty for English phoneme alignment",
                    )
                    tg_out     = gr.File(label="Download TextGrid (carries both tiers when applicable)")
                with gr.Tab("Visualizations"):
                    img_panels = gr.Image(
                        label="Time-aligned representations — waveform · spectrogram · phoneme posteriors · Soft-DP path · contrastive score (shared time axis; predicted boundaries overlaid). Click to enlarge.",
                        show_download_button=True,
                    )

    # Auto-flip the pretrained-checkpoint radio when the user changes language.
    lang_in.change(fn=lambda v: v, inputs=lang_in, outputs=pretrained_in)

    btn.click(
        fn=run_alignment,
        inputs=[audio_in, ann_in, ckpt_in, mode_in, lang_in, pretrained_in, wphi_in,
                g2p_in, lang_g2p_in],
        outputs=[status, audio_out, img_panels,
                 tg_out, table_phn_out, table_orig_out],
    )

    # Auto-shutdown the local Python server when the user closes the browser
    # tab. Disabled on HuggingFace Spaces (where SPACE_ID is set automatically)
    # because the container is shared across visitors — one tab close should
    # not tear down everyone else's session.
    if not os.environ.get("SPACE_ID"):
        # Fast path: unload events click a hidden Shutdown button -> os._exit.
        shutdown_btn = gr.Button("Shutdown", visible=False, elem_id="falcon-shutdown-btn")
        shutdown_btn.click(fn=lambda: os._exit(0), inputs=[], outputs=[])

        # Guaranteed fallback: the page heartbeats; the watchdog exits if it stops.
        hb_btn = gr.Button("hb", visible=False, elem_id="falcon-heartbeat-btn")
        hb_btn.click(fn=_heartbeat, inputs=[], outputs=[],
                     show_progress="hidden", queue=False)
        _start_shutdown_watchdog()

        demo.load(None, None, None, js="""
            () => {
                const stop = () => {
                    const btn = document.getElementById('falcon-shutdown-btn');
                    if (btn) btn.click();
                };
                window.addEventListener('beforeunload', stop);
                window.addEventListener('pagehide', stop);
                const beat = () => {
                    const hb = document.getElementById('falcon-heartbeat-btn');
                    if (hb) hb.click();
                };
                beat();
                setInterval(beat, 10000);
            }
        """)

if __name__ == "__main__":
    demo.launch()
