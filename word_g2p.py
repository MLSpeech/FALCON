"""
Stand-alone word -> Lee-Hon-39 phoneme front-end for *word-level* alignment.

This converts orthographic words into a phoneme sequence so the (otherwise
phoneme-level) aligner can run on word input. It is deliberately independent
of MFA / any user-supplied pronunciation dictionary or acoustic model:

    word  --espeak (grapheme->IPA, rule-based, multilingual)-->  IPA
    IPA   --panphon ipa_segs-->  IPA segments
    seg   --dutch_preprocess.find_best_leehon39 (articulatory distance)-->  LH39

The IPA->LH39 step is the *same* panphon mapping already used on the phoneme
path, so word-level and phoneme-level share one LH39 back-end. Only the word
case routes through here; the phoneme path is unchanged.

espeak is a stand-alone open-source phonemizer (not MFA), so nothing here
depends on the system we compare against at runtime.
"""
import os
import shutil
import subprocess

import panphon

import dutch_preprocess

# espeak voice used for word-level G2P. "en-us" = American English (TIMIT).
# Override per-language via the FDNFA_G2P_VOICE env var (e.g. "nl", "de", "he").
DEFAULT_VOICE = os.environ.get("FDNFA_G2P_VOICE", "en-us")

# Insert TIMIT-style closure/silence segments so the word-derived phoneme
# sequence matches the model's training granularity. Closures/silences are ~22%
# of TIMIT reference segments (all map to LH39 'sil') and no text G2P emits
# them; restoring them via this parameter-free rule recovers most of the
# word-vs-phoneme gap (+~16 pts @25ms). Disable with FDNFA_WORD_CLOSURES=0.
USE_CLOSURES = os.environ.get("FDNFA_WORD_CLOSURES", "1").lower() not in ("0", "false", "no")
STOPS = {"b", "d", "g", "p", "t", "k"}

_ft = panphon.FeatureTable()
_cache = {}  # (word_lower, voice) -> [lh39, ...]
# espeak decorates IPA with stress / length / syllable marks that are not
# phonemes; strip them before segmentation.
_STRIP = dict.fromkeys(map(ord, "ˈˌːˑ.‿|"), None)


def _with_closures(seq):
    """Insert a 'sil' (closure) before every stop, mirroring TIMIT segmentation."""
    out = []
    for p in seq:
        if p in STOPS:
            out.append("sil")
        out.append(p)
    return out


# Prefer espeak-ng (more languages incl. Hebrew, actively maintained) when
# installed; fall back to legacy espeak. Both accept the same -q --ipa -v flags.
_ESPEAK_BIN = shutil.which("espeak-ng") or shutil.which("espeak") or "espeak"


def _espeak_ipa(word, voice):
    try:
        out = subprocess.run(
            [_ESPEAK_BIN, "-q", "--ipa", "-v", voice, word],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception as exc:  # espeak missing / failed -> caller handles empty
        print(f"[word_g2p] {_ESPEAK_BIN} failed for {word!r}: {exc}")
        return ""
    return out.strip().translate(_STRIP)


# Word-level G2P backend: "espeak" (default, multilingual, MFA-independent) or
# "mfa" (the english_us_arpa Pynini G2P that Montreal Forced Aligner uses — for
# apples-to-apples English word-level comparison). Override via FDNFA_WORD_G2P.
G2P_BACKEND = os.environ.get("FDNFA_WORD_G2P", "espeak").lower()


def word_to_lh39(word, voice=None, backend=None):
    """Orthographic word -> list of LH39 phonemes (cached).

    Routes through the espeak or MFA-english_us_arpa G2P. `backend`/`voice`
    default to the FDNFA_WORD_G2P / FDNFA_G2P_VOICE env vars when not passed, so a
    long-running process (e.g. the Gradio app) can switch per call by argument.
    The MFA backend applies to English (it uses the english_us_arpa dictionary).
    """
    if backend is None:
        backend = os.environ.get("FDNFA_WORD_G2P", "espeak").lower()
    if voice is None:
        voice = os.environ.get("FDNFA_G2P_VOICE", DEFAULT_VOICE)
    if backend == "mfa":
        import mfa_g2p  # lazy: mfa_g2p imports closure helpers from this module
        return mfa_g2p.word_to_lh39_mfa(word, voice=voice)
    if backend == "char":
        return _char_word_lh39(word)
    return _espeak_word_lh39(word, voice)


def _char_word_lh39(word):
    """Word -> LH39 with NO G2P model: segment the (romanized) characters with
    panphon and map each directly to LH39 by articulatory-feature distance. Used
    for languages with no MFA model and where a grapheme-to-phoneme converter is
    deliberately avoided (e.g. Hebrew romanized transcripts)."""
    key = (word.lower(), "char")
    if key in _cache:
        return _cache[key]
    segs = _ft.ipa_segs(word.lower())
    lh39 = [dutch_preprocess.find_best_leehon39(s)[0] for s in segs if s.strip()]
    if not lh39:
        lh39 = ["sil"]
    if USE_CLOSURES:
        lh39 = _with_closures(lh39)
    _cache[key] = lh39
    return lh39


def _espeak_word_lh39(word, voice=DEFAULT_VOICE):
    """Word -> LH39 via espeak (cached). The espeak branch of word_to_lh39, also
    used by the MFA backend as the OOV fallback for languages with no MFA G2P."""
    key = (word.lower(), voice)
    if key in _cache:
        return _cache[key]
    ipa = _espeak_ipa(word, voice)
    segs = _ft.ipa_segs(ipa) if ipa else []
    lh39 = [dutch_preprocess.find_best_leehon39(s)[0] for s in segs if s.strip()]
    if not lh39:
        lh39 = ["sil"]
    if USE_CLOSURES:
        lh39 = _with_closures(lh39)
    _cache[key] = lh39
    return lh39
