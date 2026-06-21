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


def _espeak_ipa(word, voice):
    try:
        out = subprocess.run(
            ["espeak", "-q", "--ipa", "-v", voice, word],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception as exc:  # espeak missing / failed -> caller handles empty
        print(f"[word_g2p] espeak failed for {word!r}: {exc}")
        return ""
    return out.strip().translate(_STRIP)


def word_to_lh39(word, voice=DEFAULT_VOICE):
    """Orthographic word -> list of LH39 phonemes (cached)."""
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
