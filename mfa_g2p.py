"""
MFA-compatible word -> Lee-Hon-39 phoneme front-end for *word-level* alignment.

This is the apples-to-apples counterpart of `word_g2p.py` (which uses espeak):
instead of espeak it phonemizes orthographic words with the **same** open-source
G2P models that Montreal Forced Aligner uses, then maps the result into LH39:

    word --MFA pronunciation dictionary lookup (highest-prob entry)--> phones
    word (OOV, English only) --english_us_arpa pynini G2P WFST--> ARPAbet
    English ARPAbet --strip stress, lowercase, timit_to_leehon_map_MACRO--> LH39
    Dutch/German IPA --panphon articulatory distance (dutch_preprocess)--> LH39

MFA at alignment time looks each word up in its dictionary first and only invokes
the G2P WFST for out-of-vocabulary words; this module mirrors that. Selecting this
backend (vs espeak) isolates the G2P front-end as the only thing that differs from
MFA, so a word-level FDNFA-vs-MFA comparison measures the *aligner*, not the
phonemizer.

Language is chosen with FDNFA_G2P_VOICE (the same env word_g2p uses):
  en-us -> english_us_arpa (ARPAbet, pynini OOV)   nl -> dutch_cv (IPA)
  de    -> german_mfa (IPA)                         he -> no MFA model (espeak fallback)
"""
import os
import subprocess

from utils import timit_to_leehon_map_MACRO, timit_leehon_39_phonemes

# Default MFA model locations (the standard `mfa model download` cache).
_MFA_ROOT = os.environ.get("MFA_ROOT_DIR", os.path.expanduser("~/Documents/MFA"))
_DICT_DIR = os.path.join(_MFA_ROOT, "pretrained_models", "dictionary")
_ARPA_G2P_DIR = os.path.join(_MFA_ROOT, "extracted_models", "g2p", "english_us_arpa_g2p")
ARPA_G2P_FST = os.environ.get("FDNFA_MFA_G2P_FST", os.path.join(_ARPA_G2P_DIR, "model.fst"))
ARPA_G2P_PHONES = os.environ.get("FDNFA_MFA_G2P_PHONES", os.path.join(_ARPA_G2P_DIR, "phones.sym"))
# conda env that has pynini installed (for OOV G2P on the WFST).
MFA_ENV_PY = os.environ.get("FDNFA_MFA_ENV_PY", os.path.expanduser("~/miniconda3/envs/aligner/bin/python"))

# voice -> (dictionary file, phone alphabet). FDNFA_MFA_DICT overrides the dict.
_LANG_CFG = {
    "en-us": ("english_us_arpa.dict", "arpa"),
    "en":    ("english_us_arpa.dict", "arpa"),
    "nl":    ("dutch_cv.dict",        "ipa"),
    "de":    ("german_mfa.dict",      "ipa"),
}
_VOICE = os.environ.get("FDNFA_G2P_VOICE", "en-us")
_DICT_FILE, _ALPHABET = _LANG_CFG.get(_VOICE, ("english_us_arpa.dict", "arpa"))
ARPA_DICT = os.environ.get("FDNFA_MFA_DICT", os.path.join(_DICT_DIR, _DICT_FILE))

# Reuse the exact closure-insertion rule from the espeak front-end so the only
# thing differing between the espeak and MFA word backends is the G2P.
from word_g2p import USE_CLOSURES, _with_closures

_dict = None          # {word_lower: [phones]}  (first/highest-prob entry)
_cache = {}           # word_lower -> [lh39, ...]
_oov = set()          # words not found in the dictionary (for reporting)


def _load_dict():
    """Parse the selected MFA dictionary once. Format: word <tab> [prob cols <tab>]
    PHONES, where PHONES (final tab-separated field) is space-separated phones and
    the first float column (when present) is the pronunciation probability.

    Like MFA, keep the **highest-probability** pronunciation per word (MFA's most-
    likely variant; it then disambiguates acoustically, which we cannot). Entries
    with no probability column are treated as probability 1.0."""
    global _dict
    if _dict is not None:
        return _dict
    best = {}   # word -> (prob, phones)
    with open(ARPA_DICT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            phones = parts[-1].split()
            try:
                prob = float(parts[1]) if len(parts) >= 3 else 1.0
            except ValueError:
                prob = 1.0
            if word and (word not in best or prob > best[word][0]):
                best[word] = (prob, phones)
    _dict = {w: ph for w, (_, ph) in best.items()}
    return _dict


def arpa_to_lh39(phones):
    """ARPAbet (with stress digits) -> LH39, dropping non-phone tokens (spn/sil)."""
    out = []
    for p in phones:
        base = p.rstrip("0123456789").lower()   # AH0 -> ah, B -> b
        if base in ("spn", "sil", "sp", ""):
            continue
        if base in timit_leehon_39_phonemes:
            out.append(base)
        else:
            out.append(timit_to_leehon_map_MACRO.get(base, "sil"))
    return out


def ipa_to_lh39(phones):
    """IPA phones (dutch_cv / german_mfa dicts) -> LH39 via panphon distance — the
    same articulatory mapping the espeak/phoneme paths use (dutch_preprocess)."""
    import dutch_preprocess
    out = []
    for p in phones:
        if p in ("spn", "sil", "sp", ""):
            continue
        out.append(dutch_preprocess.find_best_leehon39(p)[0])
    return out


def _g2p_oov(word):
    """Phonemize an OOV English word with the english_us_arpa pynini WFST (run in
    the mfa env, which has pynini). Mirrors MFA's own G2P: compose the word
    acceptor with the pair-n-gram model and take the shortest path, decoded via
    phones.sym. Returns a list of ARPAbet phones, or [] if unavailable."""
    if not (os.path.exists(MFA_ENV_PY) and os.path.exists(ARPA_G2P_FST)
            and os.path.exists(ARPA_G2P_PHONES)):
        return []
    code = (
        "import sys,pynini\n"
        f"fst=pynini.Fst.read({ARPA_G2P_FST!r})\n"
        f"ps=pynini.SymbolTable.read_text({ARPA_G2P_PHONES!r})\n"
        "fst.set_output_symbols(ps)\n"
        "w=sys.argv[1].lower()\n"
        "try:\n"
        "    lat=pynini.compose(pynini.accep(w, token_type='utf8'), fst)\n"
        "    print(pynini.shortestpath(lat).string(ps))\n"
        "except Exception:\n"
        "    print('')\n"
    )
    try:
        out = subprocess.run([MFA_ENV_PY, "-c", code, word],
                             capture_output=True, text=True, timeout=30).stdout
        return out.strip().split()
    except Exception:
        return []


def word_to_lh39_mfa(word):
    """Orthographic word -> list of LH39 phonemes via the selected MFA G2P."""
    key = word.lower()
    if key in _cache:
        return _cache[key]
    d = _load_dict()
    phones = d.get(key)
    if phones is None:
        _oov.add(key)
        # Pure MFA, no espeak. English OOV -> MFA's pynini WFST. German OOV is
        # pre-resolved into the merged dictionary via MFA's own G2P. Any word
        # still unresolved (e.g. Dutch, which has no MFA G2P model) becomes a
        # single 'sil'/spn, exactly as MFA treats an unknown word.
        if _ALPHABET == "arpa":
            phones = _g2p_oov(key)
    if _ALPHABET == "arpa":
        lh39 = arpa_to_lh39(phones) if phones else []
    else:
        lh39 = ipa_to_lh39(phones) if phones else []
    if not lh39:
        lh39 = ["sil"]
    if USE_CLOSURES:
        lh39 = _with_closures(lh39)
    _cache[key] = lh39
    return lh39


def oov_words():
    return set(_oov)
