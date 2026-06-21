import re
import panphon
import panphon.distance

ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

# IFA_TO_IPA = TODO - FUNCTION THAT COMPUTES SCORE
# NOT MAPPING BUT NEEDS TO BE A PUNCTION

# IPA_TO_LEEHON39 = TODO - NOT MAPPING NEEDS TO BE A FUNCTION BUT ONE TIME - MAYBE IT IS MAPPING?

IFA_TO_IPA = {
    "p":"p", "b":"b", "t":"t", "d":"d", "k":"k", "g":"ɡ",
    "f":"f", "v":"v", "s":"s", "z":"z", "h":"h", "x":"x", "G":"ɣ",
    "m":"m", "n":"n", "N":"ŋ", "l":"l", "r":"r", "w":"ʋ", "j":"j",
    "S":"ʃ", "Z":"ʒ", "J":"ɲ", "L":"ʎ",
    "i":"i", "I":"ɪ", "e":"eː", "E":"ɛ", "a":"aː", "A":"ɑ",
    "o":"oː", "O":"ɔ", "u":"u", "y":"y", "Y":"ʏ", "2":"øː",
    "9":"œ", "@":"ə", "!" : "ɛi", "V" : "ʌu", "W" : "œy", "h#" : "h#"
}

LH39_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u", "B": "b", "CH": "tʃ", "D": "d",
    "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ"
}

timit_leehon_39_phonemes = [
    'ao', 'ae', 'ah','aw', 'er', 'ay', 
    'b', 'sil', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'm', 'en', 'ng', 'ey',
    'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'v', 'w', 'y', 'z', 'sh', 't', 'r', 's', 'th','uh', 'uw', 'oy', 'ow','p'
]

def get_ipa_from_ifa(ifa_label):
    
    if ifa_label.lower() in timit_leehon_39_phonemes:
        return [ifa_label.lower()]
    if ifa_label in ['h#', 'tcl']:
    # if ifa_label in ['h#']:
        return ["sil"]
    # if ifa_label in ['tcl']:
    #     return []
    
    # Convert underscores and hyphens to spaces, and remove colons (length is handled by the base vowel mapping or discarded)
    cleaned = ifa_label.replace(':', '').replace('_', ' ').replace('-', ' ') #.replace('tcl', ' ')
    # Remove Stress ("), Secondary Stress ('), Syllable dots (.), and nasal tildes (~)
    cleaned = re.sub(r'[".\'~]', '', cleaned)
    # parts = cleaned.split()   
    # parts = cleaned.strip()   
    parts = cleaned.strip().split()
    if not parts:
        return [] 
    if len(parts) == 1 and len(ifa_label) >1 and ifa_label not in IFA_TO_IPA: #it's a long phoneme label that needs to be splitted to several IPA symbols
        parts = list(ifa_label)
    
    ipa_list = [IFA_TO_IPA.get(p,p) for p in parts if p.strip()]
    return ipa_list

_leehon39_cache = {}

def find_best_leehon39(target_ipa):
    # Deterministic IPA->LH39 mapping; memoize since the panphon feature-edit
    # distance over a fixed inventory recomputes the same answer per occurrence.
    if target_ipa in _leehon39_cache:
        return _leehon39_cache[target_ipa]
    result = _find_best_leehon39(target_ipa)
    _leehon39_cache[target_ipa] = result
    return result

def _find_best_leehon39(target_ipa):

    if not target_ipa or target_ipa.strip() == "":
        return "sil", 0.0
    
    if target_ipa.lower() in timit_leehon_39_phonemes:
        return target_ipa.lower(), 0.0
    
    # if target_ipa.lower() in ['h#', 'tcl', 'sil']:
    if target_ipa.lower() in ['h#', 'sil']:
        return "sil", 0.0
    if target_ipa.lower() in ["r", "ɾ"]:
        return "r", 0.0
    
    best_label = "sil"
    min_dist = 100.0
    
    for lh_label, lh_ipa in LH39_IPA.items():
        
        d = dst.feature_edit_distance(target_ipa, lh_ipa)
        if d< min_dist:
            min_dist = d
            best_label = lh_label.lower()
    return best_label, round(min_dist,3)

def aligner_pipeline(ifa_input):
    ifa_segments = get_ipa_from_ifa(ifa_input)
    results = []
    
    for ipa_seg in ifa_segments:
        match, d = find_best_leehon39(ipa_seg)
        results.append( {"ifa_ipa_part" :ipa_seg, "lh39" :match, "dist" :d} )
    return results


import os

# def convert_all_lab_files(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".lab"):
#             path = os.path.join(directory, filename)
#             with open(path, 'r') as f:
#                 content = f.read().strip()
            
#             # Use your existing pipeline logic
#             # Note: We split the content by space to process each phone
#             ifa_phones = content.split()
#             ipa_output = []
#             for p in ifa_phones:
#                 # Get the IPA parts from your existing function
#                 ipa_parts = get_ipa_from_ifa(p)
#                 ipa_output.extend(ipa_parts)
            
#             # Join with spaces and write back
#             new_content = " ".join(ipa_output)
#             with open(path, 'w') as f:
#                 f.write(new_content)
#     print(f"Done! All .lab files in {directory} converted to IPA.")

# # Run this in your main block
# # convert_all_lab_files('/home/rotem/projects/datasets/IFA_dutch_split/test')

# # convert_all_lab_files('/home/rotem/projects/datasets/IFA_dutch_split/test')



import os

def create_lab_files(phn_folder, lab_folder):
    if not os.path.exists(lab_folder):
        os.makedirs(lab_folder)

    for filename in os.listdir(phn_folder):
        if filename.endswith(".phn"):
            with open(os.path.join(phn_folder, filename), 'r') as f:
                lines = f.readlines()
            
            ipa_sequence = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                label = parts[2]
                # Use your existing mapping function
                ipa_symbols = get_ipa_from_ifa(label) 
                
                # Filter out 'sil' if you want MFA to handle silence automatically, 
                # but usually keeping them is fine for phone-level alignment.
                ipa_sequence.extend(ipa_symbols)
            
            # Save to .lab file (space separated string)
            lab_filename = filename.replace(".phn", ".lab")
            with open(os.path.join(lab_folder, lab_filename), 'w') as f:
                f.write(" ".join(ipa_sequence))
                
def generate_ipa_lexicon(all_ipa_symbols, output_path):
    with open(output_path, 'w') as f:
        # Add a silence mapping just in case
        f.write("sil\tsil\n")
        # Map every unique IPA symbol to itself
        for symbol in sorted(list(set(all_ipa_symbols))):
            if symbol != "sil":
                f.write(f"{symbol}\t{symbol}\n")

# Run it
# create_lab_files("/home/rotem/projects/datasets/IFA_dutch_split/test", "/home/rotem/projects/datasets/IFA_dutch_split/test")

if __name__ == "__main__":
    test_cases = ["sil n Y l sil e: n sil t w e: sil d r i sil v i r sil v Ei f sil z E s sil z e: v @ n sil A x t sil n e: x @ sil t i n sil E l f sil t w a: l f sil n Y l sil sil"]
    # test_cases = ["x@l", "@-r-h-a", "e:-j", "r9y", "ao", "sil", "@", "E", "he:l-@_hAr", "t_b", "o:", "N"]
    for case in test_cases:
        print(f"\nINPUT: {case}")
        output = aligner_pipeline(case)
        # [x["lh39"] for x in output]
        if not output:
            print("Results: None")
        else:
            for item in output:
                print(f" Mapped '{item['ifa_ipa_part']}' -> {item['lh39']} (dist_score: {item['dist']})")
    # convert_all_lab_files('/home/rotem/projects/datasets/IFA_dutch_split/test')
    # Run it
    create_lab_files("/home/rotem/projects/datasets/IFA_dutch_split/test", "/home/rotem/projects/datasets/IFA_dutch_split/test")
                
