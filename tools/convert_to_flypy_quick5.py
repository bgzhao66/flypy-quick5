# -*- coding: utf-8 -*-
import sys
import os
import re
import argparse
import copy
import unicodedata

import unittest

"""
Convert Pinyin with diacritics → Toneless Pinyin → Shuangpin (Xiaohe scheme),
with unit tests.
"""

TONELESS_MAPPING = """
ā a
á a
ǎ a
à a
ē e
é e
ě e
è e
ê ai
ê̄ ai
ế ai
ê̌ ai
ề ai
ḿ m
m̀ m
ń n
ň n
ǹ n
ō o
ó o
ǒ o
ò o
ī i
í i
ǐ i
ì i
ū u
ú u
ǔ u
ù u
ü v
ǖ v
ǘ v
ǚ v
ǜ v
"""

def get_toneless_mapping():
    mapping = dict()
    for line in TONELESS_MAPPING.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        pinyin, toneless = line.split()
        mapping[pinyin] = toneless
    return mapping

kTonelessMapping = get_toneless_mapping()

# get the toneless pinyin from pinyin
def get_toneless_pinyin(pinyin):
    toneless = ''
    for c in pinyin:
        if c in kTonelessMapping:
            toneless += kTonelessMapping[c]
        elif unicodedata.category(c).startswith('L'):
            # If the character is a letter, keep it as is
            toneless += c
    return toneless

# get the toneless pinyin seq from pinyin seq
def get_toneless_pinyin_seq(pinyin_seq):
    return [get_toneless_pinyin(pinyin) for pinyin in pinyin_seq]

# Step 2: Xiaohe Shuangpin mapping (initials + finals)
INITIALS = {
    "b": "b", "p": "p", "m": "m", "f": "f",
    "d": "d", "t": "t", "n": "n", "l": "l",
    "g": "g", "k": "k", "h": "h",
    "j": "j", "q": "q", "x": "x",
    "zh": "v", "ch": "i", "sh": "u", "r": "r",
    "z": "z", "c": "c", "s": "s",
    "y": "y", "w": "w", "": ""
}

FINALS = {
    "iu": "q", "ei": "w", "uan":"r", "van" : "r",  "ue": "t", "ve": "t",  
    "un": "y", "vn": "y", "uo": "o", "o": "o", "ie": "p",

    "a": "a", "ong": "s", "iong": "s", "ai": "d", "en":"f", "eng": "g",
    "ang": "h", "an": "j", "uai":"k", "ing":"k", "uang": "l", "iang": "l",
    
    "ou": "z", "ua": "x", "ia": "x", "ao": "c", "ui": "v", "v":"v",
    "in": "b", "iao": "n", "ian": "m",

    "a": "a", "e": "e", "i": "i", "u": "u",
}


def pinyin_to_shuangpin(toneless: str):
    """Convert toneless Pinyin syllable to Shuangpin (Xiaohe)."""
    toneless.strip()
    if not toneless:
        raise ValueError("Input Pinyin cannot be empty.")
    special_cases = {"a": "aa", "o": "oo", "e": "ee", "ang": "ah", "eng": "eg",
                     "ng": "eg", "m": "mm", "n": "nn", "hng": "hg"}
    if toneless in special_cases:
        return special_cases[toneless]
    
    if len(toneless) == 2:
        return toneless  # Handle two-letter finals directly

    # split initial and final with RMM(right maximum match) algorithm
    i = len(toneless)
    pos = i
    while i > 0:
        i -= 1
        if toneless[i:] in FINALS:
            pos = i
    final = toneless[pos:]
    initial = toneless[:pos]
    assert initial in INITIALS, f"Unknown initial: {initial}, {toneless}"
    assert final in FINALS, f"Unknown final: {final}, {initial}, {toneless}"
    return INITIALS[initial] + FINALS[final]

# Step 3: Get standard Chinese characters and Pinyin mappings

STANDARD_CHINESE = "standard_chinese.txt"
PINYIN_CODE = "pinyin.txt"
PINYIN_PHRASE = "pinyin_phrase.txt"

# get chinese code from a file with format "pinyin: word1 word2 ..."
def get_standard_code_from_file(file):
    words = dict()
    # read the file line by line
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            #skip line beginning with '#' or empty line
            if len(line) == 0 or line[0] == '#':
                continue
            # split the line into words by spaces or commas or colons
            chars = re.split(r'[\s,:]', line)
            if len(chars) <= 1:
                continue
            pinyin = chars[0]
            for char in chars[1:]:
                if len(char.strip()) == 0:
                    continue
                if char not in words:
                    words[char] = []
                words[char].append(pinyin)
    return words

# get pinyin code from a file with format "UNICODE: py1,py2 # word"
def get_pinyin_code_from_file(file):
    words = dict()
    with open(file  , 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            parts = re.split(r'\s+', line)
            if len(parts) < 4:
                continue
            word = parts[3].strip()
            pinyins = re.split(r',', parts[1])
            if word not in words:
                words[word] = []
            for pinyin in pinyins:
                words[word].append(pinyin)
    return words

kPinyinCodes = get_pinyin_code_from_file(PINYIN_CODE)
kStandardCodes = get_standard_code_from_file(STANDARD_CHINESE)

# Merge kStandardCodes and kPinyinCodes, then return the new dictionary
def merge_character_codes():
    codes = dict()
    for word in kStandardCodes:
        codes[word] = copy.deepcopy(kStandardCodes[word])
    for word in kPinyinCodes:
        if word not in codes:
            codes[word] = copy.deepcopy(kPinyinCodes[word])
    return codes

kCharacterCodes = merge_character_codes()

# Get pinyin phrase from a file with format "word: py1 py2 ..."
# return a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}
def get_pinyin_phrase_from_file(file):
    words = dict()
    with open(file , 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            parts = re.split(r':\s+', line)
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            pinyins = re.split(r'\s+', parts[1])
            if word not in words:
                words[word] = []
            words[word].append(pinyins)
    return words

# get pinyin phrases
def get_pinyin_phrases():
    return get_pinyin_phrase_from_file(PINYIN_PHRASE)

# Check if a word and its pinyin code sequence are consistent
# word: a Chinese word
# word_code: a list of pinyin codes for the word, e.g. ['py1', 'py2']

def is_consistent(word, word_code, strict=True):
    if len(word) != len(word_code):
        return False
    for i in range(len(word)):
        if word[i] not in kCharacterCodes:
            return False
        if strict and word_code[i] not in kCharacterCodes[word[i]]:
            return False
        if not strict:
            toneless = get_toneless_pinyin(word_code[i])
            if toneless not in get_toneless_pinyin_seq(kCharacterCodes[word[i]]):
                return False
            return True
    return True

# Purge inconsistent phrases
# words: a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}
# strict: if True, check the pinyin with tones; if False, check the toneless pinyin
def purge_inconsistent_phrases(words, strict=True):
    res = dict()
    for word in words.keys():
        for pinyin_seq in words[word]:
            if is_consistent(word, pinyin_seq, strict = strict):
                if word not in res:
                    res[word] = []
                res[word].append(pinyin_seq)
    return res

kPinyinPhrases = purge_inconsistent_phrases(get_pinyin_phrases(), strict=False)

# Step 4: Get frequency-sorted simplified Chinese dictionary

PINYIN_DICT = "pinyin_trad.dict.txt"
PINYIN_EXT1_DICT = "pinyin_trad_ext1.dict.txt"

# get the frequency of words from a file
def get_frequency_from_files(files):
    freq = dict()
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                word, code, frequency = line.split('\t')
                if word not in freq:
                    freq[word] = dict()
                if code not in freq[word]:
                    freq[word][code] = 0
                freq[word][code] += int(frequency)
    return freq

kWordsFreq = get_frequency_from_files([PINYIN_DICT, PINYIN_EXT1_DICT])

# get the frequency of a character from freq_dict with default value 0.
def get_freq_of_word(word, toneless_code, freq_dict):
    if word not in freq_dict:
        return 0
    if toneless_code not in freq_dict[word]:
        return 0
    return freq_dict[word][toneless_code]

# Step 5: Get Cangjie codes from file

CANGJIE_CODE = "cangjie5.dict.yaml"

# get cangjie code from a file with format "word\tcjcode1 extra_info"
def get_cangjie_code_from_file(file):
    words = dict()
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            parts = re.split(r'\t', line)
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            cjcode = parts[1].strip().split()[0]
            if word not in words:
                words[word] = []
            words[word].append(cjcode)
    return words

kCangjieCodes = get_cangjie_code_from_file(CANGJIE_CODE)

# Step 6: Main function to convert toneless pinyin to shuangpin

def pinyin_to_shuangpin_seq(toneless_seq):
    """Convert a sequence of toneless Pinyin to Shuangpin (Xiaohe scheme)."""
    shuangpin_seq = []
    for toneless in toneless_seq:
        shuangpin = pinyin_to_shuangpin(toneless)
        shuangpin_seq.append(shuangpin)
    return shuangpin_seq

# Return a list of FlypyQuick5 sequences for a word and its Pinyin sequence.
# This function converts the Pinyin sequence to toneless Pinyin,
# then maps each toneless Pinyin to its Flypys, and then to its corresponding FlypyQuick5 code.
# The mapping is based on the Xiaohe Shuangpin scheme.

# The FlypyQuick5 sequences are generated based on the following rules:
# Notations: flypys is a list of Flypys for the toneless Pinyin,
#          cjcode is the Cangjie code for the last Chinese character in word.
# Rules:
# 1. Single characters: "".join(flypys) + cjcode[1] + cjcode[-1]
# 2. Two characters: "".join(flypys) + cjcode[-1]
# 3. Three or four characters: "".join(flypys[:4])
# 4. Five or more characters:  "".join(flypys[:4]) + cjcode[-1]
# 5. If the word is not in the frequency dictionary, use a default frequency of

# Returns a list of FlypyQuick5 sequences for the given word and Pinyin sequence.
#  [("flypyquick5_seq1", freq1), ("flypyquick5_seq2", freq2), ...]

def get_flypyquick5_seq(word, pinyin_seq):
    """Convert a word and its Pinyin sequence to FlypyQuick5 sequences."""
    toneless_seq = get_toneless_pinyin_seq(pinyin_seq)
    flypys = pinyin_to_shuangpin_seq(toneless_seq)
    char = word[-1]
    if char not in kCangjieCodes:
        raise ValueError(f"Character '{char}' not found in Cangjie codes.")
    cjcodes = kCangjieCodes[char]
    if len(cjcodes) == 0:
        raise ValueError(f"No Cangjie codes found for character '{char}'.")
    freq = get_freq_of_word(word, ' '.join(toneless_seq), kWordsFreq)
    flypyquick5_seq = []
    for cjcode in cjcodes:
        if len(word) == 1:
            # Single character: use cjcode[1] and cjcode[-1]
            flypyquick5_seq.append(("".join(flypys) + cjcode[0] + cjcode[-1], freq))
        elif len(word) == 2:
            # Two characters: use cjcode[-1]
            flypyquick5_seq.append(("".join(flypys) + cjcode[-1], freq))
        elif 3 <= len(word) <= 4:
            # Three or four characters: use first three Flypys
            flypyquick5_seq.append(("".join(flypys), freq))
        else: # len(word) >= 5
            # Five characters: use first four Flypys and cjcode[-1]
            flypyquick5_seq.append(("".join(flypys[:4]) + cjcode[-1], freq))
    if len(flypyquick5_seq) == 0:
        raise ValueError(f"No valid FlypyQuick5 sequences generated for word '{word}'.")
    # Return the list of FlypyQuick5 sequences
    return flypyquick5_seq

# Get the FlypyQuick5 dictionary from a dictionary of words and their Pinyin sequences.
# words: a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}
# return a dictionary of word and a list of FlypyQuick5 sequences, e.g. {'word': [("flypyquick5_seq1", freq1), ("flypyquick5_seq2", freq2), ...]}   

def get_flypyquick5_dict(words):
    flypyquick5_dict = dict()
    for word in words.keys():
        for pinyin_seq in words[word]:
            try:
                flypyquick5_seq = get_flypyquick5_seq(word, pinyin_seq)
                if word not in flypyquick5_dict:
                    flypyquick5_dict[word] = []
                for seq, freq in flypyquick5_seq:
                    flypyquick5_dict[word].append((seq, freq))
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
    return flypyquick5_dict

# get all descartes products of encodes which is a list of list of elements
# e.g. [[a1, a2], [b1, b2]] -> [[a1, b1], [a1, b2], [a2, b1], [a2, b2]]
def get_descartes_products(encodes):
    descartes = [[]]
    for encode in encodes:
        new_descartes = []
        for descarte in descartes:
            for element in encode:
                new_descartes.append(descarte + [element])
        descartes = new_descartes
    return descartes

# Get pinyin sequences for words from a dictionary of words and their pinyin code sequences. If a word is not in the dictionary, return a descartes product of its characters' pinyin codes.
# words is a list of words, e.g. ['word1', 'word2', ...]
# return a dictionary of word and a list of pinyin code sequences, e.g. {'word': [['py1', 'py2'], ['py3', 'py4']]}

def get_pinyin_seq_for_words(words):
    pinyin_seq_dict = dict()
    for word in words:
        if word in kPinyinPhrases:
            pinyin_seq_dict[word] = kPinyinPhrases[word]
        else:
            encodes = []
            for char in word:
                if char in kCharacterCodes:
                    encodes.append(kCharacterCodes[char])
                else:
                    encodes.append([''])
            pinyin_seq_dict[word] = get_descartes_products(encodes)
    return pinyin_seq_dict

# Get a list of words from an input file, one word per line.
def get_words_from_file(file):
    words = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            words.append(line)
    return words

# Get the difference set of phrases against the builtin Pinyin phrases.
def get_difference_set(phrase_list):
    """Get the difference set of phrases against the builtin Pinyin phrases."""
    diff_list = []
    for word in phrase_list:
        if word not in kPinyinPhrases:
            diff_list.append(word)
    return diff_list

# Step 7: Header for Rime dictionary
def get_header(name, input_tables):
    hdr = f"""# rime dictionary
# encoding: utf-8

---
name: {name}
version: "0.1"
sort: by_weight"""

    if input_tables:
        hdr += """
max_phrase_length: 10
min_phrase_weight: 100
encoder:
  rules:
    - length_equal: 2
      formula: "AaAbBaBbBzAc"
    - length_equal: 3
      formula: "AaAbBaBbCaCb"
    - length_equal: 4
      formula: "AaAbBaBbCaCbDaDb"
    - length_equal: 5
      formula: "AaAbBaBbCaCbDaDbEz"
    - length_equal: 6
      formula: "AaAbBaBbCaCbDaDbFz"
    - length_equal: 7
      formula: "AaAbBaBbCaCbDaDbGz"
    - length_equal: 8
      formula: "AaAbBaBbCaCbDaDbHz"
    - length_equal: 9
      formula: "AaAbBaBbCaCbDaDbIz"
import_tables:"""
        for table in input_tables:
            hdr += f"\n  - {table}"
    hdr += "\n...\n"
    return hdr

# get the sorted keys of a dictionary
def get_sorted_keys(dict):
    keys = list(dict.keys())
    keys.sort()
    return keys

# Convert a dictionary of word codes to a nested list format.
# word_codes: a dictionary of word and a list of tonal pinyin code sequences,
#               e.g. {'word': ["py1", "py2", ...]}]} 
# return a nested list 
#        e.g. {"word": [["py1"], ["py2"], ...]}
def convert_to_nested_dict(word_codes):
    nested_dict = dict()
    for word in word_codes:
        if word not in nested_dict:
            nested_dict[word] = []
        for code in word_codes[word]:
            nested_dict[word].append([code])
    return nested_dict

# print the word_codes which is a dictionary of key,list into a file with the format of word code frequency
# word_codes: a dictionary of word and a list of tonal pinyin code sequences,
#               e.g. {'word': [("py1 py2", freq),  ("py3 py4", freq), ...]} 
# return a nested dictionary of length, code, word and frequency
#        e.g. {"length": {"code": {"word": frequency}}}

def sort_by_length_and_code(word_codes):
    sorted_word_codes = dict()
    for word in word_codes:
        length = len(word)
        if length not in sorted_word_codes:
            sorted_word_codes[length] = dict()
        for code, freq in word_codes[word]:
            if code not in sorted_word_codes[length]:
                sorted_word_codes[length][code] = dict()
            if word not in sorted_word_codes[length][code]:
                sorted_word_codes[length][code][word] = 0
            sorted_word_codes[length][code][word] += freq
    return sorted_word_codes

# print the word_codes which is a dictionary of key,list into a file with the format of word code frequency
# word_codes: {"legth": {"code": {"word": frequency}}}
# outfile: the output file, default is sys.stdout, in the following format:
# word<tab>code<tab>frequency
def print_word_codes(word_codes, outfile=sys.stdout):
    for length in get_sorted_keys(word_codes):
        for code in get_sorted_keys(word_codes[length]):
            for word in get_sorted_keys(word_codes[length][code]):
                freq = word_codes[length][code][word]
                print("%s\t%s\t%i" % (word, code, freq), file=outfile)

# Get the sorted FlypyQuick5 dictionary from a list of words.
# words: a list of words, {"word": [["py1", "py2"], ["py3", "py4"]]}
# return a nested dictionary of length, code, word and frequency
def get_sorted_flypyquick5_dict(words):
    words_dict = get_flypyquick5_dict(words)
    sorted_dict = sort_by_length_and_code(words_dict)
    return sorted_dict

# Augment the two-character words when there are conflicts by appending the first character's Cangjie code to the FlypyQuick5 code.
# which are not most frequent ones.
def augment_two_character_words(word_codes, primary_dict = dict()):
    length = 2
    if length not in word_codes:
        return word_codes

    builtin_dict = dict()
    if length in primary_dict:
        builtin_dict = primary_dict[length]

    words_to_remove = dict()
    for code in list(word_codes[length].keys()):
        # find the most frequent word
        max_freq = -1
        max_word = None
        not_in_builtin = code not in builtin_dict
        if not_in_builtin:
            if len(word_codes[length][code]) <= 1:
                continue
            for word in word_codes[length][code]:
                freq = word_codes[length][code][word]
                if freq > max_freq:
                    max_freq = freq
                    max_word = word
        # augment other words
        for word in word_codes[length][code]:
            if word == max_word:
                continue
            char = word[0]
            if char in kCangjieCodes and len(kCangjieCodes[char]) > 0:
                for cjcode in kCangjieCodes[char]:
                    new_code = code + cjcode[0]
                    freq = word_codes[length][code][word]
                    if new_code not in word_codes[length]:
                        word_codes[length][new_code] = dict()
                    if word not in word_codes[length][new_code]:
                        word_codes[length][new_code][word] = 0
                    word_codes[length][new_code][word] += freq
        # remove the old code entry except the most frequent one
        words_to_remove[code] = [word for word in word_codes[length][code] if word != max_word]

    # remove the old code entries
    for code in words_to_remove:
        for word in words_to_remove[code]:
            del word_codes[length][code][word]
        if len(word_codes[length][code]) == 0:
            del word_codes[length][code]
    return word_codes

# process a list of words and print the FlypyQuick5 dictionary to a file
# words: a list of words
# outfile: the output file, default is sys.stdout
def process_and_print_flypyquick5_dict(words, outfile=sys.stdout, primary_dict = dict()):
    sorted_dict = get_sorted_flypyquick5_dict(words)
    augmented_dict = augment_two_character_words(sorted_dict, primary_dict)
    print_word_codes(augmented_dict, outfile)

# ---------------------- Unit Tests ----------------------
class TestShuangpin(unittest.TestCase):
    def test_basic_cases(self):
        self.assertEqual(pinyin_to_shuangpin("ao"), "ao")  
        self.assertEqual(pinyin_to_shuangpin("fei"), "fw")
        self.assertEqual(pinyin_to_shuangpin("ei"), "ei")
        self.assertEqual(pinyin_to_shuangpin("ang"), "ah")
        self.assertEqual(pinyin_to_shuangpin("eng"), "eg")

    def test_with_initials(self):
        self.assertEqual(pinyin_to_shuangpin("mao"), "mc")
        self.assertEqual(pinyin_to_shuangpin("zhong"), "vs")
        self.assertEqual(pinyin_to_shuangpin("xue"), "xt")
        self.assertEqual(pinyin_to_shuangpin("yu"), "yu")

    def test_pinyin_to_shuangpin(self):
        self.assertEqual(pinyin_to_shuangpin("a"), "aa")
        self.assertEqual(pinyin_to_shuangpin("o"), "oo")
        self.assertEqual(pinyin_to_shuangpin("e"), "ee")

        self.assertEqual(pinyin_to_shuangpin("n"), "nn")
        self.assertEqual(pinyin_to_shuangpin("m"), "mm")
        self.assertEqual(pinyin_to_shuangpin("hng"), "hg")
        self.assertEqual(pinyin_to_shuangpin("ng"), "eg")

    def test_toneless_conversion(self):
        self.assertEqual(get_toneless_pinyin("mǎ"), "ma")
        self.assertEqual(get_toneless_pinyin("nǐ"), "ni")
        self.assertEqual(get_toneless_pinyin("hǎo"), "hao")
        self.assertEqual(get_toneless_pinyin("lǜ"), "lv")
        self.assertEqual(get_toneless_pinyin("aī"), "ai")

    def test_toneless_sequence(self):
        pinyin_seq = ["mǎ", "nǐ", "hǎo", "lǜ"]
        toneless_seq = get_toneless_pinyin_seq(pinyin_seq)
        self.assertEqual(toneless_seq, ["ma", "ni", "hao", "lv"])

    def test_flypyquick5_seq(self):
        testcases = [("你好", ["nǐ", "hǎo"], "nihcd"),
                     ("长臂猿", ["cháng", "bì", "yuán"], "ihbiyr"),
                     ("世界地圖", ["shì", "jiè", "dì", "tú"], "uijpditu"),
                     ("中華人民共和國", ["zhōng", "huá", "rén", "mín", "gòng", "hé", "guó"], "vshxrfmbm")]
        for word, pinyin_seq, expected_seq in testcases:
            flypyquick5_seq = get_flypyquick5_seq(word, pinyin_seq)
            self.assertTrue(len(flypyquick5_seq) > 0)
            for seq, freq in flypyquick5_seq:
                self.assertTrue(isinstance(seq, str))
                self.assertTrue(isinstance(freq, int))
                self.assertEqual(seq, expected_seq)

    def test_flypyquick5_dict(self):
        words = {"你好": [["nǐ", "hǎo"]], "世界": [["shì", "jiè"]]}
        flypyquick5_dict = get_flypyquick5_dict(words)
        self.assertTrue("你好" in flypyquick5_dict)
        self.assertTrue("世界" in flypyquick5_dict)
        self.assertTrue(len(flypyquick5_dict["你好"]) > 0)
        self.assertTrue(len(flypyquick5_dict["世界"]) > 0)

    def test_pinyin_phrase(self):
        pinyin_phrases = get_pinyin_phrases()
        self.assertTrue("你好" in pinyin_phrases)
        self.assertTrue(len(pinyin_phrases["你好"]) > 0)
        for phrase in pinyin_phrases["你好"]:
            self.assertTrue(isinstance(phrase, list))
            self.assertTrue(len(phrase) > 0)

    def test_is_consistent(self):
        self.assertTrue(is_consistent("你好", ["ni", "hao"], strict=False))
        self.assertFalse(is_consistent("你好", ["ni", "hao", "shi"]))
        self.assertTrue(is_consistent("世界", ["shi", "jie"], strict=False))
        self.assertFalse(is_consistent("世界", ["shi", "jie"], strict=True))

    def test_purge_inconsistent_phrases(self):
        phrases = {
            "你好": [["ni", "hao"], ["ni", "hao", "shi"]],
            "世界": [["shi", "jie"]]
        }
        purged_phrases = purge_inconsistent_phrases(phrases, strict=False)
        self.assertTrue("你好" in purged_phrases)
        self.assertTrue("世界" in purged_phrases)
        self.assertEqual(len(purged_phrases["你好"]), 1)
        self.assertEqual(len(purged_phrases["世界"]), 1)

    def test_kCharacterCodes(self):
        self.assertTrue("你" in kCharacterCodes)
        self.assertTrue("好" in kCharacterCodes)
        self.assertTrue(len(kCharacterCodes["你"]) > 0)
        self.assertTrue(len(kCharacterCodes["好"]) > 0)
        self.assertTrue(isinstance(kCharacterCodes["你"][0], str))
        self.assertTrue(isinstance(kCharacterCodes["好"][0], str))

    def test_kPinyinPhrases(self):
        self.assertTrue("你好" in kPinyinPhrases)
        self.assertTrue(len(kPinyinPhrases["你好"]) > 0)
        for phrase in kPinyinPhrases["你好"]:
            self.assertTrue(isinstance(phrase, list))
            self.assertTrue(len(phrase) > 0)
    
    def test_convert_to_nested_dict(self):
        word_codes = {
            "好": ["hao3", "hao4"]
        }
        nested_dict = convert_to_nested_dict(word_codes)
        self.assertTrue("好" in nested_dict)
        self.assertTrue(len(nested_dict["好"]) > 0)
        self.assertTrue(isinstance(nested_dict["好"][0], list))
        self.assertEqual(nested_dict["好"][0][0], "hao3")

    def test_difference_set(self):
        phrase_list = ["你好", "世界", "再见", "應該沒有這個詞"]
        diff_list = get_difference_set(phrase_list)
        self.assertTrue("再见" in diff_list)
        self.assertFalse("你好" in diff_list)
        self.assertFalse("世界" in diff_list)
        self.assertTrue("應該沒有這個詞" in diff_list)

    def test_sort_by_length_and_code(self):
        word_codes = {
            "你好": [("nihcd", 100)],
            "世界": [("uijcd", 200)],
            "再见": [("zajd", 150)]
        }
        sorted_dict = sort_by_length_and_code(word_codes)
        self.assertTrue(2 in sorted_dict)
        self.assertTrue("nihcd" in sorted_dict[2])
        self.assertTrue("uijcd" in sorted_dict[2])
        self.assertTrue("zajd" in sorted_dict[2])
        self.assertEqual(sorted_dict[2]["nihcd"]["你好"], 100)
        self.assertEqual(sorted_dict[2]["uijcd"]["世界"], 200)
        self.assertEqual(sorted_dict[2]["zajd"]["再见"], 150)

    def test_augment_two_character_words(self):
        word_codes = {
            2: {
                "nihcd": {"你好": 100, "你号": 50},
                "uijcd": {"世界": 200},
            }
        }
        augmented_dict = augment_two_character_words(word_codes)
        self.assertTrue("nihcd" in augmented_dict[2])
        self.assertTrue("uijcd" in augmented_dict[2])
        self.assertTrue("nihcdo" in augmented_dict[2])
        self.assertEqual(augmented_dict[2]["nihcd"]["你好"], 100)
        self.assertEqual(augmented_dict[2]["nihcdo"]["你号"], 50)
        self.assertEqual(augmented_dict[2]["uijcd"]["世界"], 200)
        self.assertFalse("nihcd" in augmented_dict[2] and "你号" in augmented_dict[2]["nihcd"])

    def test_process_and_print_flypyquick5_dict(self):
        words = {
            "你好": [["nǐ", "hǎo"]],
            "世界": [["shì", "jiè"]]
        }
        from io import StringIO
        output = StringIO()
        process_and_print_flypyquick5_dict(words, outfile=output)
        output_str = output.getvalue()
        self.assertIn("你好", output_str)
        self.assertIn("世界", output_str)
        self.assertIn("nihcd", output_str)
        self.assertIn("uijpl", output_str)

# Steo 8: Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Convert Pinyin with diacritics to Shuangpin (Xiaohe scheme).")
    parser.add_argument("--chinese_code", help="print chinese code", action="store_true")
    parser.add_argument("--name", help="the name of the current table", required = False)
    parser.add_argument("--input_tables", nargs='*', help="Input tables to import", default=[])
    parser.add_argument("--pinyin_phrase", help="print builtin pinyin phrase", action="store_true")
    parser.add_argument("--difference", help="use difference set against the builtin pinyin phrases", action="store_true")
    parser.add_argument("--test", help="run unit tests", action="store_true")
    parser.add_argument("input_file", nargs="?", help="Input file name", default=None)
    args = parser.parse_args()

    input_tables = args.input_tables

    if args.chinese_code:
        # Print Chinese character codes
        print(get_header(args.name, input_tables))
        character_dict = convert_to_nested_dict(kCharacterCodes)
        process_and_print_flypyquick5_dict(character_dict, sys.stdout)
    elif args.pinyin_phrase:
        # Print Pinyin phrases
        print(get_header(args.name, input_tables))
        process_and_print_flypyquick5_dict(kPinyinPhrases, sys.stdout)
    elif args.input_file:
        primary_dict = get_sorted_flypyquick5_dict(kPinyinPhrases)
        # Convert Pinyin from input file to Shuangpin (Xiaohe scheme)
        print(get_header(args.name, input_tables))
        words = get_words_from_file(args.input_file)
        if args.difference:
            words = get_difference_set(words)
        pinyin_seq_dict = get_pinyin_seq_for_words(words)
        process_and_print_flypyquick5_dict(pinyin_seq_dict, sys.stdout, primary_dict)
    elif args.test:
        # Run unit tests
        unittest.main(argv=[sys.argv[0]], exit=False)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

