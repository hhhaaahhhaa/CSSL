from typing import List, Dict

from text.symbols import common_symbols, en_symbols, zh_symbols


def get_phoneme_set(path, encoding='utf-8'):
    phns = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            if line == '\n':
                continue
            phns.append('@' + line.strip())
    return phns


LANG_ID2SYMBOLS = {
    "en": en_symbols,
    "zh": zh_symbols,
    "fr": common_symbols + get_phoneme_set("lightning/downstream/phoneme_recognition/MFA/M-AILABS/fr_FR/phoneset.txt"),
    "de": common_symbols + get_phoneme_set("lightning/downstream/phoneme_recognition/MFA/CSS10/german/phoneset.txt"),
    "ru": common_symbols + get_phoneme_set("lightning/downstream/phoneme_recognition/MFA/M-AILABS/ru_RU/phoneset.txt"),
    # "es": common_symbols + get_phoneme_set("MFA/Spanish/phoneset.txt"),
    "jp": common_symbols + get_phoneme_set("lightning/downstream/phoneme_recognition/MFA/JSUT/phoneset.txt"),
    "cz": [],
    "ko": common_symbols + get_phoneme_set("lightning/downstream/phoneme_recognition/MFA/kss/phoneset.txt"),
    "nl": [],
}


LANGS = [
    "en", "zh", "fr", "de", "ru", "es", "jp", "cz", "ko", "nl"
]
LANG_ID2NAME = {i: name for i, name in enumerate(LANGS)}
LANG_NAME2ID = {name: i for i, name in enumerate(LANGS)}
