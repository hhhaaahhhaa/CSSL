import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from typing import List
import json

from parser import DataParser
from .checkers import *


def clean(data_parser: DataParser, output_path: str) -> List:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    queries = data_parser.get_all_queries()
    checkers = [
        ExistenceChecker("wav_16000", data_parser.wav_16000),
        SSLFeatureChecker("hubert", data_parser),
        SSLFeatureChecker("wav2vec2", data_parser)
    ]
    res = queries
    for checker in checkers:
        res = checker.filter(res)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)
