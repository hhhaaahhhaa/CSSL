import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import random
from typing import List
import json

import Define
from .parser import DataParser
from .utils import write_queries_to_txt
from .checkers import *


random.seed(0)


def clean(data_parser: DataParser, output_path: str) -> List:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    queries = data_parser.get_all_queries()
    checkers = [
        SSLFeatureChecker("hubert", data_parser),
        SSLFeatureChecker("wav2vec2", data_parser)
    ]
    res = queries
    for checker in checkers:
        res = checker.filter(res)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)
