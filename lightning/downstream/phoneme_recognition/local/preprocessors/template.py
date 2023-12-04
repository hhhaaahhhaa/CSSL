import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import random
from typing import List
import json

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.tts_preprocess.basic2 import *

import Define
from parser import DataParser
from .checkers import *


FRAME_PERIOD = AUDIO_CONFIG["stft"]["hop_length"] / AUDIO_CONFIG["audio"]["sampling_rate"]
random.seed(0)


def preprocess(data_parser: DataParser, queries):
    data_parser.create_unit_feature(unit_name="mfa")
    ignore_errors = not Define.DEBUG
    textgrid2segment_and_phoneme(
        queries,
        data_parser.textgrid,
        data_parser.units["mfa"].segment,
        data_parser.units["mfa"].phoneme,
        n_workers=os.cpu_count()//2,
        ignore_errors=ignore_errors
    )
    trim_wav_by_segment(
        queries,
        data_parser.wav_16000,
        data_parser.units["mfa"].segment,
        data_parser.wav_trim_16000,
        n_workers=1,
        ignore_errors=ignore_errors
    )
    segment2duration(
        queries,
        data_parser.units["mfa"].segment,
        data_parser.units["mfa"].duration,
        FRAME_PERIOD,
        n_workers=os.cpu_count()//2,
        ignore_errors=ignore_errors
    )
    
    # Generate cache
    data_parser.units["mfa"].phoneme.build_cache()
    data_parser.units["mfa"].segment.build_cache()
    data_parser.units["mfa"].duration.build_cache()


def split_monospeaker_dataset(data_parser: DataParser, queries, output_dir, val_size=1000):
    train_set = queries[:-val_size]
    test_set = queries[-val_size:]
    val_set = test_set
    write_split(data_parser, output_dir, train_set, val_set, test_set)


def clean(data_parser: DataParser, output_path: str) -> List:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    queries = data_parser.get_all_queries()
    checkers = [
        LengthChecker(data_parser),
        UnknownTokenChecker(data_parser, unk_list=["spn"]),
        SSLFeatureChecker("hubert", data_parser),
        SSLFeatureChecker("wav2vec2", data_parser)
    ]
    res = queries
    for checker in checkers:
        res = checker.filter(res)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)


def write_split(data_parser: DataParser, output_dir, train_set, val_set, test_set):
    write_queries_to_txt(data_parser, train_set, f"{output_dir}/train.txt")
    write_queries_to_txt(data_parser, val_set, f"{output_dir}/val.txt")
    write_queries_to_txt(data_parser, test_set, f"{output_dir}/test.txt")
    with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_set, f, indent=4)
    with open(f"{output_dir}/val.json", 'w', encoding='utf-8') as f:
        json.dump(val_set, f, indent=4)
    with open(f"{output_dir}/test.json", 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=4)


def write_queries_to_txt(data_parser: DataParser, queries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_parser.units["mfa"].phoneme.read_all()
    data_parser.text.read_all()
    lines = []
    for query in queries:
        try:
            line = [query["basename"], query["spk"]]
            line.append(f"{{{data_parser.units['mfa'].phoneme.read_from_query(query)}}}")
            line.append(data_parser.text.read_from_query(query))
            lines.append(line)
        except:
            print("Please delete phoneme cache and text cache and try again.")
            print("If not working, phoneme feature/text feature does not contain such query.")
            print("Failed: ", query)
            raise
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write("|".join(line))
            f.write('\n')
