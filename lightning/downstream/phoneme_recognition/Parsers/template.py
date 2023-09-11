import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import random

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.tts_preprocess.basic2 import *

import Define
from .parser import DataParser
from .utils import write_queries_to_txt


FRAME_PERIOD = AUDIO_CONFIG["stft"]["hop_length"] / AUDIO_CONFIG["audio"]["sampling_rate"]
random.seed(0)


def preprocess(data_parser: DataParser, queries):
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
    write_queries_to_txt(data_parser, train_set, f"{output_dir}/train.txt")
    write_queries_to_txt(data_parser, val_set, f"{output_dir}/val.txt")
    write_queries_to_txt(data_parser, test_set, f"{output_dir}/test.txt")
