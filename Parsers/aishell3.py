import os
import json
from pathlib import Path
import librosa
import random

from dlhlp_lib.parsers.raw_parsers import AISHELL3RawParser, AISHELL3Instance
from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp
from dlhlp_lib.audio.tools import wav_normalization

import Define
from .interface import BasePreprocessor
from .parser import DataParser
from .utils import write_queries_to_txt
from . import template


class AISHELL3Preprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = AISHELL3RawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info = []
        for instance in self.src_parser.train_set:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "train",
            }
            data_info.append(query)
        for instance in self.src_parser.test_set:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "test",
            }
            data_info.append(query)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)

        def _func(instance: AISHELL3Instance) -> None:
            query = {
                "basename": instance.id,
                "spk": instance.speaker
            }
            wav_16000 = librosa.load(instance.wav_path, sr=16000)
            wav_16000 = wav_normalization(wav_16000)
            self.data_parser.wav_16000.save(wav_16000, query)
            self.data_parser.text.save(instance.text, query)

        tasks = [(x,) for x in self.src_parser.train_set + self.src_parser.test_set]
        process_tasks_mp(tasks, ImapWrapper(_func), n_workers=n_workers, chunksize=chunksize, ignore_errors=True)
        self.data_parser.text.build_cache()

    # Use prepared textgrids from ming024's repo
    def prepare_mfa(self, mfa_data_dir: Path) -> None:
        pass
    
    # Use prepared textgrids from ming024's repo
    def mfa(self, mfa_data_dir: Path) -> None:
        pass
    
    def preprocess(self):
        queries = self.data_parser.get_all_queries()
        if Define.DEBUG:
            queries = queries[:128]
        template.preprocess(self.data_parser, queries)

    def split_dataset(self, cleaned_data_info_path: str):
        random.seed(0)
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        train_set, test_set = [], []
        for q in queries:
            if q["dset"] == "train":
                train_set.append(q)
            else:
                test_set.append(q)
        val_set = random.sample(train_set, k=2500)
        write_queries_to_txt(self.data_parser, train_set, f"{output_dir}/train.txt")
        write_queries_to_txt(self.data_parser, val_set, f"{output_dir}/val.txt")
        write_queries_to_txt(self.data_parser, test_set, f"{output_dir}/test.txt")
