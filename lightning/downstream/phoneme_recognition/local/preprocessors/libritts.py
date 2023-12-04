import os
import json
from pathlib import Path
import librosa
import random
import yaml

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.parsers.raw_parsers import LibriTTSRawParser, LibriTTSInstance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp
from dlhlp_lib.audio.tools import wav_normalization

import Define
from parser import DataParser
from . import template


class LibriTTSPreprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = LibriTTSRawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw_process(self, instance: LibriTTSInstance) -> None:
        query = {
            "basename": instance.id,
            "spk": instance.speaker,
        }
        wav_16000, _ = librosa.load(instance.wav_path, sr=16000)
        wav_16000 = wav_normalization(wav_16000)
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.text.save(instance.text, query)

    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info = []
        for instance in self.src_parser.train_clean_100:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "train-clean-100",
            }
            data_info.append(query)
        for instance in self.src_parser.dev_clean:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "dev-clean",
            }
            data_info.append(query)
        for instance in self.src_parser.test_clean:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "test-clean",
            }
            data_info.append(query)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)

        tasks = [(x,) for x in self.src_parser.train_clean_100 + self.src_parser.dev_clean + self.src_parser.test_clean]
        process_tasks_mp(tasks, self.parse_raw_process, n_workers=n_workers, chunksize=chunksize, ignore_errors=False)
        self.data_parser.text.build_cache()

    def preprocess(self):
        textgrid_root = self.data_parser.textgrid.query_parser.root
        if not os.path.exists(textgrid_root):
            self.log("Missing textgrid!")
            raise NotImplementedError

        queries = self.data_parser.get_all_queries()
        if Define.DEBUG:
            queries = queries[:128]
        template.preprocess(self.data_parser, queries)
    
    def clean(self):
        cleaned_data_info_path = "data_config/LibriTTS/clean.json"
        template.clean(self.data_parser, output_path=cleaned_data_info_path)
    
    def split_dataset(self):
        random.seed(0)
        cleaned_data_info_path = "data_config/LibriTTS/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        train_set, val_set, test_set = [], [], []
        for q in queries:
            if q["dset"] == "train-clean-100":
                train_set.append(q)
            elif q["dset"] == "dev-clean":
                val_set.append(q)
            else:
                test_set.append(q)
        val_set = random.sample(val_set, k=2500)
        test_set = random.sample(test_set, k=2500)
        template.write_split(self.data_parser, output_dir, train_set, val_set, test_set)

        # Generate config.yaml
        with open(f"data_config/LibriTTS/config.yaml", 'w') as yamlfile:
            config = {
                "name": "LibriTTS",
                "lang_id": "en",
                "data_dir": self.data_parser.root,
                "subsets": {
                    "train": "train.json",
                    "val": "val.json",
                    "test": "test.json"
                },
                "text_cleaners": ["english_cleaners"]
            }
            yaml.dump(config, yamlfile)

    def log(self, msg):
        print(f"[LibriTTSPreprocessor]: ", msg)
