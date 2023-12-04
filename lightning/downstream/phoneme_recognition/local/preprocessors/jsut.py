import os
import json
from pathlib import Path
import librosa
import yaml

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.parsers.raw_parsers import JSUTRawParser, JSUTInstance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp
from dlhlp_lib.audio.tools import wav_normalization

import Define
from parser import DataParser
from . import template


class JSUTPreprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = JSUTRawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw_process(self, instance: JSUTInstance) -> None:
        query = {
            "basename": instance.id,
            "spk": "jsut"
        }
        wav_16000, _ = librosa.load(instance.wav_path, sr=16000)
        wav_16000 = wav_normalization(wav_16000)
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.text.save(instance.text, query)
    
    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info = []
        for instance in self.src_parser.basic5000:
            query = {
                "basename": instance.id,
                "spk": "jsut"
            }
            data_info.append(query)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)

        tasks = [(x,) for x in self.src_parser.basic5000]
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

        phoneset_path = f"{os.path.dirname(__file__)}/../MFA/JSUT/phoneset.txt"
        if not os.path.exists(phoneset_path):
            self.log("Generate phoneme set...")
            from scripts.collect_phonemes import collect_phonemes, generate_phoneme_set
            phns = collect_phonemes([self.root])
            generate_phoneme_set(phns, phoneset_path)

    def clean(self):
        cleaned_data_info_path = "data_config/JSUT/clean.json"
        template.clean(self.data_parser, output_path=cleaned_data_info_path)
    
    def split_dataset(self):
        cleaned_data_info_path = "data_config/JSUT/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        template.split_monospeaker_dataset(self.data_parser, queries, output_dir, val_size=1000)

        # Generate config.yaml
        with open(f"data_config/JSUT/config.yaml", 'w') as yamlfile:
            config = {
                "name": "JSUT",
                "lang_id": "jp",
                "data_dir": self.data_parser.root,
                "subsets": {
                    "train": "train.json",
                    "val": "val.json",
                    "test": "test.json"
                },
                "text_cleaners": ["transliteration_cleaners"]
            }
            yaml.dump(config, yamlfile)

    def log(self, msg):
        print(f"[JSUTPreprocessor]: ", msg)
