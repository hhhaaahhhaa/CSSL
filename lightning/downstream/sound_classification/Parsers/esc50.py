import os
import json
from pathlib import Path
import librosa
import random
from datasets import load_dataset
from tqdm import tqdm

from dlhlp_lib.audio.tools import wav_normalization

from .base import BasePreprocessor
from .parser import DataParser
from .utils import write_queries_to_txt
from . import template


class ESC50Preprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self._connect_hf_dataset()
        self.data_parser = DataParser(root)

    def _connect_hf_dataset(self):
        self.log("Connecting huggingface...")
        self.src_dataset = load_dataset(
            "ashraq/esc50",
            split="train",
            streaming=True,
        )

    def parse_raw_process(self, instance) -> None:
        query = {
            "basename": instance["filename"][:-4],
        }
        wav_16000 = librosa.resample(instance["audio"]["array"], orig_sr=instance["audio"]["sampling_rate"], target_sr=16000)
        wav_16000 = wav_normalization(wav_16000)
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.label.save(
            {"class": instance["category"]}, 
            query
        )
    
    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info, all_classes = [], []
        for instance in tqdm(self.src_dataset):
            c = instance["category"]
            query = {
                "basename": instance["filename"][:-4],
            }
            self.parse_raw_process(instance)
            data_info.append(query)
            if c not in all_classes:
                all_classes.append(c)

        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)
        with open(f"{self.data_parser.root}/classes.json", "w", encoding="utf-8") as f:
            json.dump(all_classes, f, indent=4)
        self.data_parser.label.build_cache()
    
    def preprocess(self):
        pass

    def clean(self):
        cleaned_data_info_path = "data_config/ESC50/clean.json"
        template.clean(self.data_parser, output_path=cleaned_data_info_path)
    
    def split_dataset(self):
        cleaned_data_info_path = "data_config/ESC50/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        temp = {}
        for query in queries:
            c = self.data_parser.label.read_from_query(query)["class"]
            if c not in temp:
                temp[c] = [query]
            else:
                temp[c].append(query)
        train_set, test_set = [], []
        for c, qs in temp.items():
            train_set.extend(qs[:-4])
            test_set.extend(qs[-4:])
        assert len(test_set) == 200

        write_queries_to_txt(self.data_parser, train_set, f"{output_dir}/train.txt")
        write_queries_to_txt(self.data_parser, test_set, f"{output_dir}/val.txt")
        write_queries_to_txt(self.data_parser, test_set, f"{output_dir}/test.txt")

    def log(self, msg):
        print(f"[ESC50Preprocessor]: ", msg)
