import os
import json
from pathlib import Path
import librosa
import random
from datasets import load_dataset
from tqdm import tqdm
import yaml

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.audio.tools import wav_normalization

from parser import DataParser
from . import template


class ESC50Preprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
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
        self._connect_hf_dataset()
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
        self.data_parser.label.read_all()
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

        # Write, txt for human readable and json(unified format) for system usage
        self.write_queries_to_txt(train_set, f"{output_dir}/train.txt")
        self.write_queries_to_txt(test_set, f"{output_dir}/val.txt")
        self.write_queries_to_txt(test_set, f"{output_dir}/test.txt")
        with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
            json.dump(train_set, f, indent=4)
        with open(f"{output_dir}/val.json", 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4)
        with open(f"{output_dir}/test.json", 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4)

        # Generate config.yaml
        with open("data_config/ESC50/config.yaml", 'w') as yamlfile:
            config = {
                "name": "ESC50",
                "data_dir": self.data_parser.root,
                "subsets": {
                    "train": "train.json",
                    "val": "val.json",
                    "test": "test.json"
                }
            }
            yaml.dump(config, yamlfile)
    
    def write_queries_to_txt(self, queries, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = []
        for query in queries:
            try:
                line = [query["basename"], self.data_parser.label.read_from_query(query)["class"]]
                lines.append(line)
            except:
                print("Failed: ", query)
                raise
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write("|".join(line))
                f.write('\n')

    def log(self, msg):
        print(f"[ESC50Preprocessor]: ", msg)
