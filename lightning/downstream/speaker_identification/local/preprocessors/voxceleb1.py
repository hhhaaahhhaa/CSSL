import os
import json
from pathlib import Path
import librosa
import yaml

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.parsers.raw_parsers import Voxceleb1RawParser, Voxceleb1Instance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp

from parser import DataParser


class Voxceleb1Preprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = Voxceleb1RawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw_process(self, instance: Voxceleb1Instance) -> None:
        query = {
            "basename": instance.id,
            "spk": instance.speaker
        }
        tgt_path = self.data_parser.wav_16000.read_filename(query, raw=True)
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        os.link(instance.wav_path, tgt_path)

    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info = []
        all_speakers = []
        for instance in self.src_parser.train:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "train"
            }
            data_info.append(query)
            if instance.speaker not in all_speakers:
                all_speakers.append(instance.speaker)
        for instance in self.src_parser.dev:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "dev"
            }
            data_info.append(query)
            if instance.speaker not in all_speakers:
                all_speakers.append(instance.speaker)
        for instance in self.src_parser.test:
            query = {
                "basename": instance.id,
                "spk": instance.speaker,
                "dset": "test"
            }
            data_info.append(query)
            if instance.speaker not in all_speakers:
                all_speakers.append(instance.speaker)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)
        with open(f"{self.data_parser.root}/speakers.json", "w", encoding="utf-8") as f:
            json.dump(all_speakers, f, indent=4)

        tasks = [(x,) for x in self.src_parser.train + self.src_parser.dev + self.src_parser.test]
        process_tasks_mp(tasks, self.parse_raw_process, n_workers=n_workers, chunksize=chunksize, ignore_errors=False)
    
    def preprocess(self):
        pass

    def clean(self):
        cleaned_data_info_path = "data_config/Voxceleb1/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        os.makedirs(output_dir, exist_ok=True)
        queries = self.data_parser.get_all_queries()
        with open(cleaned_data_info_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=4)
    
    def split_dataset(self):
        cleaned_data_info_path = "data_config/Voxceleb1/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        train_set, val_set, test_set = [], [], []
        for q in queries:
            if q["dset"] == "train":
                train_set.append(q)
            elif q["dset"] == "dev":
                val_set.append(q)
            else:
                test_set.append(q)

        # Write, txt for human readable and json(unified format) for system usage
        self.write_queries_to_txt(train_set, f"{output_dir}/train.txt")
        self.write_queries_to_txt(val_set, f"{output_dir}/val.txt")
        self.write_queries_to_txt(test_set, f"{output_dir}/test.txt")
        with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
            json.dump(train_set, f, indent=4)
        with open(f"{output_dir}/val.json", 'w', encoding='utf-8') as f:
            json.dump(val_set, f, indent=4)
        with open(f"{output_dir}/test.json", 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4)

        # Generate config.yaml
        with open("data_config/Voxceleb1/config.yaml", 'w') as yamlfile:
            config = {
                "name": "Voxceleb1",
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
                line = [query["basename"], query["spk"]]
                lines.append(line)
            except:
                print("Failed: ", query)
                raise
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write("|".join(line))
                f.write('\n')
    
    def log(self, msg):
        print(f"[Voxceleb1Preprocessor]: ", msg)
