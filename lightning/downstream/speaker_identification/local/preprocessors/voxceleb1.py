import os
import json
from pathlib import Path
import librosa
import yaml
from tqdm import tqdm

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.parsers.raw_parsers import Voxceleb1RawParser, Voxceleb1Instance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp

from parser import DataParser


class VoxCeleb1Preprocessor(BasePreprocessor):

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
        # load mini
        local_dir = os.path.dirname(os.path.dirname(__file__))
        mini_names = [line.strip().split(" ")[1][:-4] for line in open(f"{local_dir}/mini_voxceleb1.txt", "r").readlines()]
        mini_names = [x.replace("/", "-") for x in mini_names]
                
        # create data info
        data_info = []
        all_speakers = []
        mini_set = []
        for dset in ["train", "dev", "test"]:
            for instance in tqdm(getattr(self.src_parser, dset)):
                if instance.id not in mini_names:
                    continue
                mini_set.append(instance)
                query = {
                    "basename": instance.id,
                    "spk": instance.speaker,
                    "dset": dset
                }
                data_info.append(query)
                if instance.speaker not in all_speakers:
                    all_speakers.append(instance.speaker)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)
        with open(f"{self.data_parser.root}/speakers.json", "w", encoding="utf-8") as f:
            json.dump(all_speakers, f, indent=4)

        tasks = [(x,) for x in mini_set]
        process_tasks_mp(tasks, self.parse_raw_process, n_workers=n_workers, chunksize=chunksize, ignore_errors=False)
    
    def preprocess(self):
        pass

    def clean(self):
        cleaned_data_info_path = "data_config/VoxCeleb1/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        os.makedirs(output_dir, exist_ok=True)
        queries = self.data_parser.get_all_queries()
        with open(cleaned_data_info_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=4)
    
    def split_dataset(self):
        cleaned_data_info_path = "data_config/VoxCeleb1/clean.json"
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
        with open("data_config/VoxCeleb1/config.yaml", 'w') as yamlfile:
            config = {
                "name": "VoxCeleb1",
                "data_dir": self.data_parser.root,
                "subsets": {
                    "train": "train.json",
                    "val": "test.json",
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
        print(f"[VoxCeleb1Preprocessor]: ", msg)
