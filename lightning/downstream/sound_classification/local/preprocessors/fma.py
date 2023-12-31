import os
import json
import librosa
import yaml

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.parsers.raw_parsers import FMAMusicRawParser, FMAMusicInstance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp
from dlhlp_lib.audio.tools import wav_normalization

from parser import DataParser
from . import template


class FMAMusicPreprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = FMAMusicRawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw_process(self, instance: FMAMusicInstance) -> None:
        query = {
            "basename": instance.id,
        }
        try:
            wav_16000, _ = librosa.load(instance.wav_path, sr=16000)
            wav_16000 = wav_normalization(wav_16000)
        except:
            print("Skipped: ", instance.wav_path)
            return
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.label.save(
            {"class": instance.genre}, 
            query
        )

    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        n_workers=1
        # create data info
        data_info = []
        all_genres = []
        for instance in self.src_parser.training:
            query = {
                "basename": instance.id,
                "dset": "training"
            }
            data_info.append(query)
            if instance.genre not in all_genres:
                all_genres.append(instance.genre)
        for instance in self.src_parser.validation:
            query = {
                "basename": instance.id,
                "dset": "validation"
            }
            data_info.append(query)
            if instance.genre not in all_genres:
                all_genres.append(instance.genre)
        for instance in self.src_parser.test:
            query = {
                "basename": instance.id,
                "dset": "test"
            }
            data_info.append(query)
            if instance.genre not in all_genres:
                all_genres.append(instance.genre)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)
        with open(f"{self.data_parser.root}/classes.json", "w", encoding="utf-8") as f:
            json.dump(all_genres, f, indent=4)

        tasks = [(x,) for x in self.src_parser.training + self.src_parser.validation + self.src_parser.test]
        process_tasks_mp(tasks, self.parse_raw_process, n_workers=n_workers, chunksize=chunksize, ignore_errors=False)
        self.data_parser.label.build_cache()
    
    def preprocess(self):
        pass

    def clean(self):
        cleaned_data_info_path = "data_config/fma/clean.json"
        template.clean(self.data_parser, output_path=cleaned_data_info_path)
    
    def split_dataset(self):
        cleaned_data_info_path = "data_config/fma/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        train_set, val_set, test_set = [], [], []
        for q in queries:
            if q["dset"] == "training":
                train_set.append(q)
            elif q["dset"] == "validation":
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
        with open("data_config/fma/config.yaml", 'w') as yamlfile:
            config = {
                "name": "fma",
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
        self.data_parser.label.read_all()
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
        print(f"[FMAMusicPreprocessor]: ", msg)
