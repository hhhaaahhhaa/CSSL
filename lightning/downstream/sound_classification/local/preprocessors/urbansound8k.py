import os
import json
import librosa
import yaml

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.parsers.raw_parsers import UrbanSound8KRawParser, UrbanSound8KInstance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp

from parser import DataParser
from . import template


class UrbanSound8KPreprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = UrbanSound8KRawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw_process(self, instance: UrbanSound8KInstance) -> None:
        query = {
            "basename": instance.id,
        }
        try:
            wav_16000, _ = librosa.load(instance.wav_path, sr=16000)
        except:
            print("Skipped: ", instance.wav_path)
            return
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.label.save(
            {"class": instance.class_name, "fold": instance.fold},
            query
        )

    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info = []
        all_classes = []
        for instance in self.src_parser.dataset:
            query = {
                "basename": instance.id
            }
            data_info.append(query)
            if instance.class_name not in all_classes:
                all_classes.append(instance.class_name)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)
        with open(f"{self.data_parser.root}/classes.json", "w", encoding="utf-8") as f:
            json.dump(all_classes, f, indent=4)

        tasks = [(x,) for x in self.src_parser.dataset]
        process_tasks_mp(tasks, self.parse_raw_process, n_workers=n_workers, chunksize=chunksize, ignore_errors=False)
        self.data_parser.label.build_cache()
    
    def preprocess(self):
        pass

    def clean(self):
        cleaned_data_info_path = "data_config/UrbanSound8K/clean.json"
        template.clean(self.data_parser, output_path=cleaned_data_info_path)
    
    def split_dataset(self):
        cleaned_data_info_path = "data_config/UrbanSound8K/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        self.data_parser.label.read_all()
        
        train_set, test_set = [], []
        for q in queries:
            fold = self.data_parser.label.read_from_query(q)["fold"]
            if fold != 10:
                train_set.append(q)
            else:
                test_set.append(q)
        val_set = test_set

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
        with open("data_config/UrbanSound8K/config.yaml", 'w') as yamlfile:
            config = {
                "name": "UrbanSound8K",
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
        print(f"[UrbanSound8KPreprocessor]: ", msg)
