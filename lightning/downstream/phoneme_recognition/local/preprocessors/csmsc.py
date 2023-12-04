import os
import json
from pathlib import Path
import librosa
from tqdm import tqdm
from praatio import textgrid
import yaml

from dlhlp_lib.parsers.Interfaces import BasePreprocessor
from dlhlp_lib.parsers.raw_parsers import CSMSCRawParser, CSMSCInstance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp
from dlhlp_lib.audio.tools import wav_normalization

import Define
from parser import DataParser
from . import template


class CSMSCPreprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = CSMSCRawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw_process(self, instance: CSMSCInstance) -> None:
        query = {
            "basename": instance.id,
            "spk": "csmsc"
        }
        wav_16000, _ = librosa.load(instance.wav_path, sr=16000)
        wav_16000 = wav_normalization(wav_16000)
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.text.save(instance.text, query)
    
    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info = []
        for instance in self.src_parser.dataset:
            query = {
                "basename": instance.id,
                "spk": "csmsc"
            }
            data_info.append(query)
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)

        tasks = [(x,) for x in self.src_parser.dataset]
        process_tasks_mp(tasks, self.parse_raw_process, n_workers=n_workers, chunksize=chunksize, ignore_errors=False)
        self.data_parser.text.build_cache()

    def preprocess(self):
        textgrid_root = self.data_parser.textgrid.query_parser.root
        if not os.path.exists(textgrid_root):
            self.log("Generate textgrid...")
            os.makedirs(f"{textgrid_root}/csmsc", exist_ok=True)
            self.generate_textgrid()

        queries = self.data_parser.get_all_queries()
        if Define.DEBUG:
            queries = queries[:128]
        template.preprocess(self.data_parser, queries)

    def clean(self):
        cleaned_data_info_path = "data_config/CSMSC/clean.json"
        template.clean(self.data_parser, output_path=cleaned_data_info_path)
    
    def split_dataset(self):
        cleaned_data_info_path = "data_config/CSMSC/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        template.split_monospeaker_dataset(self.data_parser, queries, output_dir, val_size=1000)

        # Generate config.yaml
        with open("data_config/CSMSC/config.yaml", 'w') as yamlfile:
            config = {
                "name": "CSMSC",
                "lang_id": "zh",
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
        print(f"[CSMSCPreprocessor]: ", msg)

    def generate_textgrid(self):
        for instance in tqdm(self.src_parser.dataset):
            alignment = textgrid.openTextgrid(instance.textgrid_path, includeEmptyIntervals=True)

            phones = []
            ends = []
            typo_list = [
                "006267", "007017", "007030", "007114", "008127"  # typos in original CSMSC dataset, future version may fix this issue
            ]
            tiername = f"{instance.id}.interval" if instance.id not in typo_list else instance.id
            for interval in alignment.getTier(tiername)._entries:
                phones.append(interval.label)
                ends.append(interval.end)
            # merge  "" and sp in the end
            if phones[-1] == "" and len(phones) > 1 and phones[-2] == "sp":
                phones = phones[:-1]
                durations[-2] += durations[-1]
                durations = durations[:-1]
            # replace the last "sp" with "sil" in MFA1.x
            phones[-1] = "sil" if phones[-1] == "sp" else phones[-1]
            # replace the edge "" with "sil", replace the inner "" with "sp"
            new_phones = []
            for i, phn in enumerate(phones):
                if phn == "":
                    if i in {0, len(phones) - 1}:
                        new_phones.append("sil")
                    else:
                        new_phones.append("sp")
                else:
                    new_phones.append(phn)
            phones = new_phones

            # Write to textgrid
            tgt_labels = []
            st = 0
            for (phn, ed) in zip(new_phones, ends):
                tgt_labels.append((st, ed, phn))
                st = ed
            query = {
                "basename": instance.id,
                "spk": "csmsc"
            }
            dst_path = self.data_parser.textgrid.read_filename(query, raw=True)
            self._write_textgrid(dst_path, tgt_labels)

    def _write_textgrid(self, dst_path, labels):
        """Write TextGrid from custom style labels, this function is modified from nnmnkwii.io.hts.write_textgrid

        Args:
            dst_path (str): The output file path.
            labels: custom style labels (list of (st, ed, phn))
        """
        template = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {xmax}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phoneme"
        xmin = 0
        xmax = {xmax}
        intervals: size = {size}"""
        template = template.format(xmax=labels[-1][1], size=len(labels))

        for idx, (st, ed, phn) in enumerate(labels):
            template += """
            intervals [{idx}]:
                xmin = {st}
                xmax = {ed}
                text = "{phn}" """.format(
                idx=idx + 1, st=st, ed=ed, phn=phn
            )
        template += "\n"

        with open(dst_path, "w") as of:
            of.write(template)
