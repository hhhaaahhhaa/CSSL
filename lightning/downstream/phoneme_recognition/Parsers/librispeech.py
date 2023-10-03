import os
import json
from pathlib import Path
import librosa
import random
from datasets import load_dataset
from tqdm import tqdm

from dlhlp_lib.audio.tools import wav_normalization

import Define
from .interface import BasePreprocessor
from .parser import DataParser
from .utils import write_queries_to_txt
from . import template


class LibriSpeechPreprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self._connect_hf_dataset()
        self.data_parser = DataParser(root)

    def _connect_hf_dataset(self):
        self.log("Connecting huggingface...")
        splits = [
            "train.clean.100",
            "validation.clean",
            "test.clean"
        ]
        self.src_datasets = {
            k: load_dataset(
                "librispeech_asr",
                split=k,
                streaming=True,
            )
        for k in splits}

    def parse_raw_process(self, instance) -> None:
        query = {
            "basename": instance["id"],
            "spk": instance["speaker_id"]
        }
        wav_16000 = instance["audio"]["array"]
        wav_16000 = wav_normalization(wav_16000)
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.text.save(instance["text"], query)
    
    def parse_raw(self, n_workers=8, chunksize=64) -> None:
        # create data info
        data_info = []
        for k, ds in self.src_datasets.items():
            for instance in tqdm(ds):
                query = {
                    "basename": instance["id"],
                    "spk": str(instance["speaker_id"]),
                    "dset": k
                }
                self.parse_raw_process(instance)
                data_info.append(query)

        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)
        self.data_parser.text.build_cache()
    
    def preprocess(self):
        textgrid_root = self.data_parser.textgrid.query_parser.root
        if not os.path.exists(textgrid_root):
            self.log("Missing textgrid!")
            self.mfa(textgrid_root)
        
        queries = self.data_parser.get_all_queries()
        if Define.DEBUG:
            queries = queries[:128]
        template.preprocess(self.data_parser, queries)

    def clean(self):
        cleaned_data_info_path = f"data_config/LibriSpeech/clean.json"
        template.clean(self.data_parser, output_path=cleaned_data_info_path)
    
    def split_dataset(self):
        random.seed(0)
        cleaned_data_info_path = f"data_config/LibriSpeech/clean.json"
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        train_set, val_set, test_set = [], [], []
        for q in queries:
            if q["dset"] == "train.clean.100":
                train_set.append(q)
            elif q["dset"] == "validation.clean":
                val_set.append(q)
            else:
                test_set.append(q)
        val_set = random.sample(val_set, k=2500)
        test_set = random.sample(test_set, k=2500)
        write_queries_to_txt(self.data_parser, train_set, f"{output_dir}/train.txt")
        write_queries_to_txt(self.data_parser, val_set, f"{output_dir}/val.txt")
        write_queries_to_txt(self.data_parser, test_set, f"{output_dir}/test.txt")

    def log(self, msg):
        print(f"[LibriSpeechPreprocessor]: ", msg)

    # MFA
    def _prepare_mfa_dir(self, mfa_data_dir: Path) -> None:
        if mfa_data_dir.exists():
            return
        
        # 1. construct MFA tool predefined directory structure in "mfa_data_dir"
        self.log(f"Construct MFA directory ({mfa_data_dir})...")
        queries = self.data_parser.get_all_queries()

        # 2. create hard link for wav file
        for query in tqdm(queries):
            target_dir = mfa_data_dir / query["spk"]
            target_dir.mkdir(parents=True, exist_ok=True)
            link_file = target_dir / f"{query['basename']}.wav"
            txt_link_file = target_dir / f"{query['basename']}.txt"
            wav_file = self.data_parser.wav_16000.read_filename(query, raw=True)
            txt_file = self.data_parser.text.read_filename(query, raw=True)
            
            if link_file.exists():
                os.unlink(str(link_file))
            if txt_link_file.exists():
                os.unlink(str(txt_link_file))
            os.link(wav_file, str(link_file))
            os.link(txt_file, str(txt_link_file))
    
    def mfa(self, output_path) -> None:
        mfa_data_dir = Path(self.root) / "mfa_data"

        self._prepare_mfa_dir(mfa_data_dir)
        os.system("mfa model download dictionary english_us_mfa")
        os.system("mfa model download acoustic english_mfa")

        self.log("Start MFA align!")
        cmd = f"mfa align {mfa_data_dir} english_us_mfa english_mfa {output_path} -j 8 -v --clean"
        os.system(cmd)
