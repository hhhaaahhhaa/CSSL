import os
import json
from pathlib import Path
import librosa
from tqdm import tqdm

from dlhlp_lib.parsers.raw_parsers import KSSRawParser, KSSInstance
from dlhlp_lib.tts_preprocess.basic2 import process_tasks_mp
from dlhlp_lib.audio.tools import wav_normalization
from dlhlp_lib.text.utils import remove_punctuation

import Define
from .interface import BasePreprocessor
from .parser import DataParser
from . import template


class KSSPreprocessor(BasePreprocessor):

    def __init__(self, src: str, root: str) -> None:
        super().__init__(src, root)
        self.src_parser = KSSRawParser(src)
        self.data_parser = DataParser(root)

    def parse_raw_process(self, instance: KSSInstance) -> None:
        query = {
            "basename": instance.id,
            "spk": "kss"
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
                "spk": "kss"
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
            self.log("Missing textgrid, start MFA...")
            self.mfa(textgrid_root)

        queries = self.data_parser.get_all_queries()
        if Define.DEBUG:
            queries = queries[:128]
        template.preprocess(self.data_parser, queries)

        phoneset_path = f"{os.path.dirname(__file__)}/../MFA/kss/phoneset.txt"
        if not os.path.exists(phoneset_path):
            self.log("Generate phoneme set...")
            from scripts.collect_phonemes import collect_phonemes, generate_phoneme_set
            phns = collect_phonemes([self.root])
            generate_phoneme_set(phns, phoneset_path)

    def split_dataset(self, cleaned_data_info_path: str):
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        template.split_monospeaker_dataset(self.data_parser, queries, output_dir, val_size=1000)

    def log(self, msg):
        print(f"[KSSPreprocessor]: ", msg)

    # MFA
    def _prepare_mfa_dir(self, mfa_data_dir: Path) -> None:
        if mfa_data_dir.exists():
            return
        
        # 1. construct MFA tool predefined directory structure in "mfa_data_dir"
        self.log(f"Construct MFA directory ({mfa_data_dir})...")
        queries = self.data_parser.get_all_queries()
        target_dir = mfa_data_dir / "kss"
        target_dir.mkdir(parents=True, exist_ok=True)

        # 2. create hard link for wav file
        for query in tqdm(queries):
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
    
    def _generate_dictionary(self, output_path) -> None:
        from scripts.KoG2P.g2p import g2p_ko
        if os.path.exists(output_path):
            return
        self.log(f"Create dictionary...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lexicons = {}
        for instance in self.src_parser.dataset:
            text = remove_punctuation(instance.text)
            for t in text.split(" "):
                if t not in lexicons:
                    lexicons[t] = g2p_ko(t)

        with open(output_path, 'w', encoding="utf-8") as f:
            for k, v in lexicons.items():
                f.write(f"{k}\t{v}\n")
        self.log(f"Write {len(lexicons)} words.")

    def _prepare_mfa_model(self, mfa_data_dir, dictionary_path, output_path) -> None:
        if os.path.exists(output_path):
            return
        self.log(f"Create MFA model...")
        cmd = f"mfa train {mfa_data_dir} {dictionary_path} {output_path} -j 8 -v --clean"
        os.system(cmd)

    def mfa(self, output_path) -> None:
        mfa_data_dir = Path(self.root) / "mfa_data"
        mfa_dir = f"{os.path.dirname(__file__)}/../MFA/kss"
        dictionary_path = f"{mfa_dir}/lexicon.txt"
        model_path = f"{mfa_dir}/acoustic_model.zip"

        self._prepare_mfa_dir(mfa_data_dir)
        self._generate_dictionary(dictionary_path)
        self._prepare_mfa_model(str(mfa_dir), dictionary_path, model_path)

        self.log("Start MFA align!")
        cmd = f"mfa align {str(mfa_data_dir)} {dictionary_path} {model_path} {output_path} -j 8 -v --clean"
        os.system(cmd)
