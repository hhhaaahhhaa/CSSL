import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils.tool import segment2duration
from dlhlp_lib.utils.numeric import numpy_exist_nan

from lightning.text import text_to_sequence
from lightning.text.define import LANG_ID2SYMBOLS
from .parser import DataParser


class PRDataset(Dataset):
    """
    Phoneme recognition dataset.
    """
    def __init__(self, filename, config):
        self.data_parser = DataParser(config['data_dir'])

        self.name = config["name"]
        self.unit_name = config.get("unit_name", "mfa")
        self.lang_id = config["lang_id"]
        self.cleaners = config["text_cleaners"]
        self.unit_parser = self.data_parser.units[self.unit_name]

        with open(filename, "r", encoding="utf-8") as f:  # Unify IO interface
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        query = self.data_infos[idx]
        basename = query["basename"]
        speaker = query["spk"]

        phonemes = self.unit_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        phonemes = f"{{{phonemes}}}"

        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)

        sample = {
            "id": basename,
            "speaker": -1,
            "text": text,
            "raw_text": raw_text,
            "wav": raw_feat,
            "lang_id": self.lang_id,
            "n_symbols": len(LANG_ID2SYMBOLS[self.lang_id]),
        }

        return sample


class FramewisePRDataset(Dataset):
    """
    Phoneme recognition dataset framewise version.
    """
    def __init__(self, filename, data_parser: DataParser, config):
        self.data_parser = data_parser

        self.name = config["name"]
        self.unit_name = config["unit_name"]
        self.lang_id = config["lang_id"]
        self.cleaners = config["text_cleaners"]
        self.unit_parser = self.data_parser.units[self.unit_name]

        self.fp = config["fp"]

        with open(filename, "r", encoding="utf-8") as f:  # Unify IO interface
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        query = self.data_infos[idx]
        basename = query["basename"]
        speaker = query["spk"]

        phonemes = self.unit_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        phonemes = f"{{{phonemes}}}"

        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))

        segment = self.unit_parser.segment.read_from_query(query)
        avg_frames = segment2duration(segment, fp=self.fp)
        duration = np.array(avg_frames)
        
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration)
        except:
            print(query)
            print(text)
            print(len(text), len(phonemes), len(duration))
            raise

        expanded_text = np.repeat(text, duration)
        raw_feat = self.data_parser.wav_trim_16000.read_from_query(query)

        sample = {
            "id": basename,
            "speaker": -1,
            "text": text,
            "expanded_text": expanded_text,
            "raw_text": raw_text,
            "wav": raw_feat,
            "duration": duration,
            "lang_id": self.lang_id,
            "n_symbols": len(LANG_ID2SYMBOLS[self.lang_id]),
        }

        return sample
