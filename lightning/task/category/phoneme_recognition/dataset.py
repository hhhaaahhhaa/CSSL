import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import NestSFQueryParser
from dlhlp_lib.parsers.IOObjects import WavIO, TextIO

from lightning.text import text_to_sequence


class PRDataset(Dataset):
    """
    Phoneme recognition dataset.
    """
    def __init__(self, filename, config, is_test=False):
        self.config = config
        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.cleaners = config["text_cleaners"]
        self.is_test = is_test

        with open(filename, "r", encoding="utf-8") as f:
            self.data_infos = json.load(f)

        root = config['data_dir']
        self.wav_feature = Feature(
            parser=NestSFQueryParser(f"{root}/{config['feature']['wav']}"),
            io=WavIO(sr=16000)
        )
        self.phoneme_feature = Feature(
            parser=NestSFQueryParser(f"{root}/{config['feature']['phoneme']}"),
            io=TextIO()
        )
        self.text_feature = Feature(
            parser=NestSFQueryParser(f"{root}/{config['feature']['text']}"),
            io=TextIO()
        )

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        query = self.data_infos[idx]
        basename = query["basename"]

        phonemes = self.phoneme_feature.read_from_query(query)
        raw_text = self.text_feature.read_from_query(query)
        phonemes = f"{{{phonemes}}}"

        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        raw_feat = self.wav_feature.read_from_query(query)

        sample = {
            "id": basename,
            "speaker": -1,
            "text": text,
            "raw_text": raw_text,
            "wav": raw_feat,
        }

        if self.is_test:
            sample["phoneme"] = self.phoneme_feature.read_from_query(query)

        return sample
