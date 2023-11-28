import numpy as np
from torch.utils.data import Dataset
import json

from .parser import DataParser


class ClassificationDataset(Dataset):
    """
    Simple classification dataset.
    """
    def __init__(self, filename, config):
        self.data_parser = DataParser(config['data_dir'])

        self.name = config["name"]
        with open(filename, "r", encoding="utf-8") as f:  # Unify IO interface
            self.data_infos = json.load(f)
        with open(f"{self.data_parser.root}/speakers.json", 'r') as f:
            self.classes = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        query = self.data_infos[idx]

        raw_feat = self.data_parser.wav_16000.read_from_query(query)
        sample = {
            "id": query["basename"],
            "label": self.classes.index(query["spk"]),
            "wav": raw_feat,
        }

        return sample
