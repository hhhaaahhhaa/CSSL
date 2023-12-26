import numpy as np
from torch.utils.data import Dataset
import random
import json

from .parser import DataParser


class ClassificationDataset(Dataset):
    """
    Simple classification dataset.
    """
    def __init__(self, filename, config):
        self.max_timestep = 128000  # 8s window for 16KHz wav
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

        wav = raw_feat
        if self.max_timestep is not None:
            if wav.shape[0] > self.max_timestep:
                start = random.randint(0, int(wav.shape[0]-self.max_timestep))
                wav = wav[start:start+self.max_timestep]

        sample = {
            "id": query["basename"],
            "label": self.classes.index(query["spk"]),
            "wav": wav,
        }

        return sample
