import numpy as np
from torch.utils.data import Dataset
import json

from .Parsers.parser import DataParser


class ClassificationDataset(Dataset):
    """
    Simple classification dataset.
    """
    def __init__(self, filename, config):
        self.data_parser = DataParser(config['data_dir'])

        self.name = config["name"]
        self.basename, self.labels = self.process_meta(filename)

        with open(f"{self.data_parser.root}/classes.json", 'r') as f:
            self.classes = json.load(f)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        query = {
            "basename": basename,
        }

        raw_feat = self.data_parser.wav_16000.read_from_query(query)
        label = self.data_parser.label.read_from_query(query)

        sample = {
            "id": basename,
            "label": self.classes.index(label["class"]),
            "wav": raw_feat,
        }

        return sample

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            labels = []
            for line in f.readlines():
                n, c = line.strip("\n").split("|")
                name.append(n)
                labels.append(c)
            return name, labels
