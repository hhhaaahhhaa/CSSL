import numpy as np
from torch.utils.data import Dataset

from .parser import DataParser


class ClusterDataset(Dataset):
    def __init__(self, filename, config):
        self.data_parser = DataParser(config['data_dir'])

        self.name = config["name"]
        self.unit_name = config["unit_name"]
        self.unit_parser = self.data_parser.units[self.unit_name]

        self.basename, self.speaker = self.process_meta(filename)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        query = {
            "spk": speaker,
            "basename": basename,
        }

        raw_feat = self.data_parser.wav_16000.read_from_query(query)
        idxs = self.unit_parser.clusters.read_from_query(query)
        idxs = np.array([int(x) for x in idxs.split(" ")])
        
        sample = {
            "id": basename,
            "speaker": -1,
            "idxs": idxs,
            "wav": raw_feat,
        }

        return sample

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
            return name, speaker
