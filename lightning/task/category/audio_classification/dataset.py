import numpy as np
from torch.utils.data import Dataset
import json
import random

from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import QueryParser
from dlhlp_lib.parsers.IOObjects import WavIO, JSONIO


class ClassificationDataset(Dataset):
    def __init__(self, filename, config, mode="train"):
        self.config = config
        self.name = config["name"]
        self.mode = mode
        self.max_timestep = 16000 * 15  # 15s window for 16KHz wav

        with open(filename, "r", encoding="utf-8") as f:
            self.data_infos = json.load(f)

        root = config['data_dir']
        self.wav_feature = Feature(
            parser=QueryParser(f"{root}/{config['feature']['wav']}"),
            io=WavIO(sr=16000)
        )
        self.label_feature = Feature(
            parser=QueryParser(f"{root}/{config['feature']['label']}"),
            io=JSONIO()
        )
        with open(f"{root}/classes.json", 'r') as f:
            self.classes = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        query = self.data_infos[idx]

        wav = self.wav_feature.read_from_query(query)
        label = self.label_feature.read_from_query(query)

        if self.mode == "train":
            if wav.shape[0] > self.max_timestep:
                start = random.randint(0, int(wav.shape[0]-self.max_timestep))
                wav = wav[start:start+self.max_timestep]

        sample = {
            "id": query["basename"],
            "label": self.classes.index(label["class"]),
            "class": label["class"],
            "wav": wav,
        }

        return sample
