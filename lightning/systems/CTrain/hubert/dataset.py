import numpy as np
from torch.utils.data import Dataset
import json

from .Parsers import get_parser


class CodeDataset(Dataset):
    def __init__(self, filename, config):
        self.data_parser = get_parser(config["parser"])(config['data_dir'])

        self.name = config["name"]
        self.unit_name = config["unit_name"]
        self.unit_parser = self.data_parser.units[self.unit_name]
        self.unit_type = config["unit_type"]
        with open(filename, "r", encoding="utf-8") as f:  # Unify IO interface
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        query = self.data_infos[idx]

        raw_feat = self.data_parser.wav_16000.read_from_query(query)
        idxs = self.unit_parser.codes.read_from_query(query)
        idxs = np.array([int(x) for x in idxs.split(" ")])
        
        sample = {
            "id": query['basename'],
            "speaker": query.get("spk", -1),
            "idxs": idxs,
            "wav": raw_feat,
            "unit_type": self.unit_type,  # we may need mix unit training in future
        }

        return sample
