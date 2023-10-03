import os
import json
from typing import Dict, List

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import QueryParser
from dlhlp_lib.parsers.IOObjects import WavIO, JSONIO


class DataParser(BaseDataParser):

    def __init__(self, root):
        super().__init__(root)

        self.wav_16000 = Feature(
            QueryParser(f"{self.root}/wav_16000"), WavIO(sr=16000))
        self.label = Feature(
            QueryParser(f"{self.root}/label"), JSONIO(), enable_cache=True)
        
        self.metadata_path = f"{self.root}/data_info.json"

    def _init_structure(self):
        os.makedirs(f"{self.root}/wav_16000", exist_ok=True)
        os.makedirs(f"{self.root}/label", exist_ok=True)
    
    def get_all_queries(self) -> List:
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        return data_infos
    
    def get_feature(self, query: str) -> Feature:
        return getattr(self, query)
