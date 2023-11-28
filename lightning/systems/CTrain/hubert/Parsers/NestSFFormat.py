import os
import json
from typing import Dict, List

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import NestSFQueryParser
from dlhlp_lib.parsers.IOObjects import NumpyIO, WavIO, TextGridIO, TextIO, JSONIO


class UnitDataParser(BaseDataParser):
    def __init__(self, root):
        super().__init__(root)
        self.codes = Feature(
            NestSFQueryParser(f"{self.root}/codes"), TextIO(), enable_cache=True)

    def _init_structure(self, *args, **kwargs) -> None:
        pass
    
    def get_feature(self, query: str) -> Feature:
        return getattr(self, query)


class DataParser(BaseDataParser):

    units: Dict[str, UnitDataParser]

    def __init__(self, root):
        super().__init__(root)
        self.__init_units()

        self.wav_16000 = Feature(
            NestSFQueryParser(f"{self.root}/wav_16000"), WavIO(sr=16000))
    
        self.metadata_path = f"{self.root}/data_info.json"

    def _init_structure(self):
        os.makedirs(f"{self.root}/wav_16000", exist_ok=True)
    
    def get_all_queries(self) -> List:
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        return data_infos
    
    def __init_units(self):
        self.units = {}
        os.makedirs(f"{self.root}/units", exist_ok=True)
        unit_names = os.listdir(f"{self.root}/units")
        for unit_name in unit_names:
            self.units[unit_name] = UnitDataParser(f"{self.root}/units/{unit_name}")

    def create_unit_feature(self, unit_name):
        if unit_name not in self.units:
            self.units[unit_name] = UnitDataParser(f"{self.root}/units/{unit_name}")

    def get_feature(self, query: str) -> Feature:
        if "/" not in query:
            return getattr(self, query)
        prefix, subquery = query.split("/", 1)
        if prefix == "units":
            unit_name, subquery = subquery.split("/", 1)
            return self.units[unit_name].get_feature(subquery)
        else:
            raise NotImplementedError
