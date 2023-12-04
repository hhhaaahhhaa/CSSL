import gc
import torch
from tqdm import tqdm
from typing import List

from dlhlp_lib.s3prl import S3PRLExtractor
from dlhlp_lib.utils.numeric import torch_exist_nan

from parser import DataParser


class BaseChecker(object):
    def __init__(self) -> None:
        pass
    
    def _check(self) -> None:
        raise NotImplementedError
    
    def check(self, query) -> bool:
        try:
            self._check(query)
        except:
            return False
        return True
    
    def filter(self, queries) -> List:
        filtered = []
        for query in tqdm(queries):
            if self.check(query):
                filtered.append(query)
        print(f"{len(queries)} => {len(filtered)}")
        return filtered


class LengthChecker(BaseChecker):
    def __init__(self, data_parser: DataParser, mi=1, mx=15):
        self.data_parser = data_parser
        self.data_parser.units["mfa"].segment.read_all()
        self.mi = mi
        self.mx = mx

    def _check(self, query) -> None:
        segment = self.data_parser.units["mfa"].segment.read_from_query(query)
        l = segment[-1][1] - segment[0][0]
        assert self.mi <= l and l <= self.mx

    def filter(self, queries) -> List:
        print(f"Check length ({self.mi}s to {self.mx}s)...")
        return super().filter(queries)


class SSLFeatureChecker(BaseChecker):
    def __init__(self, s3prl_name: str, data_parser: DataParser):
        self.data_parser = data_parser
        self.extractor = S3PRLExtractor(s3prl_name)

    def _check(self, query) -> bool:
        wav_path = self.data_parser.wav_trim_16000.read_filename(query, raw=True)
        with torch.no_grad():
            repr, _ = self.extractor.extract_from_paths([wav_path])
            assert not torch_exist_nan(repr)

    def filter(self, queries) -> List:
        gc.collect()
        print(f"Check SSL feature({self.extractor.name})...")
        self.extractor.cuda()
        res = super().filter(queries)
        self.extractor.cpu()

        return res


class UnknownTokenChecker(BaseChecker):
    def __init__(self, data_parser: DataParser, unk_list=[]):
        self.unk_list = unk_list
        self.data_parser = data_parser
        self.data_parser.units["mfa"].phoneme.read_all()

    def _check(self, query) -> bool:
        phoneme = self.data_parser.units["mfa"].phoneme.read_from_query(query)
        for token in self.unk_list:
            assert token not in phoneme.split(" ")

    def filter(self, queries) -> List:
        print("Check unknown tokens (spn)...")
        return super().filter(queries)
