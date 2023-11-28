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


class SSLFeatureChecker(BaseChecker):
    def __init__(self, s3prl_name: str, data_parser: DataParser):
        self.data_parser = data_parser
        self.extractor = S3PRLExtractor(s3prl_name)

    def _check(self, query) -> bool:
        wav_path = self.data_parser.wav_16000.read_filename(query, raw=True)
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
