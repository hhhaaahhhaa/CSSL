from pathlib import Path
import abc


class BasePreprocessor(metaclass=abc.ABCMeta):
    def __init__(self, src: str, root: str, *args, **kwargs) -> None:
        self.src = src
        self.root = root

    @abc.abstractmethod
    def parse_raw(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def prepare_mfa(self, mfa_data_dir: Path) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def mfa(self, mfa_data_dir: Path) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def preprocess(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def split_dataset(self, *args, **kwargs) -> None:
        raise NotImplementedError
