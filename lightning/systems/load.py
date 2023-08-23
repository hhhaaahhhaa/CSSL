from typing import Type

from .system import System
from .s3prl import S3PRLWrapper


SYSTEM_CTRAIN = {
    "ctrain/hubert": None,
}


SYSTEM = {
    **SYSTEM_CTRAIN,
}


def get_system(system_name: str) -> Type[System]:
    return SYSTEM[system_name]


def load_system(system_name, ckpt_file):
    if system_name not in ["hubert", "wav2vec2"]:
        assert ckpt_file is not None
        system = get_system(system_name)
        upstream = system.load_from_checkpoint(ckpt_file)
        return upstream
    
    # s3prl upstreams
    return S3PRLWrapper(system_name)
