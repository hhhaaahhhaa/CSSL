from typing import Type

from dlhlp_lib.parsers.Interfaces import BasePreprocessor

from .esc50 import ESC50Preprocessor
from .urbansound8k import UrbanSound8KPreprocessor
from .fma import FMAMusicPreprocessor
from .vocal_set import VocalSetPreprocessor


PREPROCESSORS = {
    "ESC50": ESC50Preprocessor,
    "UrbanSound8K": UrbanSound8KPreprocessor,
    "FMA": FMAMusicPreprocessor,
    "VocalSet": VocalSetPreprocessor,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]
