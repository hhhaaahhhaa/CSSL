from typing import Type

from dlhlp_lib.parsers.Interfaces import BasePreprocessor

from .voxceleb1 import Voxceleb1Preprocessor


PREPROCESSORS = {
    "Voxceleb1": Voxceleb1Preprocessor,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]
