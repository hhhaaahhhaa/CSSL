from typing import Type

from dlhlp_lib.parsers.Interfaces import BasePreprocessor

from .voxceleb1 import VoxCeleb1Preprocessor


PREPROCESSORS = {
    "VoxCeleb1": VoxCeleb1Preprocessor,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]
