from typing import Type

from .base import BasePreprocessor
from .esc50 import ESC50Preprocessor


PREPROCESSORS = {
    "ESC50": ESC50Preprocessor,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]
