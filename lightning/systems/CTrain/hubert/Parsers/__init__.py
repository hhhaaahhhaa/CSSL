from typing import Type

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from . import NestSFFormat
from . import PlainFormat


PARSER = {
    "plain": PlainFormat.DataParser,
    "nest-sf": NestSFFormat.DataParser,
}


def get_parser(parser_name: str) -> Type[BaseDataParser]:
    return PARSER[parser_name]
