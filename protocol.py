#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

from typing import Iterator, NamedTuple
import pathlib, io, dataclasses
import yaml

@dataclasses.dataclass
class Entity:
    id: str
    type: str

class PortAddress(NamedTuple):
    operation_id: str
    port_id: str

@dataclasses.dataclass
class PortConnection:
    input: PortAddress
    output: PortAddress

class Protocol:

    def __init__(self, file: str | pathlib.PurePath | io.IOBase | None) -> None:
        self.__data = None

        if file is not None:
            self.load(file)

    def load(self, file: str | pathlib.PurePath | io.IOBase | None) -> None:
        if isinstance(file, str):
            with pathlib.Path(file).open() as f:
                self.__load(f)
        elif isinstance(file, pathlib.PurePath):
            with file.open() as f:
                self.__load(f)
        elif isinstance(file, io.IOBase):
            self.__load(file)
        else:
            raise TypeError(f"Invalid type [{type(file)}]")
        
    def __load(self, file: io.IOBase) -> None:
        self.__data = yaml.load(file, Loader=yaml.Loader)

    def save(self, file: str | pathlib.PurePath | io.IOBase) -> None:
        if isinstance(file, str):
            with pathlib.Path(file).open('w') as f:
                self.__save(f)
        elif isinstance(file, pathlib.PurePath):
            with file.open('w') as f:
                self.__save(f)
        elif isinstance(file, io.IOBase):
            self.__save(file)
        else:
            raise TypeError(f"Invalid type [{type(file)}]")
        
    def __save(self, file: io.IOBase) -> None:
        yaml.dump(self.__data, file)

    def input(self) -> Iterator[Entity]:
        return (Entity(**value) for value in self.__data.get("input", ()))

    def output(self) -> Iterator[Entity]:
        return (Entity(**value) for value in self.__data.get("output", ()))

    def operations(self) -> Iterator[Entity]:
        return (Entity(**value) for value in self.__data.get("operations", ()))

    def connections(self) -> Iterator[PortConnection]:
        return (
            PortConnection(input=PortAddress(*value["input"]), output=PortAddress(*value["output"]))
            for value in self.__data.get("connections", ())
            )
