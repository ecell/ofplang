#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import pathlib, io
import yaml


class Definitions:

    def __init__(self, file: str | pathlib.PurePath | io.IOBase | None) -> None:
        self.__data = None

        if file is not None:
            self.load(file)

    def load(self, file: str | pathlib.PurePath | io.IOBase) -> None:
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

    def get_by_name(self, name: str) -> dict:
        for x in self.__data:
            if x["name"] == name:
                return x.copy()  #XXX: deepcopy?
        raise ValueError(f"Unknown name [{name}]")

    def has(self, name: str) -> bool:
        for x in self.__data:
            if x["name"] == name:
                return True
        return False

    def __iter__(self):
        for x in self.__data:
            yield x.copy()  #XXX: deepcopy?