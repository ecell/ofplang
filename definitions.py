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

    def get_by_id(self, id: str) -> dict:
        for x in self.__data:
            if x["id"] == id:
                return x.copy()
        raise ValueError(f"Unknown id [{id}]")

    def has(self, id: str) -> bool:
        for x in self.__data:
            if x["id"] == id:
                return True
        return False
