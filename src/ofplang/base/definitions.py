#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import pathlib
import yaml  # type: ignore
from copy import deepcopy
from typing import IO
import io
import importlib.resources

logger = getLogger(__name__)


class Definitions:

    BUILTIN_DEFINITIONS_FILE = str(importlib.resources.files("ofplang.base").joinpath("builtin_definitions.yaml"))

    def __init__(self, file: str | pathlib.Path | IO | None = None, key: str | None = None) -> None:
        self.__data: list = []
        self.load(self.BUILTIN_DEFINITIONS_FILE)
        if file is not None:
            self.load(file, key)

    def load(self, file: str | pathlib.Path | IO, key: str | None = None) -> None:
        if isinstance(file, str):
            with pathlib.Path(file).open() as f:
                self.__load(f, key)
        elif isinstance(file, pathlib.Path):
            with file.open() as f:
                self.__load(f, key)
        elif isinstance(file, io.IOBase):
            self.__load(file, key)
        else:
            raise TypeError(f"Invalid type [{type(file)}]")

    def __load(self, file: IO, key: str | None) -> None:
        data = yaml.load(file, Loader=yaml.Loader)
        if key is not None:
            assert key in data
            data = data[key]
        self.__data.append(data)

    def get_by_name(self, name: str) -> dict:
        for data in reversed(self.__data):
            for x in data:
                if x["name"] == name:
                    return deepcopy(x)
        raise ValueError(f"Unknown name [{name}]")

    def has(self, name: str) -> bool:
        for data in reversed(self.__data):
            for x in data:
                if x["name"] == name:
                    return True
        return False

    def __iter__(self):
        for data in self.__data:
            for x in data:
                yield deepcopy(x)
