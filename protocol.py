#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

from typing import Union, Optional
import pathlib, io
import yaml


class Protocol:

    def __init__(self, file: Optional[Union[str, pathlib.PurePath, io.IOBase]]) -> None:
        self.__data = None

        if file is not None:
            self.load(file)

    def load(self, file: Union[str, pathlib.PurePath, io.IOBase]) -> None:
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

    def save(self, file: Union[str, pathlib.PurePath, io.IOBase]) -> None:
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
