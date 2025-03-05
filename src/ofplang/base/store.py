#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import dataclasses
import uuid
from pathlib import Path
from abc import ABCMeta, abstractmethod

logger = getLogger(__name__)


@dataclasses.dataclass
class Location:
    id: str
    uri: str

class Handler:

    def __init__(self):
        pass

    def create_run(self, id, metadata) -> None:
        pass

    def create_process(self, id, metadata) -> None:
        pass

    def create_operation(self, id, metadata) -> None:
        pass

    def update_run(self, id, metadata) -> None:
        pass

    def update_process(self, id, metadata) -> None:
        pass

    def update_operation(self, id, metadata) -> None:
        pass

class Store(metaclass=ABCMeta):

    def __init__(self) -> None:
        self.__handlers: list[Handler] = []

    def add_handler(self, handler: Handler) -> None:
        self.__handlers.append(handler)
    
    @property
    def handlers(self) -> list[Handler]:
        return self.__handlers

    @abstractmethod
    def create_run(self, metadata) -> str:
        """"""

    @abstractmethod
    def create_process(self, metadata) -> str:
        """"""

    @abstractmethod
    def create_operation(self, metadata) -> str:
        """"""

    @abstractmethod
    def get_run_uri(self, id: str) -> str:
        """"""

    @abstractmethod
    def get_process_uri(self, id: str) -> str:
        """"""

    @abstractmethod
    def get_operation_uri(self, id: str) -> str:
        """"""

    @abstractmethod
    def update_run(self, id: str, metadata) -> None:
        """"""
        
    @abstractmethod
    def update_process(self, id: str, metadata) -> None:
        """"""

    @abstractmethod
    def update_operation(self, id: str, metadata) -> None:
        """"""

class FileStore(Store):

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.__root = path or (Path.cwd() / 'experiments')
    
    def __create(self, prefix: str, *, id: str | None = None) -> str:
        id = id or str(uuid.uuid4())
        path = self.__get_path(prefix, id)
        assert not path.is_dir()
        # path.mkdir(parents=True)
        return id
    
    def __get_path(self, prefix: str, id: str) -> Path:
        path = self.__root / f'{prefix}/{id}'
        return path
    
    def create_run(self, metadata) -> str:
        id = self.__create('runs')
        [handler.create_run(id, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]
        return id
    
    def create_process(self, metadata) -> str:
        id = self.__create('processes')
        [handler.create_process(id, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]
        return id
    
    def create_operation(self, metadata) -> str:
        id = self.__create('operations')
        [handler.create_operation(id, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]
        return id

    def update_run(self, id: str, metadata) -> None:
        [handler.update_run(id, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]

    def update_process(self, id: str, metadata) -> None:
        [handler.update_process(id, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]

    def update_operation(self, id: str, metadata) -> None:
        [handler.update_operation(id, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]

    def get_run_uri(self, id: str) -> str:
        return self.__get_path('runs', id).as_uri()

    def get_process_uri(self, id: str) -> str:
        return self.__get_path('processes', id).as_uri()

    def get_operation_uri(self, id: str) -> str:
        return self.__get_path('operations', id).as_uri()
