#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

from typing import Iterator, NamedTuple
from copy import deepcopy
import pathlib, io, dataclasses, sys
import yaml

@dataclasses.dataclass
class EntityDescription:
    id: str
    type: str

class PortAddress(NamedTuple):
    operation_id: str
    port_id: str

@dataclasses.dataclass
class Port:
    id: str
    type: str
    default: dict | None = None

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

    def dump(self) -> None:
        self.save(sys.stdout)

    def __save(self, file: io.IOBase) -> None:
        yaml.dump(self.__data, file)

    def input(self) -> Iterator[EntityDescription]:
        return (Port(**value) for value in self.__data.get("input", ()))

    def output(self) -> Iterator[EntityDescription]:
        return (Port(**value) for value in self.__data.get("output", ()))

    def operations(self) -> Iterator[EntityDescription]:
        return (EntityDescription(id=value["id"], type=value["type"]) for value in self.__data.get("operations", ()))

    def operations_with_dict(self) -> Iterator[tuple[EntityDescription, dict]]:
        return ((EntityDescription(id=value["id"], type=value["type"]), deepcopy(value)) for value in self.__data.get("operations", ()))

    def connections(self) -> Iterator[PortConnection]:
        return (
            PortConnection(input=PortAddress(*value["input"]), output=PortAddress(*value["output"]))
            for value in self.__data.get("connections", ())
            )

    def graph(self, filename: str) -> None:
        import graphviz

        g = graphviz.Digraph(format='png', graph_attr={'dpi': "300"})

        for operation in self.operations():
            g.node(operation.id)

        for connection in self.connections():
            attributes = {"headlabel": connection.input.port_id, "taillabel": connection.output.port_id}
            g.edge(
                connection.input.operation_id, connection.output.operation_id,
                _attributes=attributes)

        g.render(outfile=filename)
