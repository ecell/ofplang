#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from typing import NamedTuple, IO, Any
from collections.abc import Iterator
from copy import deepcopy
import pathlib
import dataclasses
import sys
import hashlib
import io
import yaml  # type: ignore

logger = getLogger(__name__)


@dataclasses.dataclass
class EntityDescription:
    id: str
    type: str

class PortAddress(NamedTuple):
    process_id: str
    port_id: str

@dataclasses.dataclass
class Port:
    id: str
    type: str
    default: dict[str, Any] | None = None

@dataclasses.dataclass
class PortConnection:
    input: PortAddress
    output: PortAddress

class Protocol:

    def __init__(self, file: str | pathlib.Path | IO | None) -> None:
        if file is not None:
            self.load(file)

    def load(self, file: str | pathlib.Path | IO | None) -> None:
        if isinstance(file, str):
            with pathlib.Path(file).open() as f:
                self.__load(f)
        elif isinstance(file, pathlib.Path):
            with file.open() as f:
                self.__load(f)
        elif isinstance(file, io.IOBase):
            self.__load(file)
        else:
            raise TypeError(f"Invalid type [{type(file)}]")

    def __load(self, file: IO) -> None:
        self.__data = yaml.load(file, Loader=yaml.Loader)
        if "contents" not in self.__data:
            raise ValueError("'contents' is required in a protocol.")
        self.__contents = self.__data["contents"]

    def save(self, file: str | pathlib.Path | IO) -> None:
        if isinstance(file, str):
            with pathlib.Path(file).open('w') as f:
                self.__save(f)
        elif isinstance(file, pathlib.Path):
            with file.open('w') as f:
                self.__save(f)
        elif isinstance(file, io.IOBase):  #XXX: isinstance(sys.stdout, IO) doesn't work.
            self.__save(file)
        else:
            raise TypeError(f"Invalid type [{type(file)}]")

    def md5(self) -> str:
        f = io.StringIO()
        self.__save(f)
        s = f.getvalue().encode('utf-8')
        md5_hash = hashlib.md5()
        md5_hash.update(s)
        return md5_hash.hexdigest()

    def dump(self) -> None:
        self.save(sys.stdout)

    def __save(self, file: IO) -> None:
        yaml.dump(self.__data, file)

    def input(self) -> Iterator[Port]:
        return (Port(**value) for value in self.__contents.get("input", ()))

    def output(self) -> Iterator[Port]:
        return (Port(**value) for value in self.__contents.get("output", ()))

    def processes(self) -> Iterator[EntityDescription]:
        return (EntityDescription(id=value["id"], type=value["type"]) for value in self.__contents.get("processes", ()))

    def processes_with_dict(self) -> Iterator[tuple[EntityDescription, dict]]:
        return ((EntityDescription(id=value["id"], type=value["type"]), deepcopy(value)) for value in self.__contents.get("processes", ()))

    def connections(self) -> Iterator[PortConnection]:
        return (
            PortConnection(input=PortAddress(*value["input"]), output=PortAddress(*value["output"]))
            for value in self.__contents.get("connections", ())
            )

    def graph(self, filename: str) -> None:
        import graphviz  # type: ignore

        g = graphviz.Digraph(format='png', graph_attr={'dpi': "300"})

        for process in self.processes():
            g.node(process.id)

        for connection in self.connections():
            attributes = {"headlabel": connection.output.port_id, "taillabel": connection.input.port_id}
            g.edge(
                connection.input.process_id, connection.output.process_id,
                _attributes=attributes)

        g.render(outfile=filename)
