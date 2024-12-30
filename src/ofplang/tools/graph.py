from ofplang.base.definitions import Definitions
from ofplang.base.protocol import PortAddress, Protocol, Port, EntityDescription
from ofplang.base.runner import UntypedProcess

from collections import namedtuple
from typing import Any, IO
import sys
import re
import io
import copy
import pathlib
import yaml

class ProcessProxy:

    def __init__(self, id: str, definition: dict, graph: "ProtocolGraph") -> None:
        self.__process = UntypedProcess(EntityDescription(id, definition["name"]), definition)
        self.__graph = graph

    def __call__(self, *args, **kwargs) -> Any:
        connections = []
        for address, port in self.__process.input():
            if address.port_id in kwargs:
                connections.append((kwargs[address.port_id], address))
                
        for input_address, output_address in connections:
            self.__graph.add_connection(input_address, output_address)

        # See https://github.com/python/mypy/issues/848
        cls = namedtuple("EntityProxy", [address.port_id for address, port in self.__process.output()])  # type: ignore
        ret = cls(**{address.port_id: address for address, port in self.__process.output()})
        # print(self.__process.type, connections, ret)
        return ret

class IOProxy:
    __slots__ = ('_IOProxy__name', '_IOProxy__graph')

    def __init__(self, name: str, graph: "ProtocolGraph") -> None:
        self.__name = name
        self.__graph = graph

    def __setattr__(self, name, value) -> None:
        if name in object.__getattribute__(self, '__slots__'):
            object.__setattr__(self, name, value)
        else:
            if not self.__graph.has_output(name):
                self.__graph.add_output(name)
            self.__graph.add_connection(value, PortAddress("output", name))
        
    def __getattr__(self, name) -> Any:
        if not self.__graph.has_input(name):
            self.__graph.add_input(name)
        return PortAddress(self.__name, name)
 
def generate_process_id_prefix_from_type(type_name: str) -> str:
    # Split at the upper case letters
    elements = re.findall('(?:[A-Z]*[a-zA-Z][^0-9A-Z]*)|(?:[0-9]+)', type_name)
    prefix = "_".join(element.lower() for element in elements)
    return f"{prefix}_"

class ProtocolGraph:

    def __init__(self, definitions: str | Definitions, *, name: str | None = None, author: str | None = None, description: str | None = None) -> None:
        definitions = definitions if isinstance(definitions, Definitions) else Definitions(definitions)
        self.__definitions = definitions

        self.__data: dict[str, Any] = {"contents": {"input": [], "output": [], "processes": [], "connections": []}}
        self.__contents = self.__data["contents"]
        if name is not None:
            self.__data["name"] = name
        if author is not None:
            self.__data["author"] = author
        if description is not None:
            self.__data["description"] = description

    @property
    def input(self) -> IOProxy:
        return IOProxy("input", self)

    @property
    def output(self) -> IOProxy:
        return IOProxy("output", self)

    def __getattr__(self, name) -> Any:
        if self.__definitions.has(name):
            definition = self.__definitions.get_by_name(name)
            name = self.add_process(definition)
            return ProcessProxy(name, definition, self)
        return object.__getattribute__(self, name)
    
    def has_input(self, id: str) -> bool:
        return any(input["id"] == id for input in self.__contents["input"])
    
    def add_input(self, id: str, type: str = "") -> None:
        self.__contents["input"].append({"id": id, "type": type})

    def has_output(self, id: str) -> bool:
        return any(input["id"] == id for input in self.__contents["output"])
    
    def add_output(self, id: str, type: str = "") -> None:
        self.__contents["output"].append({"id": id, "type": type})

    def add_process(self, definition: dict) -> str:
        #XXX: This would be slow in the case with many processes.
        prefix = generate_process_id_prefix_from_type(definition["name"])
        i = 0
        while True:
            i += 1
            newid = f"{prefix}{i:d}"
            if all(p["id"] != newid for p in self.__contents["processes"]):
                break

        self.__contents["processes"].append({"id": newid, "type": definition["name"]})
        return newid
    
    def add_connection(self, input: PortAddress, output: PortAddress) -> None:
        self.__contents["connections"].append({"input": [input.process_id, input.port_id], "output": [output.process_id, output.port_id]})

    @staticmethod
    def __set_io_types(data: dict, definitions: Definitions) -> dict:
        contents = data["contents"]

        text = io.StringIO()
        yaml.dump(data, text)
        with io.StringIO(text.getvalue()) as f:
            protocol = Protocol(f)

        #XXX: Better to use Model?
        port_types: dict[PortAddress, Port] = {}
        for entity in protocol.processes():
            process = UntypedProcess(entity, definitions.get_by_name(entity.type))
            port_types.update(process.input())
            port_types.update(process.output())
        
        connections = list(protocol.connections())
        
        for i, port in enumerate(contents["output"]):
            if port["type"] == "":
                address = PortAddress("output", port["id"])
                partners = [connection.input for connection in connections if connection.output == address]
                partner_types = [port_types[another] for another in partners]
                assert len(partner_types) > 0, address
                #XXX: Do something when multiple partners exists
                contents["output"][i]["type"] = partner_types[0].type

        for i, port in enumerate(contents["input"]):
            if port["type"] == "":
                address = PortAddress("input", port["id"])
                partners = [connection.output for connection in connections if connection.input == address]
                partner_types = [port_types[another] for another in partners]
                assert len(partner_types) > 0, address
                #XXX: Do something when multiple partners exists
                contents["input"][i]["type"] = partner_types[0].type
    
        return data

    def save(self, file: str | pathlib.Path | IO) -> None:
        data = copy.deepcopy(self.__data)
        data = self.__set_io_types(data, self.__definitions)

        if "name" not in data:
            if isinstance(file, str):
                data["name"] = pathlib.Path(file).stem
            elif isinstance(file, pathlib.Path):
                data["name"] = file.stem
            else:
                pass  #XXX: Show warning

        if isinstance(file, str):
            with pathlib.Path(file).open('w') as f:
                yaml.dump(data, f)
        elif isinstance(file, pathlib.Path):
            with file.open('w') as f:
                yaml.dump(data, f)
        elif isinstance(file, io.IOBase):  #XXX: isinstance(sys.stdout, IO) doesn't work.
            yaml.dump(data, file)
        else:
            raise TypeError(f"Invalid type [{type(file)}]")

    def dump(self) -> None:
        self.save(sys.stdout)

    def __save(self, file: IO) -> None:
        yaml.dump(self.__data, file)