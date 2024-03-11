#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import dataclasses, pathlib, io
from collections import defaultdict, deque, OrderedDict
from typing import Any, Iterator, Callable
from enum import IntEnum, auto

from protocol import PortAddress, Port, Protocol, PortConnection, Entity
from definitions import Definitions
from validate import check_definitions, check_protocol

class Operation:

    def __init__(self, id: str, definition: dict) -> None:
        self.__id = id
        self.__definition = definition

    def input(self) -> Iterator[tuple[PortAddress, Port]]:
        input = {
            PortAddress(self.__id, port["id"]): Port(**port)
            for port in self.__definition.get("input", [])}
        return input.items()

    def output(self) -> Iterator[tuple[PortAddress, Port]]:
        output = {
            PortAddress(self.__id, port["id"]): Port(**port)
            for port in self.__definition.get("output", [])}
        return output.items()

    def asentity(self) -> Entity:
        return Entity(self.__id, self.__definition["name"])

    @property
    def id(self) -> str:
        return self.__id

    @property
    def definition(self) -> dict:
        return self.__definition.copy()  # deepcopy

class Model:

    def __init__(self, protocol: Protocol, definitions: Definitions) -> None:
        self.__protocol = protocol
        self.__definitions = definitions
        self.__load()
    
    def __load(self) -> None:
        self.__operations = OrderedDict([
            (operation.id, Operation(operation.id, self.__definitions.get_by_name(operation.type)))
            for operation in self.__protocol.operations()])

    def get_by_id(self, id):
        return self.__operations[id]

    def connections(self) -> Iterator[PortConnection]:
        return self.__protocol.connections()
    
    def operations(self) -> Iterator[Operation]:
        return self.__operations.values()

    def input(self) -> Iterator[tuple[PortAddress, Port]]:
        #XXX: default?
        return ((PortAddress("input", port.id), port) for port in self.__protocol.input())

    def output(self) -> Iterator[tuple[PortAddress, Port]]:
        return ((PortAddress("output", port.id), port) for port in self.__protocol.output())

@dataclasses.dataclass
class Token:
    address: PortAddress
    value: str

class StatusEnum(IntEnum):
    ACTIVE = auto()
    INACTIVE = auto()

def default_callback(runner: 'Runner', tasks: list[tuple[Entity, dict]]) -> None:
    for operation, input_tokens in tasks:
        logger.info(f"default_callback: {(operation, input_tokens)}")

class Runner:

    def __init__(self, protocol: Protocol, definitions: Definitions) -> None:
        self.__model = Model(protocol, definitions)
        self.__tokens = defaultdict(deque)

        self.__operation_status = {operation.id: StatusEnum.INACTIVE for operation in self.__model.operations()}
        self.__callbacks = [default_callback]

    def set_callback(self, func: Callable[["Runner", list[tuple[Entity, dict]]], None]) -> None:
        self.__callbacks = [func]

    def add_callback(self, func: Callable[["Runner", list[tuple[Entity, dict]]], None]) -> None:
        self.__callbacks.append(func)

    def activate_all(self) -> None:
        for operation_id in self.__operation_status.keys():
            self.__operation_status[operation_id] = StatusEnum.ACTIVE
    
    def deactivate(self, id: str) -> None:
        assert id in self.__operation_status
        self.__operation_status[id] = StatusEnum.INACTIVE

    def add_token(self, token: Token) -> None:
        self.__tokens[token.address].append(token)

    def add_tokens(self, tokens: list[Token]) -> None:
        for token in tokens:
            self.add_token(token)

    def transmit_token(self) -> None:
        new_tokens = defaultdict(deque)
        for connection in self.__model.connections():
            for token in self.__tokens[connection.input]:
                new_token = Token(connection.output, token.value)
                new_tokens[new_token.address].append(new_token)
        for connection in self.__model.connections():
            del self.__tokens[connection.input]
        for address in new_tokens:
            self.__tokens[address].extend(new_tokens[address])

    def list_tasks(self, max_iterations=1) -> list[tuple[str, dict[str, Any]]]:
        tasks = []
        for operation in self.__model.operations():
            if self.__operation_status[operation.id] is not StatusEnum.ACTIVE:
                continue
            for _ in range(max_iterations):
                if any(port.default is None and not self.has_token(address) for address, port in operation.input()):
                    continue
                # pop tokens here
                input_tokens = {
                    address: (self.__tokens[address].pop() if self.has_token(address) else Token(address, port.default))
                    for address, port in operation.input()}
                tasks.append((operation.asentity(), {address.port_id: token.value for address, token in input_tokens.items()}))
        return tasks

    def run(self, inputs: dict) -> dict:
        self.clear_tokens()
        for key, value in inputs.items():
            self.add_token(Token(PortAddress("input", key), {"value": value}))

        self.activate_all()

        while self.num_tokens() > 0:
            self.transmit_token()
            tasks = self.list_tasks()
            tuple(callback(self, tasks) for callback in self.__callbacks)
            if all(self.has_token(address) for address, _ in self.model.output()):
                break
        else:
            raise RuntimeError("Never get here.")
        
        outputs = {address.port_id: self.__tokens[address].pop().value for address, _ in self.model.output()}
        # finalize
        return outputs

    def _tokens(self):
        print({k: [x.value for x in v] for k, v in self.__tokens.items() if len(v) > 0})

    @property
    def model(self):
        return self.__model
    
    def num_tokens(self):
        return len(self.__tokens)
  
    def has_token(self, address: PortAddress) -> bool:
        return len(self.__tokens[address]) > 0
      
    def clear_tokens(self) -> None:
        self.__tokens.clear()

def run(
        inputs: dict, 
        protocol: Protocol | str | pathlib.PurePath | io.IOBase, 
        definitions: Definitions | str | pathlib.PurePath | io.IOBase, 
        callback: Callable[["Runner", list[tuple[Entity, dict]]], None]) -> dict:
    if not isinstance(definitions, Definitions):
        definitions = Definitions(definitions)
    check_definitions(definitions)

    if not isinstance(protocol, Protocol):
        protocol = Protocol(protocol)
    check_protocol(protocol, definitions)

    runner = Runner(protocol, definitions)
    runner.set_callback(callback)
    outputs = runner.run(inputs=inputs)
    return outputs