#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import dataclasses, pathlib, io, uuid
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
    value: dict[str, Any]

class StatusEnum(IntEnum):
    ACTIVE = auto()
    INACTIVE = auto()

@dataclasses.dataclass
class Run:
    operation: Entity
    inputs: list[Token]
    outputs: list[Token] | None
    metadata: dict

class Experiment:

    def __init__(self) -> None:
        self.__runs: dict[str, Run] = {}

    def new_token(self, address: PortAddress, value: dict[str, Any]) -> Token:
        return Token(address, value)

    def new_run(self, operation: Entity, inputs: list[Token], metadata: dict | None = None) -> str:
        metadata = metadata or {}
        run_id = str(uuid.uuid4())
        self.__runs[run_id] = Run(operation, inputs, None, metadata)
        return run_id

    def complete_run(self, run_id: str, outputs: list[Token], metadata: dict | None = None) -> str:
        metadata = metadata or {}
        assert run_id in self.__runs, run_id
        run = self.__runs[run_id]
        self.__runs[run_id] = Run(run.operation, run.inputs, outputs, dict(run.metadata, **metadata))

def default_executor(runner: 'Runner', tasks: list[tuple[str, Entity, dict]]) -> None:
    for run_id, operation, input_tokens in tasks:
        logger.info(f"default_executor: {(run_id, operation, input_tokens)}")

class Runner:

    def __init__(self, protocol: Protocol, definitions: Definitions) -> None:
        self.__model = Model(protocol, definitions)
        self.__tokens = defaultdict(deque)

        self.__operation_status = {operation.id: StatusEnum.INACTIVE for operation in self.__model.operations()}
        self.__executor = default_executor
        self.__experiment = Experiment()

    @property
    def executor(self) -> Callable[["Runner", list[tuple[str, Entity, dict]]], None]:
        return self.__executor

    @executor.setter
    def executor(self, func: Callable[["Runner", list[tuple[str, Entity, dict]]], None]) -> None:
        self.__executor = func

    def activate_all(self) -> None:
        for operation_id in self.__operation_status.keys():
            self.__operation_status[operation_id] = StatusEnum.ACTIVE

    def deactivate(self, id: str) -> None:
        assert id in self.__operation_status
        self.__operation_status[id] = StatusEnum.INACTIVE

    def transmit_token(self) -> None:
        new_tokens = defaultdict(deque)
        for connection in self.__model.connections():
            for token in self.__tokens[connection.input]:
                new_token = self.__experiment.new_token(connection.output, token.value)
                new_tokens[new_token.address].append(new_token)
        for connection in self.__model.connections():
            del self.__tokens[connection.input]
        for address in new_tokens:
            self.__tokens[address].extend(new_tokens[address])

    def list_tasks(self, max_iterations=1) -> list[tuple[str, Entity, dict[str, Any]]]:
        tasks = []
        for operation in self.__model.operations():
            if self.__operation_status[operation.id] is not StatusEnum.ACTIVE:
                continue
            for _ in range(max_iterations):
                if any(port.default is None and not self.has_token(address) for address, port in operation.input()):
                    continue
                # pop tokens here
                input_tokens = [
                    (self.__tokens[address].pop() if self.has_token(address) else self.__experiment.new_token(address, port.default))
                    for address, port in operation.input()]
                run_id = self.__experiment.new_run(operation.asentity(), input_tokens)
                tasks.append((run_id, operation.asentity(), {token.address.port_id: token.value for token in input_tokens}))
        return tasks

    def complete_task(self, run_id: str, operation: Entity, outputs: dict[str, Any]) -> None:
        output_tokens = [self.__experiment.new_token(PortAddress(operation.id, key), value) for key, value in outputs.items()]
        self.__experiment.complete_run(run_id, outputs)
        for token in output_tokens:
            self.__tokens[token.address].append(token)

    def run(self, inputs: dict) -> dict:
        self.clear_tokens()
        input_operation = Entity("input", "Operation")
        self.complete_task(self.__experiment.new_run(input_operation, {}), input_operation, inputs)
        self.activate_all()

        while self.num_tokens() > 0:
            self.transmit_token()
            tasks = self.list_tasks()
            self.__executor(self, tasks)
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
        executor: Callable[["Runner", list[tuple[Entity, dict]]], None]) -> dict:
    if not isinstance(definitions, Definitions):
        definitions = Definitions(definitions)
    check_definitions(definitions)

    if not isinstance(protocol, Protocol):
        protocol = Protocol(protocol)
    check_protocol(protocol, definitions)

    runner = Runner(protocol, definitions)
    runner.executor = executor
    outputs = runner.run(inputs=inputs)
    return outputs