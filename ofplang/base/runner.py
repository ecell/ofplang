#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import dataclasses
import pathlib
import uuid
from collections import defaultdict, deque, OrderedDict
from typing import Any, ValuesView, IO
from collections.abc import Iterable, Iterator, MutableMapping
from enum import IntEnum, auto

from .protocol import PortAddress, Port, Protocol, PortConnection, EntityDescription
from .definitions import Definitions
from .entity_type import TypeManager, IOProcess
from . import entity_type
from .validate import check_definitions, check_protocol

logger = getLogger(__name__)


@dataclasses.dataclass
class Entity:
    id: str
    type: type

class Process:

    def __init__(self, entity: Entity, definition: dict) -> None:
        self.__entity = entity
        self.__definition = definition

    def input(self) -> Iterable[tuple[PortAddress, Port]]:
        input = {
            PortAddress(self.__entity.id, port["id"]): Port(**port)
            for port in self.__definition.get("input", [])}
        return input.items()

    def output(self) -> Iterable[tuple[PortAddress, Port]]:
        output = {
            PortAddress(self.__entity.id, port["id"]): Port(**port)
            for port in self.__definition.get("output", [])}
        return output.items()

    def asentity(self) -> Entity:
        return self.__entity

    def asentitydesc(self) -> EntityDescription:
        return EntityDescription(self.__entity.id, self.__definition["name"])

    @property
    def id(self) -> str:
        return self.__entity.id

    @property
    def type(self) -> type:
        return self.__entity.type

    @property
    def definition(self) -> dict:
        return self.__definition.copy()  # deepcopy

class Model:

    def __init__(self, protocol: Protocol, definitions: Definitions) -> None:
        self.__protocol = protocol
        self.__definitions = definitions

        # check inputs
        check_definitions(definitions)
        check_protocol(protocol, definitions)

        self.__type_manager = TypeManager(self.__definitions)
        self.__load()

    def __load(self) -> None:
        self.__processes = OrderedDict()
        for process, process_dict in self.__protocol.processes_with_dict():
            process_type = self.__type_manager.eval_primitive_type(process.type)
            assert issubclass(process_type, entity_type.Process), f"[{process.type}] is not Process."
            definition = self.__definitions.get_by_name(process.type)
            if "input" in process_dict:
                input_defaults = {port["id"]: {"value": port["value"], "type": port["type"]} for port in process_dict["input"]}
                for port in definition["input"]:
                    if port["id"] in input_defaults:
                        port["default"] = input_defaults[port["id"]]
            self.__processes[process.id] = Process(Entity(process.id, process_type), definition)

    def get_by_id(self, id: str) -> Process:
        return self.__processes[id]

    def connections(self) -> Iterator[PortConnection]:
        return self.__protocol.connections()

    def processes(self) -> Iterable[Process]:
        return self.__processes.values()

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
class Job:
    operation: EntityDescription
    inputs: list[Token]
    outputs: list[Token] | None
    metadata: dict

class Experiment:

    def __init__(self, metadata: dict | None = None) -> None:
        metadata = metadata or {}
        self.__metadata = metadata
        self.__running_jobs: dict[str, Job] = {}
        self.__complete_jobs: list[Job] = []

    def new_job(self, operation: EntityDescription, inputs: list[Token], metadata: dict | None = None) -> str:
        metadata = metadata or {}
        job_id = str(uuid.uuid4())
        self.__running_jobs[job_id] = Job(operation, inputs, None, metadata)
        return job_id

    def complete_job(self, job_id: str, outputs: list[Token], metadata: dict | None = None) -> None:
        metadata = metadata or {}
        assert job_id in self.__running_jobs, job_id
        job = self.__running_jobs.pop(job_id)
        self.__complete_jobs.append(Job(job.operation, job.inputs, outputs, dict(job.metadata, **metadata)))

    @property
    def input(self) -> dict[str, Any]:
        assert len(self.__complete_jobs) > 0
        job = self.__complete_jobs[0]
        assert job.outputs is not None
        assert job.operation.id == "input"
        return {token.address.port_id: token.value for token in job.outputs}

    @property
    def output(self) -> dict[str, Any]:
        assert len(self.__complete_jobs) > 1
        job = self.__complete_jobs[-1]
        assert job.operation.id == "output"
        return {token.address.port_id: token.value for token in job.inputs}

    def jobs(self) -> list[Job]:
        return self.__complete_jobs  #XXX: copy?

    def running(self) -> ValuesView[Job]:
        return self.__running_jobs.values()

class OperationNotSupportedError(RuntimeError):
    pass

class ExecutorBase:

    def initialize(self) -> None:
        pass

    def __call__(self, runner: "Runner", jobs: Iterable[tuple[str, EntityDescription, dict]]) -> None:
        raise NotImplementedError()

# class _Preprocessor:

#     def __init__(self, runner: "Runner", jobs: list[tuple[str, Process, dict]]) -> None:
#         self.__runner = runner
#         self.__jobs = jobs

#     def __iter__(self) -> Iterator[tuple[str, EntityDescription, dict]]:
#         for job_id, process, inputs in self.__jobs:
#             if issubclass(process.type, BuiltinProcess):
#                 outputs = self.execute(job_id, process, inputs)
#                 self.__runner.complete_job(job_id, process.asentitydesc(), outputs)
#             else:
#                 yield (job_id, process.asentitydesc(), inputs)

#     def execute(self, job_id: str, process: Process, inputs: dict) -> dict:
#         outputs: dict = {}
#         if issubclass(process.type, RunScript):
#             script = inputs["script"]["value"]
#             localdict = {key: value["value"] for key, value in inputs.items() if key != "script"}
#             exec(script, {}, localdict)  #XXX: Not safe
#             for _, port in process.output():
#                 assert port.id in localdict, f"No output for [{port.id}]"
#             outputs = {port.id: {"value": localdict[port.id], "type": port.type} for _, port in process.output()}
#         elif process.asentitydesc().type == "SpotArrayFromLabware":
#             indices = inputs["indices"]["value"]
#             assert ((0 <= indices) & (indices < 96)).all()
#             outputs = {"out1": {"value": {"id": inputs["in1"]["value"]["id"], "indices": indices}, "type": "SpotArray"}}
#         else:
#             raise ProcessNotSupportedError(f"Undefined process given [{process.asentitydesc().id}, {process.asentitydesc().type}].")
#         return outputs

class Runner:

    def __init__(self, protocol: str | Protocol, definitions: str | Definitions, executor: ExecutorBase | None = None) -> None:
        definitions = definitions if isinstance(definitions, Definitions) else Definitions(definitions)
        protocol = protocol if isinstance(protocol, Protocol) else Protocol(protocol)

        self.__model = Model(protocol, definitions)
        self.__tokens: MutableMapping[PortAddress, deque[Token]] = defaultdict(deque)

        self.__process_status = {process.id: StatusEnum.INACTIVE for process in self.__model.processes()}
        self.__executor = executor
        self.__experiment: Experiment = Experiment()

    @property
    def executor(self) -> ExecutorBase | None:
        return self.__executor

    @executor.setter
    def executor(self, func: ExecutorBase) -> None:
        self.__executor = func

    def activate_all(self) -> None:
        for process_id in self.__process_status.keys():
            self.__process_status[process_id] = StatusEnum.ACTIVE

    def deactivate(self, id: str) -> None:
        assert id in self.__process_status
        self.__process_status[id] = StatusEnum.INACTIVE

    def transmit_token(self) -> None:
        new_tokens: MutableMapping[PortAddress, deque[Token]] = defaultdict(deque)
        for connection in self.__model.connections():
            for token in self.__tokens[connection.input]:
                new_token = Token(connection.output, token.value)
                new_tokens[new_token.address].append(new_token)
        for connection in self.__model.connections():
            if connection.input in self.__tokens:
                del self.__tokens[connection.input]
        for address in new_tokens:
            self.__tokens[address].extend(new_tokens[address])

    def list_jobs(self, max_iterations=1) -> list[tuple[str, Process, dict[str, Any]]]:
        jobs = []
        for process in self.__model.processes():
            for _ in range(max_iterations):
                if self.__process_status[process.id] is not StatusEnum.ACTIVE:
                    continue
                elif any(port.default is None and not self.has_token(address) for address, port in process.input()):
                    continue
                # pop tokens here
                input_tokens = [
                    (self.__tokens[address].pop() if self.has_token(address) else Token(address, port.default or {}))  #XXX: port.default can be None
                    for address, port in process.input()]
                job_id = self.__experiment.new_job(process.asentitydesc(), input_tokens)
                jobs.append((job_id, process, {token.address.port_id: token.value for token in input_tokens}))

                # IOProcess fires only once.
                if issubclass(process.type, IOProcess):
                    self.deactivate(process.id)
        return jobs

    def complete_job(self, job_id: str, process: EntityDescription, outputs: dict[str, Any]) -> None:
        output_tokens = [Token(PortAddress(process.id, key), value) for key, value in outputs.items()]
        self.__experiment.complete_job(job_id, output_tokens)
        for token in output_tokens:
            self.__tokens[token.address].append(token)

    def run(self, inputs: dict, *, executor: ExecutorBase | None = None) -> Experiment:
        executor = executor or self.__executor
        assert executor is not None

        self.__experiment = Experiment()
        self.clear_tokens()
        executor.initialize()

        input_outputs = inputs.copy()
        for address, port in self.__model.input():
            if address.port_id not in inputs:
                if port.default is None:
                    raise RuntimeError(f"Input [{address.port_id}] is missing.")
                else:
                    input_outputs[address.port_id] = port.default.copy()

        input_process = EntityDescription("input", "IOProcess")
        self.complete_job(self.__experiment.new_job(input_process, []), input_process, input_outputs)
        self.activate_all()

        while self.num_tokens() > 0:
            self.transmit_token()
            jobs = self.list_jobs()
            # executor(self, _Preprocessor(self, jobs))
            executor(self, ((job[0], job[1].asentitydesc(), job[2]) for job in jobs))
            if all(self.has_token(address) for address, _ in self.model.output()):
                break
        else:
            raise RuntimeError("Never get here.")

        output_tokens = [self.__tokens[address].pop() for address, _ in self.model.output()]
        output_operation = EntityDescription("output", "IOProcess")
        self.complete_job(self.__experiment.new_job(output_operation, output_tokens), output_operation, {})
        experiment, self.__experiment = self.__experiment, Experiment()
        assert len(experiment.running()) == 0, f"Running job(s) remained [{len(experiment.running())}]."
        return experiment

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
        protocol: Protocol | str | pathlib.Path | IO,
        definitions: Definitions | str | pathlib.Path | IO,
        executor: ExecutorBase) -> dict:
    if not isinstance(definitions, Definitions):
        definitions = Definitions(definitions)
    check_definitions(definitions)

    if not isinstance(protocol, Protocol):
        protocol = Protocol(protocol)
    check_protocol(protocol, definitions)

    runner = Runner(protocol, definitions)
    runner.executor = executor
    experiment = runner.run(inputs=inputs)
    return experiment.output
