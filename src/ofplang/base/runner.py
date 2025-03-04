#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import dataclasses
import pathlib
from collections import defaultdict, deque
from typing import Any, ValuesView, IO
from collections.abc import MutableMapping
from enum import IntEnum, auto
import asyncio

from .protocol import PortAddress, Protocol, EntityDescription
from .model import UntypedProcess, Model
from .definitions import Definitions
from .executor import Executor
from .store import Store, FileStore, Handler

logger = getLogger(__name__)


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

class Run:

    def __init__(self, metadata: dict | None = None) -> None:
        metadata = metadata or {}
        self.__metadata = metadata
        self.__running_jobs: dict[str, Job] = {}
        self.__complete_jobs: list[Job] = []

    def new_job(self, job_id: str, operation: EntityDescription, inputs: list[Token], metadata: dict | None = None) -> str:
        metadata = metadata or {}
        # job_id = str(uuid.uuid4())
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

class RunHandler(Handler):

    def __init__(self) -> None:
        self.__run: Run | None = None

    @property    
    def run(self) -> Run:
        assert self.__run is not None
        return self.__run

    def create_run(self, id, metadata) -> None:
        self.__run = Run(metadata={'id': id})

    def create_process(self, id, metadata) -> None:
        assert self.__run is not None
        self.__run.new_job(id, EntityDescription(metadata['id'], metadata['base']), metadata['inputs'])

    def create_operation(self, id, metadata) -> None:
        assert self.__run is not None
        pass

    def update_process(self, id, metadata) -> None:
        assert self.__run is not None
        self.__run.complete_job(id, metadata['outputs'])

class Runner:

    def __init__(self, protocol: str | Protocol, definitions: str | Definitions, executor: Executor | None = None, store: Store | None = None) -> None:
        definitions = definitions if isinstance(definitions, Definitions) else Definitions(definitions)
        protocol = protocol if isinstance(protocol, Protocol) else Protocol(protocol)

        self.__model = Model(protocol, definitions)
        self.__tokens: MutableMapping[PortAddress, deque[Token]] = defaultdict(deque)

        self.__process_status = {process.id: StatusEnum.INACTIVE for process in self.__model.processes()}
        self.__executor = executor

        self.__store = store or FileStore()
        self.__store.add_handler(RunHandler())

    @property
    def executor(self) -> Executor | None:
        return self.__executor

    @executor.setter
    def executor(self, func: Executor) -> None:
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

    def list_jobs(self, max_iterations=1) -> list[tuple[str, UntypedProcess, dict[str, Any]]]:
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
                job_id = self.__store.create_process(dict(id=process.id, base=process.type, inputs=input_tokens))
                jobs.append((job_id, process, {token.address.port_id: token.value for token in input_tokens}))

                # IOProcess fires only once.
                if self.__model.issubclass(process.type, "IOProcess"):
                    self.deactivate(process.id)
        return jobs

    def complete_job(self, job_id: str, process: EntityDescription, outputs: dict[str, Any]) -> None:
        logger.info(f"complete_job {job_id} {process.type}")
        output_tokens = [Token(PortAddress(process.id, key), value) for key, value in outputs.items()]
        self.__store.update_process(job_id, dict(outputs=output_tokens))
        for token in output_tokens:
            self.__tokens[token.address].append(token)

    async def run(self, inputs: dict, *, executor: Executor | None = None) -> Run:
        executor = executor or self.__executor
        assert executor is not None

        _ = self.__store.create_run({})

        self.clear_tokens()
        executor.initialize()
        executor.set_store(self.__store)

        input_outputs = inputs.copy()
        for address, port in self.__model.input():
            if address.port_id not in inputs:
                if port.default is None:
                    raise RuntimeError(f"Input [{address.port_id}] is missing.")
                else:
                    input_outputs[address.port_id] = port.default.copy()

        input_process = EntityDescription("input", "IOProcess")
        job_id = self.__store.create_process(dict(id="input", base="IOProcess", inputs=[]))
        self.complete_job(job_id, input_process, input_outputs)
        self.activate_all()

        tasks = []
        while self.num_tokens() > 0:
            self.transmit_token()
            jobs = self.list_jobs()

            tasks.extend([asyncio.create_task(executor(self.model, process.asentitydesc(), inputs, job_id)) for job_id, process, inputs in jobs])
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for job_done in done:
                self.complete_job(*job_done.result())
            tasks = list(pending)

            # for job in jobs: print(f"execute {job[0]} {job[1].type}")
            # executor(self, ((job[0], job[1].asentitydesc(), job[2]) for job in jobs))

            if all(self.has_token(address) for address, _ in self.model.output()) and len(tasks) == 0:
                break
        else:
            raise RuntimeError("Never get here.")

        output_tokens = [self.__tokens[address].pop() for address, _ in self.model.output()]
        output_operation = EntityDescription("output", "IOProcess")
        job_id = self.__store.create_process(dict(id="output", base="IOProcess", inputs=output_tokens))
        self.complete_job(job_id, output_operation, {})

        assert isinstance(self.__store.handlers[-1], RunHandler)
        experiment = self.__store.handlers[-1].run  #XXX:
        assert len(experiment.running()) == 0, f"Running job(s) remained [{len(experiment.running())}]."
        return experiment

    def run_sync(self, inputs: dict, *, executor: Executor | None = None) -> Run:
        return asyncio.run(self.run(inputs, executor=executor))

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
        executor: Executor) -> dict:
    if not isinstance(definitions, Definitions):
        definitions = Definitions(definitions)

    if not isinstance(protocol, Protocol):
        protocol = Protocol(protocol)

    runner = Runner(protocol, definitions, executor=executor)
    experiment = runner.run_sync(inputs=inputs)
    return experiment.output