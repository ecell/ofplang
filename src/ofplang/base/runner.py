#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import dataclasses
import inspect
import pathlib
from collections import defaultdict, deque
from typing import Any, IO
from collections.abc import MutableMapping
from enum import IntEnum, auto
from io import StringIO
import asyncio

from .protocol import PortAddress, Protocol, EntityDescription
from .model import UntypedProcess, Model
from .definitions import Definitions
from .executor import Executor
from .store import Store

logger = getLogger(__name__)

@dataclasses.dataclass
class Token:
    address: PortAddress
    value: dict[str, Any]
    current_id: str = ''
    previous_id: str = ''

class StatusEnum(IntEnum):
    ACTIVE = auto()
    INACTIVE = auto()

class Runner:

    def __init__(self, protocol: str | Protocol, definitions: str | Definitions, executor: Executor | None = None, store: Store | None = None) -> None:
        definitions = definitions if isinstance(definitions, Definitions) else Definitions(definitions)
        protocol = protocol if isinstance(protocol, Protocol) else Protocol(protocol)

        self.__model = Model(protocol, definitions)
        self.__tokens: MutableMapping[PortAddress, deque[Token]] = defaultdict(deque)

        self.__process_status = {process.id: StatusEnum.INACTIVE for process in self.__model.processes()}
        self.__executor = executor

        self.__store = store or Store()

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
                new_token = Token(connection.output, token.value, previous_id=token.current_id)
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
                job_id = self.start_job(process.asentitydesc(), input_tokens)
                jobs.append((job_id, process, {token.address.port_id: token.value for token in input_tokens}))

                # IOProcess fires only once.
                if self.__model.issubclass(process.type, "IOProcess"):
                    self.deactivate(process.id)
        return jobs

    def start_job(self, process: EntityDescription, input_tokens: list[Token]) -> str:
        dependencies = sorted(set(token.previous_id for token in input_tokens if token.previous_id != ''))
        job_id = self.__store.create_process(dict(id=process.id, base=process.type, inputs=input_tokens, dependencies=dependencies))
        return job_id

    def complete_job(self, job_id: str, process: EntityDescription, outputs: dict[str, Any]) -> None:
        logger.info(f"complete_job {job_id} {process.type}")
        output_tokens = [Token(PortAddress(process.id, key), value, current_id=job_id) for key, value in outputs.items()]
        for token in output_tokens:
            self.__tokens[token.address].append(token)
        self.__store.finish_process(job_id, metadata=dict(outputs=output_tokens))

    def execute_io(self, job_id: str, process: EntityDescription, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        #XXX: input/output are not executed by Executor
        operation_id = self.__store.create_operation(dict(process_id=job_id, name=process.type))
        func_name = inspect.currentframe().f_code.co_qualname  # type: ignore[union-attr]
        text = StringIO()  # for logging
        text.write(f"{func_name}: process={str(process)}\n")
        text.write(f"{func_name}: inputs={str(inputs)}\n")
        text.write(f"{func_name}: job_id={job_id}\n")
        text.write(f"{func_name}: outputs={str(outputs)}\n")
        self.__store.log_operation_text(operation_id, text.getvalue(), "log.txt")
        self.__store.finish_operation(operation_id)

    async def run(self, inputs: dict, *, executor: Executor | None = None) -> dict[str, Any]:
        executor = executor or self.__executor
        assert executor is not None

        run_id = self.__store.create_run({'checksum': self.__model.protocol.md5()})

        self.clear_tokens()
        executor.initialize()
        executor.set_store(self.__store)

        input_tokens: list[Token] = []
        input_inputs = {token.address.port_id: token.value for token in input_tokens}
        input_outputs = inputs.copy()
        for address, port in self.__model.input():
            if address.port_id not in inputs:
                if port.default is None:
                    raise RuntimeError(f"Input [{address.port_id}] is missing.")
                else:
                    input_outputs[address.port_id] = port.default.copy()

        input_process = EntityDescription("input", "IOProcess")
        job_id = self.start_job(input_process, input_tokens)
        self.execute_io(job_id, input_process, input_inputs, input_outputs)
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
        output_inputs = {token.address.port_id: token.value for token in output_tokens}
        output_outputs: dict[str, Any] = {}
        output_process = EntityDescription("output", "IOProcess")
        job_id = self.start_job(output_process, output_tokens)
        self.execute_io(job_id, output_process, output_inputs, output_outputs)
        self.complete_job(job_id, output_process, output_outputs)

        self.__store.finish_run(run_id)

        return {token.address.port_id: token.value for token in output_tokens}

    def run_sync(self, inputs: dict, *, executor: Executor | None = None) -> dict[str, Any]:
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
        executor: Executor) -> dict[str, Any]:
    if not isinstance(definitions, Definitions):
        definitions = Definitions(definitions)

    if not isinstance(protocol, Protocol):
        protocol = Protocol(protocol)

    runner = Runner(protocol, definitions, executor=executor)
    outputs = runner.run_sync(inputs=inputs)
    return outputs