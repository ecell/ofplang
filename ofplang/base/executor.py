#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger
from collections.abc import Iterable

from .protocol import EntityDescription
from .entity_type import RunScript

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .runner import Runner, Model

logger = getLogger(__name__)


class Executor:

    def initialize(self) -> None:
        pass

    def __call__(self, runner: 'Runner', jobs: Iterable[tuple[str, EntityDescription, dict]]) -> None:
        raise NotImplementedError()

class OperationNotSupportedError(RuntimeError):
    pass

class ExecutorBase(Executor):

    def __init__(self) -> None:
        pass

    def __call__(self, runner: 'Runner', jobs: Iterable[tuple[str, EntityDescription, dict]]) -> None:
        for job_id, operation, inputs in jobs:
            outputs = self.execute(runner.model, operation, inputs)
            runner.complete_job(job_id, operation, outputs)

    def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        # logger.info(f"execute: {(operation, inputs)}")

        outputs = {}
        if issubclass(model.get_by_id(operation.id).type, RunScript):
            _operation = model.get_by_id(operation.id)
            script = inputs["script"]["value"]
            localdict = {key: value["value"] for key, value in inputs.items() if key != "script"}
            exec(script, {}, localdict)  #XXX: Not safe
            for _, port in _operation.output():
                assert port.id in localdict, f"No output for [{port.id}]"
            outputs = {port.id: {"value": localdict[port.id], "type": port.type} for _, port in _operation.output()}
        else:
            raise OperationNotSupportedError(f"Undefined operation given [{operation.id}, {operation.type}].")
        return outputs

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