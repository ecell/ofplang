#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from .protocol import EntityDescription
from .entity_type import RunScript

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .runner import Model

logger = getLogger(__name__)


class Executor:

    def initialize(self) -> None:
        pass

    async def __call__(self, model: 'Model', operation: EntityDescription, inputs: dict, job_id: str, outputs_training: dict | None = None) -> tuple[str, EntityDescription, dict]:
        raise NotImplementedError()

class OperationNotSupportedError(RuntimeError):
    pass

class ExecutorBase(Executor):

    async def __call__(self, model: 'Model', operation: EntityDescription, inputs: dict, job_id: str, outputs_training: dict | None = None) -> tuple[str, EntityDescription, dict]:
        outputs = await self.execute(model, operation, inputs, outputs_training)
        result = (job_id, operation, outputs)
        return result
    
    async def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
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