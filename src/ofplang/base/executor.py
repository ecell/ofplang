#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from .protocol import EntityDescription
from .store import Store

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import Model

logger = getLogger(__name__)


class Executor:

    def __init__(self) -> None:
        self.__store: Store | None = None
    
    def set_store(self, store: Store) -> None:
        self.__store = store
    
    @property
    def store(self) -> Store:
        assert self.__store is not None
        return self.__store
    
    def initialize(self) -> None:
        pass

    async def __call__(self, model: 'Model', process: EntityDescription, inputs: dict, job_id: str, outputs_training: dict | None = None) -> tuple[str, EntityDescription, dict]:
        raise NotImplementedError()

class OperationNotSupportedError(RuntimeError):
    pass

class ExecutorBase(Executor):

    async def __call__(self, model: 'Model', process: EntityDescription, inputs: dict, job_id: str, outputs_training: dict | None = None) -> tuple[str, EntityDescription, dict]:
        outputs = await self.execute(model, process, inputs, outputs_training)
        result = (job_id, process, outputs)
        return result
    
    async def execute(self, model: 'Model', process: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        # logger.info(f"execute: {(process, inputs)}")

        outputs = {}
        if model.issubclass(model.get_by_id(process.id).type, "BuiltinProcess"):
            if model.issubclass(model.get_by_id(process.id).type, "RunScript"):
                _process = model.get_by_id(process.id)
                script = inputs["script"]["value"]
                localdict = {key: value["value"] for key, value in inputs.items() if key != "script"}
                exec(script, {}, localdict)  #XXX: Not safe
                for _, port in _process.output():
                    assert port.id in localdict, f"No output for [{port.id}]"
                outputs = {port.id: {"value": localdict[port.id], "type": port.type} for _, port in _process.output()}
            elif model.issubclass(model.get_by_id(process.id).type, "Switch"):
                condition = inputs["condition"]["value"]
                out1, out2 = (inputs["in1"]["value"], None) if condition else (None, inputs["in1"]["value"])
                outputs = {"out1": {"value": out1, "type": f"Optional[{inputs["in1"]["type"]}]"}, "out2": {"value": out2, "type": f"Optional[{inputs["in1"]["type"]}]"}}
            elif model.issubclass(model.get_by_id(process.id).type, "Gather"):
                definition = model.get_definition_by_name(model.get_by_id(process.id).type)
                in1, in2 = inputs["in1"]["value"], inputs["in2"]["value"]
                output_type = definition["output"][0]["type"]
                if in1 is not None:
                    assert in2 is None, in2
                    outputs = {"out1": {"value": in1, "type": output_type}}
                elif in2 is not None:
                    assert in1 is None, in1
                    outputs = {"out1": {"value": in2, "type": output_type}}
                else:
                    pass
            else:
                raise OperationNotSupportedError(f"Undefined process given [{process.id}, {process.type}].")
        else:
            raise OperationNotSupportedError(f"ExecutorBase only supports BuiltinProcess [{process.id}, {process.type}].")
        return outputs