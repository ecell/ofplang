#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import uuid, itertools
from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Iterable
import numpy

from ..base.entity_type import RunScript
from ..base.runner import Runner, ExecutorBase, Experiment, OperationNotSupportedError, Model
from ..base.protocol import EntityDescription


class BuiltinExecutor(ExecutorBase):

    def __init__(self) -> None:
        pass

    def __call__(self, runner: Runner, jobs: Iterable[tuple[str, EntityDescription, dict]]) -> None:
        for job_id, operation, inputs in jobs:
            outputs = self.execute(runner.model, operation, inputs)
            runner.complete_job(job_id, operation, outputs)

    def execute(self, model: Model, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        logger.info(f"execute: {(operation, inputs)}")

        outputs = {}
        if operation.type == "LabwareToSpotArray":
            indices = inputs["indices"]["value"]
            assert ((0 <= indices) & (indices < 96)).all()
            outputs = {"out1": {"value": {"id": inputs["in1"]["value"]["id"], "indices": indices}, "type": "SpotArray"}}
        elif issubclass(model.get_by_id(operation.id).type, RunScript):
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
