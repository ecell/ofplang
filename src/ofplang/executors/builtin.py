#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from ..base.model import Model
from ..base.executor import ExecutorBase, OperationNotSupportedError
from ..base.protocol import EntityDescription

logger = getLogger(__name__)


class BuiltinExecutor(ExecutorBase):

    async def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        try:
            outputs = await super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "LabwareToSpotArray":
                indices = inputs["indices"]["value"]
                assert ((0 <= indices) & (indices < 96)).all()
                outputs = {"out1": {"value": {"id": inputs["in1"]["value"]["id"], "indices": indices}, "type": "SpotArray"}}
            else:
                raise err
        return outputs