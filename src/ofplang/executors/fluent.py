#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from typing import Any
import numpy

from ..base.executor import OperationNotSupportedError
from ..base.model import Model
from ..base.protocol import EntityDescription

from .simulator import SimulatorBase

logger = getLogger(__name__)


class TecanFluentController(SimulatorBase):

    async def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."
        from . import tecan

        try:
            outputs = await super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                params: dict[str, Any] = {}
                (data, ), _ = tecan.read_absorbance_3colors(**params)
                if inputs["in1"]["type"] == "Plate96":
                    pass
                else:
                    assert inputs["in1"]["type"] == "SpotArray"
                    indices = inputs["in1"]["value"]["indices"]
                    data = [data[0][indices], data[1][indices], data[2][indices]]
                outputs["value"] = {"value": data, "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]
            else:
                raise err

        if operation.type == "DispenseLiquid96Wells":
            channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
            if inputs["in1"]["type"] == "Plate96":
                assert len(volume) == 96
            else:
                indices = inputs["in1"]["value"]["indices"]
                assert len(volume) == len(indices)
                volume, tmp = numpy.zeros(96), volume
                volume[indices] = tmp

            volume = volume.astype(int)
            params = {'data': volume, 'channel': channel}
            _ = tecan.dispense_liquid_96wells(**params)

        return outputs
