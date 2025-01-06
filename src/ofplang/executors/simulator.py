#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import uuid
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import numpy

from ..base.executor import OperationNotSupportedError
from ..base.model import Model
from ..base.protocol import EntityDescription

from .builtin import BuiltinExecutor

logger = getLogger(__name__)

@dataclass
class Plate96:
    id: str
    contents: defaultdict[int, numpy.ndarray] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

class SimulatorBase(BuiltinExecutor):

    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        super().initialize()

        # global state
        self.__plates: dict[str, Plate96] = {}
        self.__liquids: defaultdict[int, float] = defaultdict(float)  # stock

    def new_plate(self, plate_id: str | None = None) -> str:
        plate_id = plate_id or str(uuid.uuid4())
        assert plate_id not in self.__plates
        self.__plates[plate_id] = Plate96(plate_id)
        return plate_id

    def get_plate(self, plate_id: str) -> Plate96:
        return self.__plates[plate_id]

    async def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        logger.info(f"execute: {(operation, inputs)}")

        try:
            outputs = await super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ServePlate96":
                plate_id = self.new_plate(None if outputs_training is None else outputs_training["value"]["value"]["id"])
                outputs["value"] = {"value": {"id": plate_id}, "type": "Plate96"}
            elif operation.type == "StoreLabware":
                pass
            elif operation.type == "DispenseLiquid96Wells":
                channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
                plate_id = inputs["in1"]["value"]["id"]
                if inputs["in1"]["type"] == "Plate96":
                    assert len(volume) == 96, f"The length of volume must be 96. [{len(volume)}] was given."
                else:
                    indices = inputs["in1"]["value"]["indices"]
                    assert len(volume) == len(indices), f"The length of volume have to be the same with indices [{len(volume)} != {len(indices)}]."
                    volume, tmp = numpy.zeros(96), volume
                    volume[indices] = tmp
                self.get_plate(plate_id).contents[channel] += volume
                self.__liquids[channel] += sum(volume)
                outputs["out1"] = inputs["in1"]
            elif operation.type == "Sleep":
                duration = inputs["duration"]["value"]
                await asyncio.sleep(duration)
                outputs["out1"] = inputs["in1"]
            elif operation.type == "Gather":
                outputs["out1"] = inputs["in1"]
                outputs["out2"] = inputs["in2"]
            else:
                raise err
        return outputs
class Simulator(SimulatorBase):

    async def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."

        try:
            outputs = await super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                # start = numpy.zeros(96, dtype=float)  # self.get_plate(plate_id).contents.default_factory()
                # contents = sum(self.get_plate(plate_id).contents.values(), start)
                # contents = self.get_plate(plate_id).contents[2]
                contents = self.get_plate(plate_id).contents

                x = numpy.zeros(96, dtype=float)
                if 1 in contents:
                    x += contents[1] * 1.0
                if 2 in contents:
                    x += contents[2] * 1.0
                value1 = 30 * numpy.cos(x / 10.0 * numpy.pi) + 50  # Cosine
                value1 += numpy.random.normal(scale=0.1, size=value1.shape)

                x = numpy.zeros(96, dtype=float)
                if 1 in contents:
                    x += contents[1] * 0.2
                if 2 in contents:
                    x += contents[2] * 1.8
                value2: numpy.ndarray = 100 * x / (x + 180.0) + 50  # Sigmoid
                value2 += numpy.random.normal(scale=0, size=value2.shape)

                x = numpy.zeros(96, dtype=float)
                if 1 in contents:
                    x += contents[1] * 1.8
                if 2 in contents:
                    x += contents[2] * 0.2
                value3: numpy.ndarray = 30 * (numpy.sin(x / 50.0 * numpy.pi) + 1.0) + 15  # Sin
                value3 += numpy.random.normal(scale=0, size=value3.shape)

                if inputs["in1"]["type"] == "Plate96":
                    pass
                else:
                    assert inputs["in1"]["type"] == "SpotArray"
                    indices = inputs["in1"]["value"]["indices"]
                    value1, value2, value3 = value1[indices], value2[indices], value3[indices]

                outputs["value"] = {"value": [value1, value2, value3], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]
            else:
                raise err
        return outputs