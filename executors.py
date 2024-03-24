#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import uuid
from dataclasses import dataclass, field
from collections import defaultdict
import numpy
import numpy.typing

from runner import Runner, ExecutorBase
from protocol import EntityDescription


@dataclass
class Plate96:
    id: str
    contents: dict[int, numpy.typing.ArrayLike] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

class Simulator(ExecutorBase):

    def __init__(self):
        pass

    def initialize(self, runner: "Runner") -> None:
        # global state
        self.__plates: dict[str, Plate96] = {}

    def __call__(self, runner: Runner, jobs: list[tuple[str, EntityDescription, dict]]) -> None:
        for job_id, operation, inputs in jobs:
            outputs = self.execute(operation, inputs)
            runner.complete_job(job_id, operation, outputs)

    def execute(self, operation: EntityDescription, inputs: dict) -> None:
        logger.info(f"execute: {(operation, inputs)}")

        outputs = {}
        if operation.type == "ServePlate96":
            plate_id = str(uuid.uuid4())
            self.__plates[plate_id] = Plate96(plate_id)
            outputs["value"] = {"value": {"id": plate_id}, "type": "Plate96"}
        elif operation.type == "StoreLabware":
            pass
        elif operation.type == "DispenseLiquid96Wells":
            outputs["out1"] = inputs["in1"]

            channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
            plate_id = inputs["in1"]["value"]["id"]
            self.__plates[plate_id].contents[channel] += volume
        elif operation.type == "ReadAbsorbance3Colors":
            outputs["out1"] = inputs["in1"]

            plate_id = inputs["in1"]["value"]["id"]
            contents = sum(self.__plates[plate_id].contents.values())
            value = contents ** 3 / (contents ** 3 + 100.0 ** 3)  # Sigmoid
            value += numpy.random.normal(scale=0.05, size=value.shape)

            outputs["value"] = {"value": [value], "type": "Spread[Array[Float]]"}
        else:
            raise RuntimeError(f"Undefined operation given [{operation.type}].")
        return outputs
