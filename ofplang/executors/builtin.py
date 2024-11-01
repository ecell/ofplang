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


class BultinExecutor(ExecutorBase):

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

@dataclass
class Plate96:
    id: str
    contents: defaultdict[int, numpy.ndarray] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

class SimulatorBase(BultinExecutor):

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

    def execute(self, model: Model, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        logger.info(f"execute: {(operation, inputs)}")

        try:
            outputs = super().execute(model, operation, inputs, outputs_training)
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
            else:
                raise err
        return outputs
