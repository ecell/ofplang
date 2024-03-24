#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import uuid
from dataclasses import dataclass, field
from collections import defaultdict
import numpy
from numpy.typing import ArrayLike

from runner import Runner, ExecutorBase
from protocol import EntityDescription

class OperationNotSupportedError(RuntimeError):
    pass

@dataclass
class Plate96:
    id: str
    contents: dict[int, ArrayLike] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

class SimulatorBase(ExecutorBase):

    def __init__(self) -> None:
        pass

    def initialize(self, runner: "Runner") -> None:
        # global state
        self.__plates: dict[str, Plate96] = {}

    def new_plate(self) -> str:
        plate_id = str(uuid.uuid4())
        self.__plates[plate_id] = Plate96(plate_id)
        return plate_id

    def get_plate(self, plate_id: str) -> Plate96:
        return self.__plates[plate_id]

    def __call__(self, runner: Runner, jobs: list[tuple[str, EntityDescription, dict]]) -> None:
        for job_id, operation, inputs in jobs:
            outputs = self.execute(operation, inputs)
            runner.complete_job(job_id, operation, outputs)

    def execute(self, operation: EntityDescription, inputs: dict) -> None:
        logger.info(f"execute: {(operation, inputs)}")

        outputs = {}
        if operation.type == "ServePlate96":
            plate_id = self.new_plate()
            outputs["value"] = {"value": {"id": plate_id}, "type": "Plate96"}
        elif operation.type == "StoreLabware":
            pass
        elif operation.type == "DispenseLiquid96Wells":
            channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
            plate_id = inputs["in1"]["value"]["id"]
            self.get_plate(plate_id).contents[channel] += volume
            outputs["out1"] = inputs["in1"]
        else:
            raise OperationNotSupportedError(f"Undefined operation given [{operation.type}].")
        return outputs

class Simulator(SimulatorBase):

    def execute(self, operation: EntityDescription, inputs: dict) -> None:
        try:
            outputs = super().execute(operation, inputs)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                contents = sum(self.get_plate(plate_id).contents.values())
                value = contents ** 3 / (contents ** 3 + 100.0 ** 3)  # Sigmoid
                value += numpy.random.normal(scale=0.05, size=value.shape)
                outputs["value"] = {"value": [value], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]
            else:
                raise err
        return outputs

class GaussianProcessExecutor(SimulatorBase):

    def __init__(self) -> None:
        super().__init__()

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF

        kernel = ConstantKernel() * RBF() + WhiteKernel()
        self._regressor = GaussianProcessRegressor(kernel=kernel, alpha=0, random_state=0)

    def execute(self, operation: EntityDescription, inputs: dict) -> None:
        try:
            outputs = super().execute(operation, inputs)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                contents = sum(self.get_plate(plate_id).contents.values())
                value, _ = self.predict(contents)
                outputs["value"] = {"value": [value], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]
            else:
                raise err
        return outputs

    def predict(self, contents: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        print("predict:", contents)
        pred_mu, pred_sigma = self._regressor.predict(contents.reshape(-1, 1), return_std=True)
        pred_mu, pred_sigma = pred_mu.ravel(), pred_sigma.ravel()
        return pred_mu, pred_sigma
