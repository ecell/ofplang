#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import uuid, itertools
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterable
import numpy
from numpy.typing import ArrayLike

from .runner import Runner, ExecutorBase, Experiment
from .protocol import EntityDescription

class OperationNotSupportedError(RuntimeError):
    pass

@dataclass
class Plate96:
    id: str
    contents: dict[int, ArrayLike] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

class SimulatorBase(ExecutorBase):

    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        # global state
        self.__plates: dict[str, Plate96] = {}

    def new_plate(self, plate_id: str | None = None) -> str:
        plate_id = plate_id or str(uuid.uuid4())
        assert plate_id not in self.__plates
        self.__plates[plate_id] = Plate96(plate_id)
        return plate_id

    def get_plate(self, plate_id: str) -> Plate96:
        return self.__plates[plate_id]

    def __call__(self, runner: Runner, jobs: list[tuple[str, EntityDescription, dict]]) -> None:
        for job_id, operation, inputs in jobs:
            outputs = self.execute(operation, inputs)
            runner.complete_job(job_id, operation, outputs)

    def execute(self, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> None:
        logger.info(f"execute: {(operation, inputs)}")

        outputs = {}
        if operation.type == "ServePlate96":
            plate_id = self.new_plate(None if outputs_training is None else outputs_training["value"]["value"]["id"])
            outputs["value"] = {"value": {"id": plate_id}, "type": "Plate96"}
        elif operation.type == "StoreLabware":
            pass
        elif operation.type == "DispenseLiquid96Wells":
            channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
            plate_id = inputs["in1"]["value"]["id"]
            self.get_plate(plate_id).contents[channel] += volume
            outputs["out1"] = inputs["in1"]
        else:
            raise OperationNotSupportedError(f"Undefined operation given [{operation.id}, {operation.type}].")
        return outputs

class Simulator(SimulatorBase):

    def execute(self, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> None:
        assert outputs_training is None, "'teach' is not supported."

        try:
            outputs = super().execute(operation, inputs, outputs_training)
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
        from modAL.models import ActiveLearner

        kernel = ConstantKernel() * RBF() + WhiteKernel()
        self.__learner = ActiveLearner(
            estimator=GaussianProcessRegressor(kernel=kernel, alpha=0, random_state=0))

    def initialize(self) -> None:
        super().initialize()
        self.__uncertainty = 0.0

    @property
    def uncertainty(self):
        return self.__uncertainty

    def execute(self, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> None:
        try:
            outputs = super().execute(operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                contents = sum(self.get_plate(plate_id).contents.values())
                if outputs_training is not None:
                    # train here
                    self.__teach(contents, outputs_training["value"]["value"][0])
                value, std = self.__predict(contents)
                outputs["value"] = {"value": [value], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]

                self.__uncertainty = max(self.__uncertainty, std.max())
            else:
                raise err
        return outputs

    def __teach(self, x_training: ArrayLike, y_training: ArrayLike) -> None:
        self.__learner.teach(x_training.reshape(-1, 1), y_training)

    def __predict(self, contents: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        pred_mu, pred_sigma = self.__learner.predict(contents.reshape(-1, 1), return_std=True)
        pred_mu, pred_sigma = pred_mu.ravel(), pred_sigma.ravel()
        return pred_mu, pred_sigma

    def teach(self, experiment: Experiment) -> None:
        self.initialize()
        for job in experiment.jobs():
            if job.operation.id == "input" or job.operation.id == "output":
                continue

            inputs = {token.address.port_id: token.value for token in job.inputs}
            outputs = {token.address.port_id: token.value for token in job.outputs}
            self.execute(job.operation, inputs, outputs)

    def query(self, runner: Runner | Iterable[Runner], inputs: Iterable[dict]) -> tuple[int, float]:
        if isinstance(runner, Runner):
            runner = itertools.repeat(runner)
        idx_query, uncertainty_query = None, 0.0
        for idx, (runner_, inputs_) in enumerate(zip(runner, inputs)):
            _ = runner_.run(inputs=inputs_, executor=self)
            if idx_query is None or self.uncertainty > uncertainty_query:
                idx_query, uncertainty_query = idx, self.uncertainty
        if idx_query is None:
            raise RuntimeError(f"No sample.")
        return idx_query, uncertainty_query