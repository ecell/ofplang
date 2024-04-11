#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import uuid, itertools
from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Iterable
import numpy

from .runner import Runner, ExecutorBase, Experiment, OperationNotSupportedError
from .protocol import EntityDescription


@dataclass
class Plate96:
    id: str
    contents: defaultdict[int, numpy.ndarray] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

class SimulatorBase(ExecutorBase):

    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
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

    def __call__(self, runner: Runner, jobs: Iterable[tuple[str, EntityDescription, dict]]) -> None:
        for job_id, operation, inputs in jobs:
            outputs = self.execute(operation, inputs)
            runner.complete_job(job_id, operation, outputs)

    def execute(self, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
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
            self.__liquids[channel] += sum(volume)
            outputs["out1"] = inputs["in1"]
        else:
            raise OperationNotSupportedError(f"Undefined operation given [{operation.id}, {operation.type}].")
        return outputs

class TecanFluentController(SimulatorBase):

    def execute(self, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."
        from . import tecan

        outputs = {}
        if operation.type == "ServePlate96":
            outputs = super().execute(operation, inputs, outputs_training)
        elif operation.type == "StoreLabware":
            outputs = super().execute(operation, inputs, outputs_training)
        elif operation.type == "DispenseLiquid96Wells":
            outputs = super().execute(operation, inputs, outputs_training)

            channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
            volume = volume.astype(int)
            params = {'data': volume, 'channel': channel}
            _ = tecan.dispense_liquid_96wells(**params)
        elif operation.type == "ReadAbsorbance3Colors":
            params = {}
            (data, ), _ = tecan.read_absorbance_3colors(**params)
            outputs["value"] = {"value": data, "type": "Spread[Array[Float]]"}
            outputs["out1"] = inputs["in1"]
        else:
            raise OperationNotSupportedError(f"Undefined operation given [{operation.id}, {operation.type}].")
        return outputs

class Simulator(SimulatorBase):

    def execute(self, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."

        try:
            outputs = super().execute(operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                start = numpy.zeros(96, dtype=float)  # self.get_plate(plate_id).contents.default_factory()
                contents = sum(self.get_plate(plate_id).contents.values(), start)

                value1: numpy.ndarray = contents ** 3 / (contents ** 3 + 100.0 ** 3)  # Sigmoid
                value1 += numpy.random.normal(scale=0.05, size=value1.shape)
                value2: numpy.ndarray = 100 * contents / (contents + 180.0) + 50  # Sigmoid
                value2 += numpy.random.normal(scale=5, size=value2.shape)
                value3: numpy.ndarray = 30 * (numpy.sin(contents / 50.0 * numpy.pi) + 1.0) + 15  # Sin
                value3 += numpy.random.normal(scale=3, size=value3.shape)

                outputs["value"] = {"value": [value1, value2, value3], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]
            else:
                raise err
        return outputs

class GaussianProcessExecutor(SimulatorBase):

    def __init__(self) -> None:
        super().__init__()

        from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
        from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF  # type: ignore
        # from modAL.models import ActiveLearner  # type: ignore

        kernel = ConstantKernel() * RBF() + ConstantKernel() + WhiteKernel()
        # self.__learner = ActiveLearner(
        #     estimator=GaussianProcessRegressor(kernel=kernel, alpha=0, random_state=0))
        self.__estimator = GaussianProcessRegressor(kernel=kernel, alpha=0, random_state=0)
        self.__X_training = None
        self.__y_training = None

    def initialize(self) -> None:
        super().initialize()
        self.__uncertainty = 0.0

    @property
    def uncertainty(self):
        return self.__uncertainty

    def execute(self, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        try:
            outputs = super().execute(operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                start = numpy.zeros(96, dtype=float)  # self.get_plate(plate_id).contents.default_factory()
                contents = sum(self.get_plate(plate_id).contents.values(), start)
                if outputs_training is not None:
                    # train here
                    y_training = numpy.array(outputs_training["value"]["value"]).T
                    self.__teach(contents, y_training)
                (value, std) = self.__predict(contents)
                print(f"value.shape = {value.shape}")
                outputs["value"] = {"value": [value.T[0], value.T[1], value.T[2]], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]

                self.__uncertainty = max(self.__uncertainty, std.ravel().max())
            else:
                raise err
        return outputs

    def __teach(self, x_training: numpy.ndarray, y_training: numpy.ndarray) -> None:
        # self.__learner.teach(x_training.reshape(-1, 1), y_training)

        # _add_training_data
        if self.__X_training is None:
            self.__X_training = x_training.reshape(-1, 1)
            self.__y_training = y_training
        else:
            self.__X_training = numpy.concatenate((self.__X_training, x_training.reshape(-1, 1)))
            self.__y_training = numpy.concatenate((self.__y_training, y_training))
        # _fit_to_known
        self.__estimator.fit(self.__X_training, self.__y_training)

    def __predict(self, contents: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        # pred_mu, pred_sigma = self.__learner.predict(contents.reshape(-1, 1), return_std=True)
        pred_mu, pred_sigma = self.__estimator.predict(contents.reshape(-1, 1), return_std=True)
        # pred_mu, pred_sigma = pred_mu.ravel(), pred_sigma.ravel()
        return pred_mu, pred_sigma

    def teach(self, experiment: Experiment) -> None:
        self.initialize()
        for job in experiment.jobs():
            if job.operation.id == "input" or job.operation.id == "output":
                continue

            inputs = {token.address.port_id: token.value for token in job.inputs}
            assert job.outputs is not None
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