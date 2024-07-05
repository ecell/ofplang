#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import uuid, itertools
from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Iterable
import numpy

from .entity_type import RunScript
from .runner import Runner, ExecutorBase, Experiment, OperationNotSupportedError, Model
from .protocol import EntityDescription


@dataclass
class Plate96:
    id: str
    contents: defaultdict[int, numpy.ndarray] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

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

class TecanFluentController(SimulatorBase):

    def execute(self, model: Model, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."
        from . import tecan

        try:
            outputs = super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                params = {}
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

class Simulator(SimulatorBase):

    def execute(self, model: Model, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."

        try:
            outputs = super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                # start = numpy.zeros(96, dtype=float)  # self.get_plate(plate_id).contents.default_factory()
                # contents = sum(self.get_plate(plate_id).contents.values(), start)
                # contents = self.get_plate(plate_id).contents[2]
                contents = self.get_plate(plate_id).contents

                x = np.zeros(96, dtype=float)
                if 1 in contents:
                    x += contents[1] * 1.0
                if 2 in contents:
                    x += contents[2] * 1.0
                value1 = 30 * np.cos(x / 10.0 * np.pi) + 50  # Cosine
                value1 += np.random.normal(scale=0.1, size=value1.shape)

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

import numpy as np
import torch
import gpytorch

class GaussianProcessExecutor(SimulatorBase):

    def __init__(self) -> None:
        super().__init__()
        self.clear()
        self.mean_x = None  # Add member variable for mean
        self.covar_x = None  # Add member variable for covariance
        self.__uncertainty = 0.0  # Initialize uncertainty

    def clear(self) -> None:
        self.__X_training = None
        self.__y_training = None
        self.__feature_indices = {}
        self.__models = []
        self.__likelihoods = []
        self.mean_x = None  # Clear mean
        self.covar_x = None  # Clear covariance
        self.__uncertainty = 0.0  # Clear uncertainty

    @property
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.__X_training.copy(), self.__y_training.copy())

    @property
    def mean(self) -> np.ndarray:
        return self.mean_x

    @property
    def covariance(self) -> np.ndarray:
        return self.covar_x

    @property
    def uncertainty(self) -> float:
        return self.__uncertainty

    def __add_feature(self, channel: int) -> int:
        assert channel not in self.__feature_indices
        value = len(self.__feature_indices)
        self.__feature_indices[channel] = value
        if self.__X_training is not None:
            self.__X_training = np.hstack((self.__X_training, np.zeros(self.__X_training.shape[0], dtype=float).reshape(-1, 1)))
        return value

    def execute(self, model: Model, operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        try:
            outputs = super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                plate = self.get_plate(plate_id)

                if inputs["in1"]["type"] == "Plate96":
                    indices = np.arange(96)
                else:
                    assert inputs["in1"]["type"] == "SpotArray"
                    indices = inputs["in1"]["value"]["indices"]

                for channel in plate.contents.keys():
                    if channel not in self.__feature_indices:
                        self.__add_feature(channel)

                contents = np.zeros((len(indices), len(self.__feature_indices)), dtype=float)
                for channel, value in plate.contents.items():
                    contents.T[self.__feature_indices[channel]] = np.array(value)[indices]

                if outputs_training is not None:
                    y_training = np.array(outputs_training["value"]["value"]).T
                    self.__teach(contents, y_training)
                (value, std) = self.__predict(contents)

                outputs["value"] = {"value": [value.T[0], value.T[1], value.T[2]], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]

                self.__uncertainty = max(self.__uncertainty, std.ravel().max())
            else:
                raise err
        return outputs

    def __teach(self, X_training: np.ndarray, y_training: np.ndarray) -> None:
        if self.__X_training is None:
            self.__X_training = X_training
            self.__y_training = y_training
        else:
            assert self.__y_training is not None
            self.__X_training = np.concatenate((self.__X_training, X_training))
            self.__y_training = np.concatenate((self.__y_training, y_training))

        print("X_training shape:", self.__X_training.shape)
        print("y_training shape:", self.__y_training.shape)
        
        self.__train_model()

    def __train_model(self) -> None:
        train_x = torch.tensor(self.__X_training, dtype=torch.float32)
        train_y = torch.tensor(self.__y_training, dtype=torch.float32)

        print("train_x shape:", train_x.shape)
        print("train_y shape:", train_y.shape)

        for i in range(train_y.shape[1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_y[:, i], likelihood)
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            training_iter = 50
            for _ in range(training_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y[:, i])
                loss.backward()
                optimizer.step()

            self.__models.append(model)
            self.__likelihoods.append(likelihood)

    def __predict(self, contents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.__models:
            raise RuntimeError("No training yet.")

        test_x = torch.tensor(contents, dtype=torch.float32)

        pred_mu_list = []
        pred_sigma_list = []

        for model, likelihood in zip(self.__models, self.__likelihoods):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))
                pred_mu = observed_pred.mean.cpu().numpy()
                pred_sigma = observed_pred.stddev.cpu().numpy()
                pred_mu_list.append(pred_mu)
                pred_sigma_list.append(pred_sigma)

                # Compute covariance matrix
                covar = observed_pred.covariance_matrix.cpu().numpy()
                self.covar_x = covar  # Store the covariance matrix

        pred_mu = np.stack(pred_mu_list, axis=-1)
        pred_sigma = np.stack(pred_sigma_list, axis=-1)

        self.mean_x = pred_mu  # Store the mean

        return pred_mu, pred_sigma

    def teach(self, model: Model, experiment: Experiment) -> None:
        self.initialize()
        for job in experiment.jobs():
            if job.operation.id == "input" or job.operation.id == "output":
                continue

            inputs = {token.address.port_id: token.value for token in job.inputs}
            assert job.outputs is not None
            outputs = {token.address.port_id: token.value for token in job.outputs}
            self.execute(model, job.operation, inputs, outputs)

    def query(self, runner: Runner | Iterable[Runner], inputs: Iterable[dict]) -> tuple[int, float]:
        if isinstance(runner, Runner):
            runner = itertools.repeat(runner)
        idx_query, uncertainty_query = None, 0.0
        for idx, (runner_, inputs_) in enumerate(zip(runner, inputs)):
            _ = runner_.run(inputs=inputs_, executor=self)
            if idx_query is None or self.uncertainty > uncertainty_query:
                idx_query, uncertainty_query = idx, self.uncertainty
        if idx_query is None:
            raise RuntimeError("No sample.")
        return idx_query, uncertainty_query

    
class ConstantKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(self, batch_shape=None, constant_prior=None, constant_constraint=None, active_dims=None):
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)
        self.register_parameter(
            name="raw_constant",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        if constant_prior is not None:
            self.register_prior(
                "constant_prior",
                constant_prior,
                lambda m: m.constant,
                lambda m, v: m._set_constant(v),
            )

        if constant_constraint is None:
            constant_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_constant", constant_constraint)

    @property
    def constant(self):
        return self.raw_constant_constraint.transform(self.raw_constant)

    @constant.setter
    def constant(self, value):
        self._set_constant(value)

    def _set_constant(self, value):
        value = value.view(*self.batch_shape, 1)
        self.initialize(raw_constant=self.raw_constant_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        dtype = torch.promote_types(x1.dtype, x2.dtype)
        batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        shape = batch_shape + (x1.shape[-2],) + (() if diag else (x2.shape[-2],))
        constant = self.constant.to(dtype=dtype, device=x1.device)

        if not diag:
            constant = constant.unsqueeze(-1)

        if last_dim_is_batch:
            constant = constant.unsqueeze(-1)

        return constant.expand(shape)

class CustomCombinedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        from gpytorch.priors import NormalPrior
        from gpytorch.constraints import Interval
        super(CustomCombinedKernel, self).__init__(**kwargs)
        self.rbf_kernel = gpytorch.kernels.RBFKernel()
        self.matern_kernel = gpytorch.kernels.MaternKernel()
        self.constant_kernel = ConstantKernel()

    def forward(self, x1, x2, **params):
        rbf_component = self.rbf_kernel(x1, x2, **params)
        constant_component = self.constant_kernel(x1, x2, **params)
        
        return rbf_component + constant_component

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(CustomCombinedKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
