#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from definitions import Definitions
from protocol import Protocol
from runner import Runner
from simulator import Simulator

definitions = Definitions('./manipulate.yaml')
protocol = Protocol("./sample.yaml")
runner = Runner(protocol, definitions)
runner.set_callback(Simulator())

n_training = 10

volume = numpy.zeros(96, dtype=float)
volume[: n_training] = numpy.random.uniform(0, 200, n_training)
inputs = {"volume": volume, "type": "Array[Float]"}
outputs = runner.run(inputs=inputs)

x_training = volume[: n_training]
y_training = outputs["data"]["value"][0][: n_training]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from modAL.models import ActiveLearner

def GP_regression_std(regressor, X):
    _, _pred_sigma = regressor.predict(X, return_std=True)
    query_idx = numpy.argmax(_pred_sigma)
    return query_idx, X[query_idx]

kernel = ConstantKernel() * RBF() + WhiteKernel()
regressor = ActiveLearner(
    estimator=GaussianProcessRegressor(kernel=kernel, alpha=0, random_state=0),
    query_strategy=GP_regression_std,
    X_training=x_training.reshape(-1, 1),
    y_training=y_training.reshape(-1, 1)
    )

n_queries = 40
n_queries = min(n_queries, 95 - n_training)
x_taught = []
y_taught = []
x_grid = numpy.linspace(0, 200, 101)
for idx in range(n_queries):
    query_idx, query_instance = regressor.query(x_grid.reshape(-1, 1))
    print(query_idx, query_instance)

    x = x_grid[query_idx]
    volume = numpy.zeros(96, dtype=float)
    volume[n_training + idx] = x
    inputs = {"volume": volume, "type": "Array[Float]"}
    outputs = runner.run(inputs)
    y = outputs["data"]["value"][0][n_training + idx]

    regressor.teach(numpy.array([[x]]), numpy.array([[y]]))
    x_taught.append(x)
    y_taught.append(y)

print(x_taught, y_taught)

x_grid = numpy.linspace(0, 200, 101)
pred_mu, pred_sigma = regressor.predict(x_grid.reshape(-1, 1), return_std=True)
pred_mu, pred_sigma = pred_mu.ravel(), pred_sigma.ravel()

import matplotlib.pyplot as plt
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.plot(x_grid, x_grid ** 3 / (x_grid ** 3 + 100.0 ** 3), '--', color='k')
ax.plot(x_grid, pred_mu, color='k')
ax.fill_between(x_grid, pred_mu - pred_sigma, pred_mu + pred_sigma, alpha=0.2, color='gray')
# ax.scatter(x[nsamples: ], y[nsamples: ], color='skyblue', s=10)
ax.scatter(x_training, y_training, color='red')
ax.scatter(x_taught, y_taught, color='k')
for i in range(n_queries):
    ax.annotate(f"{i+1}", (x_taught[i], y_taught[i] + 0.03))
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig('GPR.png')

import optuna

def objective(trial):
    x = trial.suggest_float('x', 0, 200)
    pred_mu, _ = regressor.predict(numpy.array([[x]]), return_std=True)
    y = pred_mu[0]
    return (y - 0.5) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=96)

import matplotlib.pyplot as plt
ax = optuna.visualization.matplotlib.plot_optimization_history(study)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('optimization_history.png', dpi=300)

print(study.best_params)  # E.g. {'x': 99.94241349236856}

volume = numpy.zeros(96, dtype=float)
volume[n_training + n_queries] = study.best_params['x']
inputs = {"volume": volume, "type": "Array[Float]"}
outputs = runner.run(inputs=inputs)
y = outputs["data"]["value"][0][n_training + n_queries]
print(y)