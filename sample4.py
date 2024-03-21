#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from runner import run
from simulator import Simulator

inputs = {"volume": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}
print(f"inputs = {inputs}")
outputs = run(inputs, protocol="./sample.yaml", definitions='./manipulate.yaml', executor=Simulator())
print(f"outputs = {outputs}")

nsamples = 20
x = inputs["volume"]
y = outputs["data"]["value"][0]
x_training = x[: nsamples]
y_training = y[: nsamples]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF

kernel = ConstantKernel() * RBF() + WhiteKernel()
regressor = GaussianProcessRegressor(kernel=kernel, alpha=0, random_state=0)
regressor.fit(x_training.reshape(-1, 1), y_training) 

x_grid = numpy.linspace(0, 200, 100)
pred_mu, pred_sigma = regressor.predict(x_grid.reshape(-1, 1), return_std=True)
pred_mu, pred_sigma = pred_mu.ravel(), pred_sigma.ravel()

import matplotlib.pyplot as plt
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.plot(x_grid, x_grid ** 3 / (x_grid ** 3 + 100.0 ** 3), '--', color='k')
ax.plot(x_grid, pred_mu, color='k')
ax.fill_between(x_grid, pred_mu - pred_sigma, pred_mu + pred_sigma, alpha=0.2, color='gray')
ax.scatter(x[nsamples: ], y[nsamples: ], color='skyblue', s=10)
ax.scatter(x_training, y_training, color='red')
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$y$')
plt.tight_layout()
plt.savefig('GPR.png')