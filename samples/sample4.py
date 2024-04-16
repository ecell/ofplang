#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from ofplang.definitions import Definitions
from ofplang.protocol import Protocol
from ofplang.runner import Runner
from ofplang.executors import Simulator, GaussianProcessExecutor


definitions = Definitions('./manipulate.yaml')
protocol = Protocol("./sample1.yaml")
runner = Runner(protocol, definitions, executor=Simulator())

inputs_training = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs_training}")
experiment = runner.run(inputs=inputs_training)
print(f"outputs = {experiment.output}")

executor = GaussianProcessExecutor()
executor.teach(runner.model, experiment)

inputs = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs}")
new_experiment = runner.run(inputs=inputs)
print(f"outputs = {new_experiment.output}")

prediction = runner.run(inputs=inputs, executor=executor)
print(f"outputs = {prediction.output}")

# plotting
import matplotlib.pyplot as plt
for idx in range(3):
    x = new_experiment.output["data"]["value"][idx]
    y = prediction.output["data"]["value"][idx]
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    vmin, vmax = vmin - (vmin + vmax) * 0.05, vmax + (vmin + vmax) * 0.05
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot((vmin, vmax), (vmin, vmax), "k--")
    ax.plot(x, y, 'k.')
    #ax.plot(inputs["volume"]["value"], x, 'k.')
    #ax.plot(inputs["volume"]["value"], y, 'r.')
    ax.set_xlabel(r'$Experiment$')
    ax.set_ylabel(r'$Prediction$')
    # ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(f'comparison{idx}.png')
