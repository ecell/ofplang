#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from definitions import Definitions
from protocol import Protocol
from runner import Runner
from executors import Simulator, GaussianProcessExecutor


definitions = Definitions('./manipulate.yaml')
protocol = Protocol("./sample.yaml")
runner = Runner(protocol, definitions, executor=Simulator())

inputs_training = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs_training}")
experiment = runner.run(inputs=inputs_training)
print(f"outputs = {experiment.output}")

executor = GaussianProcessExecutor()
executor.teach(experiment)

inputs = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs}")
new_experiment = runner.run(inputs=inputs)
print(f"outputs = {new_experiment.output}")

prediction = runner.run(inputs=inputs, executor=executor)
print(f"outputs = {prediction.output}")


import matplotlib.pyplot as plt
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.plot((-0.1, +1.1), (-0.1, +1.1), "k--")
ax.plot(new_experiment.output["data"]["value"][0], prediction.output["data"]["value"][0], 'k.')
ax.set_xlabel(r'$Experiment$')
ax.set_ylabel(r'$Prediction$')
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig('comparison.png')
