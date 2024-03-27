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

# initial training set
value = numpy.pad(numpy.random.uniform(0, 200, 8), (0, 96 - 8), 'constant')
inputs_training = {"volume": {"value": value, "type": "Array[Float]"}}
print(f"inputs = {inputs_training}")
experiment = runner.run(inputs=inputs_training)
print(f"outputs = {experiment.output}")

executor = GaussianProcessExecutor()
executor.teach(experiment)

# first test
inputs = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs}")
experiment1 = runner.run(inputs=inputs)
print(f"outputs = {experiment1.output}")

prediction1 = runner.run(inputs=inputs, executor=executor)
print(f"outputs = {prediction1.output}")

# generating the data
input_samples = [
    {"volume": {"value": numpy.pad(numpy.random.uniform(0, 200, 8), (0, 96 - 8), 'constant'), "type": "Array[Float]"}}
    for _ in range(10)]

# active learning
for _ in range(12):
    idx_query, uncertainty_query = executor.query(runner, input_samples)
    print(f"query, uncertainty = {idx_query}, {uncertainty_query}")
    inputs_query = input_samples[idx_query]

    # print(f"inputs = {inputs_query}")
    experiment = runner.run(inputs=inputs_query)
    # print(f"outputs = {experiment.output}")

    executor.teach(experiment)

# second test
inputs = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs}")
experiment2 = runner.run(inputs=inputs)
print(f"outputs = {experiment2.output}")

prediction2 = runner.run(inputs=inputs, executor=executor)
print(f"outputs = {prediction2.output}")

# plotting
import matplotlib.pyplot as plt
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.plot((-0.1, +1.1), (-0.1, +1.1), "k--")
ax.plot(experiment1.output["data"]["value"][0], prediction1.output["data"]["value"][0], 'k.')
ax.plot(experiment2.output["data"]["value"][0], prediction2.output["data"]["value"][0], 'r.')
ax.set_xlabel(r'$Experiment$')
ax.set_ylabel(r'$Prediction$')
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig('comparison.png')
