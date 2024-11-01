#!/usr/bin/python
# -*- coding: utf-8 -*-

from logging import getLogger, StreamHandler, Formatter, INFO

handler = StreamHandler()
handler.setLevel(INFO)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# logger = getLogger(__name__)
# logger.addHandler(handler)
# logger.setLevel(INFO)
getLogger('executors').addHandler(handler)
getLogger('executors').setLevel(INFO)

import numpy

from ofplang.definitions import Definitions
from ofplang.protocol import Protocol
from ofplang.validate import check_definitions, check_protocol
from ofplang.runner import Runner
from ofplang.executors import Simulator, TecanFluentController, GaussianProcessExecutor


definitions = Definitions('./manipulate.yaml')
definitions.load("./sample3.yaml", "definitions")
check_definitions(definitions)

protocol = Protocol("./sample3.yaml")
check_protocol(protocol, definitions)
# protocol.dump()
protocol.graph("graph.png")

# runner = Runner(protocol, definitions, executor=TecanFluentController())
runner = Runner(protocol, definitions, executor=Simulator())

N, M = 48, 16
inputs = {"volume1": {"value": numpy.random.uniform(0, 75, N), "type": "Array[Float]"}, "volume2": {"value": numpy.random.uniform(0, 75, N), "type": "Array[Float]"}, "indices": {"value": numpy.arange(N), "type": "Array[Integer]"}}
experiment0 = runner.run(inputs=inputs)
for job in experiment0.jobs():
    print(job)
print(experiment0.output)

executor = GaussianProcessExecutor()
executor.teach(runner.model, experiment0)

inputs = {"volume1": {"value": numpy.random.uniform(0, 75, M), "type": "Array[Float]"}, "volume2": {"value": numpy.random.uniform(0, 75, M), "type": "Array[Float]"}, "indices": {"value": numpy.arange(N, N + M), "type": "Array[Integer]"}}
experiment1 = runner.run(inputs=inputs)
print(f"outputs = {experiment1.output}")
experiment2 = runner.run(inputs=inputs, executor=executor)
print(f"outputs = {experiment2.output}")

# inputs = {"volume": {"value": numpy.linspace(0, 150, 96), "type": "Array[Float]"}, "indices": {"value": numpy.arange(96), "type": "Array[Integer]"}}
# experiment3 = runner.run(inputs=inputs, executor=executor)
# print(f"outputs = {experiment2.output}")

# plotting
import matplotlib.pyplot as plt

fig = plt.figure(dpi=300)
for idx in range(3):
    ax = fig.add_subplot(2, 2, idx + 1)
    x = experiment1.output["data"]["value"][idx]
    y = experiment2.output["data"]["value"][idx]
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    vmin, vmax = vmin - (vmin + vmax) * 0.05, vmax + (vmin + vmax) * 0.05
    ax.plot((vmin, vmax), (vmin, vmax), "k--")
    ax.plot(x, y, 'k.')
    #ax.plot(inputs["volume"]["value"], x, 'k.')
    #ax.plot(inputs["volume"]["value"], y, 'r.')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Prediction')
    ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig(f'comparison.png')

_, ax = plt.subplots(dpi=300)
x = experiment0.input["volume1"]["value"]
y = experiment0.input["volume2"]["value"]
ax.plot(x, y, 'k.')
x = experiment1.input["volume1"]["value"]
y = experiment1.input["volume2"]["value"]
ax.plot(x, y, 'rx')
ax.set_xlim(0, 75)
ax.set_ylim(0, 75)
ax.set_xlabel('volume1')
ax.set_ylabel('volume1')
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig('result.png')