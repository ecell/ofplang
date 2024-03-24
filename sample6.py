#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from definitions import Definitions
from protocol import Protocol
from validate import check_definitions, check_protocol
from runner import Runner
from executors import Simulator

definitions = Definitions('./manipulate.yaml')
protocol = Protocol("./sample.yaml")
runner = Runner(protocol, definitions)

inputs = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs}")
runner.executor = Simulator()
experiment = runner.run(inputs={"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}})
print(f"outputs = {experiment.outputs}")

executor = Simulator()
trainer = Trainer(executor)
trainer.train([experiment])

inputs = {"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}}
print(f"inputs = {inputs}")
runner.executor = executor
experiment = runner.run(inputs={"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}})
print(f"outputs = {experiment.outputs}")
