#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from runner import run
from simulator import Simulator

inputs = {"volume": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}
print(f"inputs = {inputs}")
outputs = run(inputs, protocol="./sample.yaml", definitions='./manipulate.yaml', callback=Simulator())
print(f"outputs = {outputs}")
