#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from ofplang.prelude import *
from ofplang.executors import Simulator
from ofplang.executors.checker import TypeChecker

inputs = {"volume1": {"value": numpy.zeros(96, dtype=float), "type": "Array[Float]"}, "volume2": {"value": numpy.zeros(96, dtype=float), "type": "Array[Float]"}}
# outputs = run(inputs, "./protocol1.yaml", "./definitions.yaml", TypeChecker())
outputs = run(inputs, "./protocol1.yaml", "./definitions.yaml", Simulator())

# inputs = {"volume1": {"value": numpy.linspace(0, 100, 96, dtype=float), "type": "Array[Float]"}, "condition": {"value": True, "type": "Boolean"}}
# outputs = run(inputs, "./protocol2.yaml", "./definitions.yaml", Simulator())

print(outputs)