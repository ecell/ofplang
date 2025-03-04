#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from ofplang.prelude import *
from ofplang.executors import TecanFluentSimulator
# from ofplang.executors.checker import TypeChecker

import logging
logging.basicConfig()
logging.getLogger('ofplang').setLevel(level=logging.INFO)

import asyncio
from ofplang.executors.fluent import tecan_fluent_operator

async def main():
    asyncio.create_task(tecan_fluent_operator())

    inputs = {"volume1": {"value": numpy.zeros(96, dtype=float), "type": "Array[Float]"}, "volume2": {"value": numpy.zeros(96, dtype=float), "type": "Array[Float]"}}
    # outputs = run(inputs, "./protocol1.yaml", "./definitions.yaml", TypeChecker())
    # outputs = run(inputs, "./protocol1.yaml", "./definitions.yaml", TecanFluentSimulator())

    runner = Runner("./protocol1.yaml", "./definitions.yaml")
    outputs = (await runner.run(inputs, executor=TecanFluentSimulator())).output

    # inputs = {"volume1": {"value": numpy.linspace(0, 100, 96, dtype=float), "type": "Array[Float]"}, "condition": {"value": True, "type": "Boolean"}}
    # outputs = run(inputs, "./protocol2.yaml", "./definitions.yaml", Simulator())

    print(outputs)

asyncio.run(main())
