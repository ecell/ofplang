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

from definitions import Definitions
from protocol import Protocol
from validate import check_definitions, check_protocol
from runner import Runner
from executors import Simulator


definitions = Definitions('./manipulate.yaml')
check_definitions(definitions)

protocol = Protocol("./sample.yaml")
check_protocol(protocol, definitions)
protocol.dump()
protocol.graph("graph.png")

runner = Runner(protocol, definitions, executor=Simulator())
experiment = runner.run(inputs={"volume": {"value": numpy.random.uniform(0, 200, 96), "type": "Array[Float]"}})
print(experiment.output)
