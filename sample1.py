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
getLogger('runner').addHandler(handler)
getLogger('runner').setLevel(INFO)

import numpy

from definitions import Definitions
from protocol import Protocol, Entity
from validate import check_definitions, check_protocol
from runner import Runner, Token
from simulator import Simulator


definitions = Definitions('./manipulate.yaml')
check_definitions(definitions)

protocol = Protocol("./sample.yaml")
check_protocol(protocol, definitions)
protocol.dump()
protocol.graph("graph.png")

runner = Runner(protocol, definitions)

# def func(runner: Runner, tasks: list[tuple[Entity, dict]]) -> None:
#     for operation, _ in tasks:
#         # exec
#         runner.add_tokens([
#             Token(address, {"value": None})
#             for address, _ in runner.model.get_by_id(operation.id).output()])
#         if operation.type == "ServePlate96":  #XXX
#             runner.deactivate(operation.id)
# runner.add_callback(func)

callback = Simulator()
runner.add_callback(callback)

outputs = runner.run(inputs={"volume": numpy.ones(96, dtype=float), "type": "Array[Float]"})
print(outputs)
