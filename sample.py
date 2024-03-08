#!/usr/bin/python
# -*- coding: utf-8 -*-

from definitions import Definitions
from protocol import Protocol, PortAddress
from validate import check_protocol
from runner import Runner, Token

import sys

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

definitions = Definitions('./manipulate.yaml')

protocol = Protocol("./sample.yaml")
print(list(protocol.connections()))
protocol.save(sys.stdout)

check_protocol(protocol, definitions)

runner = Runner(protocol, definitions)

def func(runner: Runner, tasks: list) -> None:
    for operation, input_tokens in tasks:
        # exec
        runner.add_tokens([
            Token(address, {"value": None})
            for address, _ in runner.model.get_by_id(operation.id).output()])
        if operation.type == "ServePlate96":  #XXX
            runner.deactivate(operation.id)
runner.add_callback(func)

outputs = runner.run(inputs={"volume": None})
print(outputs)
