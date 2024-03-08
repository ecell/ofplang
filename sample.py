#!/usr/bin/python
# -*- coding: utf-8 -*-

from definitions import Definitions
from protocol import Protocol, PortAddress
from validate import check_protocol
from runner import Runner, Token

import sys


definitions = Definitions('./manipulate.yaml')

protocol = Protocol("./sample.yaml")
print(list(protocol.connections()))
protocol.save(sys.stdout)

check_protocol(protocol, definitions)

inputs = {"volume": None}

runner = Runner(protocol, definitions)

for key, value in inputs.items():
    runner.add_token(Token(PortAddress("input", key), {"value": value}))

runner.activate_all()

while runner.num_tokens() > 0:
    runner.transmit_token()
    tasks = runner.run()

    for operation, input_tokens in tasks:
        # exec
        print(operation)
        runner.add_tokens([
            Token(address, {"value": None})
            for address, _ in runner.model.get_by_id(operation.id).output()])

        if operation.type == "ServePlate96":  #XXX
            runner.deactivate(operation.id)

    if all(runner.has_token(address) for address, _ in runner.model.output()):
        break
    # runner._tokens()
