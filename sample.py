#!/usr/bin/python
# -*- coding: utf-8 -*-

from definitions import Definitions
from protocol import Protocol
from validate import check_protocol

import sys


definitions = Definitions('./manipulate.yaml')
print(definitions.get_by_id("Plate96"))

protocol = Protocol("./sample.yaml")
print(list(protocol.connections()))
protocol.save(sys.stdout)

check_protocol(protocol, definitions)