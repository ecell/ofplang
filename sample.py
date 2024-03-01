#!/usr/bin/python
# -*- coding: utf-8 -*-

from protocol import Protocol
from validate import check_protocol

import sys


protocol = Protocol("./sample.yaml")
print(list(protocol.connections()))
protocol.save(sys.stdout)

check_protocol(protocol)