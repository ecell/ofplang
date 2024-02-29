#!/usr/bin/python
# -*- coding: utf-8 -*-

from protocol import Protocol

import sys


protocol = Protocol("./sample.yaml")
print(list(protocol.connections()))
# protocol.save(sys.stdout)