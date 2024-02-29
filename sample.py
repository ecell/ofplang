#!/usr/bin/python
# -*- coding: utf-8 -*-

from protocol import Protocol

import sys


protocol = Protocol("./sample.yaml")
protocol.save(sys.stdout)