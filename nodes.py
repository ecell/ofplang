#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import yaml


with open("./manipulate.yaml") as f:
    conf = yaml.load(f, Loader=yaml.Loader)

for node in conf.get("nodes", []):
    name = node['id']

    cls = type(name, (), {})
    print(cls)
    break