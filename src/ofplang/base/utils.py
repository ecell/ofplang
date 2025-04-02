#!/usr/bin/python
# -*- coding: utf-8 -*-

import io
import numpy
import yaml  # type: ignore

def ndarray_representer(dumper: yaml.Dumper, array: numpy.ndarray) -> yaml.Node:
    # represent ndarray as list.
    return dumper.represent_list(array.tolist())
    # return dumper.represent_sequence("!ndarray", array.tolist())

# def ndarray_constructor(loader, node):
#   values = loader.construct_mapping(node, deep=True)
#   return numpy.ndarray(values)

yaml.add_representer(numpy.ndarray, ndarray_representer)
# yaml.add_constructor('!ndarray', ndarray_constructor)

def load_params(stream) -> dict:
    return yaml.load(stream, Loader=yaml.Loader)

def dump_params(obj: dict, stream) -> None:
    yaml.dump(obj, stream, indent=2)

def dumps_params(obj: dict) -> None:
    s = io.StringIO()
    dump_params(obj, s)
    return s.getvalue()