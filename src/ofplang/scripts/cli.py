#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger
import sys
import pathlib
import json
import contextlib
import asyncio

import numpy
import yaml
import click

from ofplang.prelude import Runner
from ofplang.executors import TecanFluentController
logger = getLogger(__name__)

def ndarray_representer(dumper: yaml.Dumper, array: numpy.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

yaml.add_representer(numpy.ndarray, ndarray_representer)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(obj)


@click.group()
def cli() -> None:
    pass

@cli.command(help="Run a protocol")
@click.argument('protocol')
@click.option("--definitions", "-d", help="The definition YAML file", default=None)
@click.option("--cli-input-yaml", default=None)
@click.option("--format", default="json")
@click.option("--output", "-o", default=None)
def run(protocol: str, definitions: str | None, cli_input_yaml: str | None, format: str, output: str | None) -> None:
    logger.debug(f"Run protocol [{protocol}] with definitions [{definitions}]")

    if definitions is None:
        definitions = "./definitions.yaml"
    runner = Runner(protocol, definitions, executor=TecanFluentController())
    
    logger.debug(f"Parse inputs [{cli_input_yaml}]")
    if cli_input_yaml is not None:
        with pathlib.Path(cli_input_yaml).open() as f:
            inputs = yaml.load(f, Loader=yaml.Loader)
    else:
        inputs = {}

    logger.debug(f"Input value: {inputs}")
    outputs = asyncio.run(runner.run(inputs)).output
    logger.debug(f"Output value: {outputs}")

    with contextlib.nullcontext(sys.stdout) if output is None else pathlib.Path(output).open('w') as f:
        if format.lower() == "yaml":
            yaml.dump(outputs, f, indent=2)
        else:
            # assert format.lower() == "json", format
            json.dump(outputs, f, cls=NumpyEncoder, indent=2)

def main() -> None:
    cli()


if __name__ == "__main__":
    main()