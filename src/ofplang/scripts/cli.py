#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger
import pathlib
from typing import IO
import yaml
import click

from ofplang.prelude import *
from ofplang.executors import Simulator

logger = getLogger(__name__)


@click.group()
def cli() -> None:
    pass

@cli.command(help="Run a protocol")
@click.argument('protocol')
@click.option("--definitions", "-d", help="The definition YAML file", default=None)
@click.option("--cli-input-yaml", default=None)
def run(protocol: str, definitions: str | None, cli_input_yaml: str | None) -> None:
    click.echo(f"Run protocol [{protocol}] with definitions [{definitions}]")

    if definitions is None:
        definitions = "./definitions.yaml"
    runner = Runner(protocol, definitions, executor=Simulator())
    
    click.echo(f"Parse inputs [{cli_input_yaml}]")
    if cli_input_yaml is not None:
        with pathlib.Path(cli_input_yaml).open() as f:
            inputs = yaml.load(f, Loader=yaml.Loader)
    else:
        inputs = {}

    click.echo(f"Input value: {inputs}")    
    outputs = runner.run(inputs).output
    click.echo(f"Output value: {outputs}")

def main() -> None:
    cli()


if __name__ == "__main__":
    main()