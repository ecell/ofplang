#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import click

logger = getLogger(__name__)


@click.group()
def cli():
    pass

@cli.command()
def run():
    pass

def main():
    cli()


if __name__ == "__main__":
    main()