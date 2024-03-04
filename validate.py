#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import itertools

from definitions import Definitions
from protocol import Protocol


def check_protocol(protocol: Protocol, definitions: Definitions | None = None) -> None:
    check_unique_id(protocol)
    check_connection_port_exists(protocol)

def check_unique_id(protocol: Protocol) -> None:
    is_valid = True
    id_list = []
    for entity in itertools.chain(protocol.input(), protocol.output()):
        if entity.id not in id_list:
            id_list.append(entity.id)
        else:
            logger.error(f"Id [{entity.id}] already exists.")
            is_valid = False
    
    id_list = ["input", "output"]
    for entity in protocol.operations():
        if entity.id not in id_list:
            id_list.append(entity.id)
        else:
            logger.error(f"Id [{entity.id}] already exists.")
            is_valid = False

    assert is_valid, "Id must be unique."

def check_connection_port_exists(protocol: Protocol) -> None:
    id_list = ["input", "output"]
    for entity in protocol.operations():
        if entity.id not in id_list:
            id_list.append(entity.id)

    is_valid = True
    for connection in protocol.connections():
        if connection.input.operation_id not in id_list:
            logger.error(f"Id [{connection.input.operation_id}] does not exist.")
            is_valid = False
        if connection.output.operation_id not in id_list:
            logger.error(f"Id [{connection.output.operation_id}] does not exist.")
            is_valid = False

    assert is_valid, "Invalid connection."