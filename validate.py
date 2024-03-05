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

    definitions = definitions or Definitions()
    check_operation_types(protocol, definitions)
    check_port_types(protocol, definitions)

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

def check_operation_types(protocol: Protocol, definitions: Definitions) -> None:
    is_valid = True
    operation_types = {}
    for entity in protocol.operations():
        if not definitions.has(entity.type):
            logger.error(f"Unknown operation type [{entity.type}].")
            is_valid = False
            continue
        definition = definitions.get_by_id(entity.type)
        if definition.get("ref") != "Operation":
            logger.error(f"Wrong operation type [{entity.type}].")
            is_valid = False
            continue
        operation_types[entity.id] = definition
    
    for connection in protocol.connections():
        if connection.input.operation_id in operation_types:
            for port in operation_types[connection.input.operation_id].get("output", []):
                if connection.input.port_id == port["id"]:
                    break
            else:
                logger.error(f"Unknown port [{connection.input}].")
                is_valid = False
        else:
            assert connection.input.operation_id == "input"
        if connection.output.operation_id in operation_types:
            for port in operation_types[connection.output.operation_id].get("input", []):
                if connection.output.port_id == port["id"]:
                    break
            else:
                logger.error(f"Unknown port [{connection.output}].")
                is_valid = False
        else:
            assert connection.output.operation_id == "output"

    assert is_valid, "Invalid operation type."

def check_port_types(protocol: Protocol, definitions: Definitions) -> None:
    operation_types = {}
    for entity in protocol.operations():
        operation_types[entity.id] = definitions.get_by_id(entity.type)

    is_valid = True
    port_types = {}
    for connection in protocol.connections():
        if connection.input.operation_id in operation_types:
            for port in operation_types[connection.input.operation_id].get("output", []):
                if connection.input.port_id == port["id"]:
                    port_types[connection.input] = port["type"]
                    input_port = port["type"]
                    break
        else:
            assert connection.input.operation_id == "input"
            for port in protocol.input():
                if port.id == connection.input.port_id:
                    port_types[connection.input] = port.type
                    input_port = port.type
                    break

        if connection.output.operation_id in operation_types:
            for port in operation_types[connection.output.operation_id].get("input", []):
                if connection.output.port_id == port["id"]:
                    port_types[connection.output] = port["type"]
                    output_port = port["type"]
                    break
        else:
            assert connection.output.operation_id == "output"
            for port in protocol.output():
                if port.id == connection.output.port_id:
                    port_types[connection.output] = port.type
                    output_port = port.type
                    break
        
        #XXX:
        if input_port != output_port:
            logger.error(f"Type mismatch [{input_port} != {output_port}] in [{connection}]")
            is_valid = False

    assert is_valid, "Invalid port type."
