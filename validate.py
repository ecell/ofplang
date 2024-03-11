#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import itertools
from collections import defaultdict

import entity_type
from definitions import Definitions
from protocol import Protocol


def check_protocol(protocol: Protocol, definitions: Definitions | None = None) -> None:
    check_unique_id(protocol)
    check_connection_port_exists(protocol)

    definitions = definitions or Definitions()
    check_definitions(definitions)
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

def check_definitions(definitions: Definitions) -> None:
    is_valid = True
    names = []
    for definition in definitions:
        assert isinstance(definition, dict)
        if "name" not in definition:
            logger.error(f"'name' not defined [{definition}].")
            is_valid = False
        elif definition['name'] not in names:
            names.append(definition['name'])
        else:
            logger.error(f"'name' [{definition['name']}] already defined.")
            is_valid = False
        if "ref" not in definition:
            logger.error(f"'ref' not defined [{definition}].")
            is_valid = False
    
    for definition in definitions:
        if definition['ref'] != "Operation":  #XXX
            continue
        for key in ("input", "output"):
            if key not in definition:
                continue
            assert isinstance(definition[key], list)
            for port in definition[key]:
                assert isinstance(port, dict)
                if "id" not in port:
                    logger.error(f"'id' not defined [{port}].")
                    is_valid = False
                if "type" not in port:
                    logger.error(f"'id' not defined [{port}].")
                    is_valid = False
                if "default" in port:
                    assert isinstance(port["default"], dict)
                    if "type" not in port["default"]:
                        logger.error(f"'type' for default not defined [{port}].")
                        is_valid = False
                    if "value" not in port["default"]:
                        logger.error(f"'value' for default not defined [{port}].")
                        is_valid = False

    assert is_valid, "Invalid definitions."

def check_operation_types(protocol: Protocol, definitions: Definitions) -> None:
    is_valid = True
    operation_types = {}
    for entity in protocol.operations():
        if not definitions.has(entity.type):
            logger.error(f"Unknown operation type [{entity.type}].")
            is_valid = False
            continue
        definition = definitions.get_by_name(entity.type)
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
    type_checker = entity_type.TypeChecker(definitions)

    is_valid = True

    operation_types = {}
    for entity in protocol.operations():
        definition = definitions.get_by_name(entity.type)
        operation_types[entity.id] = definition

        for port in definition.get("input", []):
            cnt = 0
            for connection in protocol.connections():
                if connection.output == (entity.id, port["id"]):
                    cnt += 1
            if cnt == 0 and "default" not in port:
                logger.error(f"Missing connection [{(entity.id, port['id'])}]")
                is_valid = False

        for port in definition.get("output", []):
            if type_checker.is_data(port["type"]):
                continue
            cnt = 0
            for connection in protocol.connections():
                if connection.input == (entity.id, port["id"]):
                    cnt += 1
            if cnt == 0:
                logger.error(f"Missing connection [{(entity.id, port['id'])}]")
                is_valid = False
            elif cnt > 1:
                logger.error(f"More than 1 connections found [{(entity.id, port['id'])}]")
                is_valid = False

    port_types = {}
    connection_counts = defaultdict(list)
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
        
        if not type_checker.is_acceptable(input_port, output_port):
            logger.error(f"Type mismatch [{input_port} != {output_port}] in [{connection}]")
            is_valid = False

        if type_checker.is_object(input_port):
            connection_counts[connection.input].append(connection)
        if type_checker.is_object(output_port):
            connection_counts[connection.output].append(connection)

    for address, connections in connection_counts.items():
        if len(connections) == 0:
            logger.error(f"Object port [{address}] has no connection.")
            is_valid = False
        elif len(connections) > 1:
            logger.error(f"Object port [{address}] has multiple connections [{connections}].")
            is_valid = False

    assert is_valid, "Invalid port type."
