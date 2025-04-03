#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import io
from typing import IO, Any
import itertools
from collections import defaultdict
import keyword

from .utils import join_and
from .entity_type import TypeManager, Process, EntityTypeLoader, Entity
from .definitions import Definitions
from .protocol import Protocol

logger = getLogger(__name__)


import importlib.resources
import os.path
import pathlib

import yamale
from yamale.validators import DefaultValidators, Validator

class Identifier(Validator):
    """Custom validator for identifier"""
    tag = 'identifier'

    def _is_valid(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        value_ = value.strip()
        return value_.isidentifier() and not keyword.iskeyword(value_)

def load_yaml(file: str | pathlib.Path | IO) -> tuple[list[tuple], list[str] | None]:
    if isinstance(file, str):
        with pathlib.Path(file).open() as f:
            content = f.read()
    elif isinstance(file, pathlib.Path):
        with file.open() as f:
            content = f.read()
    elif isinstance(file, io.IOBase):
        content = f.read()
    else:
        raise TypeError(f"Invalid type [{type(file)}]")

    try:
        data = yamale.make_data(content=content)
    except Exception as e:
        error_messages = [f"Failed to load the given YAML document. {str(e)}"]
        logger.error(str(e))
        return ([], error_messages)
    return (data, None)

def check_yaml_schema(filename: str, data: list[tuple]) -> list[str] | None:
    validators = DefaultValidators.copy()  # This is a dictionary
    validators[Identifier.tag] = Identifier
    schema = yamale.make_schema(os.path.join(importlib.resources.files("ofplang.base"), filename), validators=validators)

    try:
        yamale.validate(schema, data)
        # logger.info('Validation success!')
    except yamale.YamaleError as e:
        # logger.error('Validation failed!')
        error_messages = []
        for result in e.results:
            logger.error(f"Error validating data '{result.data}' with '{result.schema}'\n\t")
            for error in result.errors:
                error_messages.append(str(error))
                logger.error(f"\t{str(error)}")
        return error_messages

    return None

def check_definitions_schema(definitions: list[tuple]) -> list[str] | None:
    return check_yaml_schema("definitions_schema.yml", definitions)

def check_protocol_schema(protocol: list[tuple]) -> list[str] | None:
    return check_yaml_schema("protocol_schema.yml", protocol)

def check_unique_ids(protocol: list[tuple]) -> list[str] | None:
    data = protocol[0][0]['contents']
    error_messages = []

    for key in ("input", "output", "processes"):
        ids = [x["id"] for x in data[key]]
        for i, x in enumerate(data[key]):
            if ids.count(x["id"]) != 1:
                err = f"{key}.{i}.id: The given id '{x["id"]}' is not unique."
                logger.error(err)
                error_messages.append(err)
    
    for i, x in enumerate(data["processes"]):
        for key in ("input", "output"):
            if x["id"] == key:
                err = f"processes.{i}.id: The process id can not be '{key}'."
                logger.error(err)
                error_messages.append(err)
    
    return (None if len(error_messages) == 0 else error_messages)

def check_definitions_type(definitions: Definitions, loader: EntityTypeLoader) -> list[str] | None:
    error_messages = []

    for i, xdef in enumerate(definitions):
        # xdef["base"] is already checked by the schema, see 'identifier'.

        for key in ("input", "output"):
            if key not in xdef:
                continue
            for j, port in enumerate(xdef[key]):
                if not loader.is_valid(port["type"]):
                    err = f"{i}.{key}.{j}.type: The type of '{xdef["name"]}.{port["id"]}', '{port["type"]}', is invalid."
                    logger.error(err)
                    error_messages.append(err)
                if 'default' in port:
                    if not loader.is_valid(port["default"]["type"]):
                        err = f"{i}.{key}.{j}.default.type: The type '{port["default"]["type"]}' is invalid [{xdef["name"]}.{port["id"]}]."
                        logger.error(err)
                        error_messages.append(err)

    return (None if len(error_messages) == 0 else error_messages)

def check_each_type(protocol: list[tuple], loader: EntityTypeLoader) -> list[str] | None:
    data = protocol[0][0]['contents']
    error_messages = []

    for i, x in enumerate(data["processes"]):
        if not loader.is_valid(x["type"], primitive=True):
            err = f"processes.{i}.type: The process type '{x["type"]}' is invalid."
            logger.error(err)
            error_messages.append(err)
        elif not issubclass(loader.evaluate(x["type"]), Process):
            err = f"processes.{i}.type: The process type '{x["type"]}' is not 'Process'."
            logger.error(err)
            error_messages.append(err)

    for key in ("input", "output"):
        for i, port in enumerate(data[key]):
            if not loader.is_valid(port["type"]):
                err = f"{key}.{i}.type: The {key} port type '{port["type"]}' is invalid."
                logger.error(err)
                error_messages.append(err)
            if 'default' in port and not loader.is_valid(port['default']['type']):
                err = f"{key}.{i}.default.type: The default type '{port["default"]["type"]}' for {key} port is invalid."
                logger.error(err)
                error_messages.append(err)

    return (None if len(error_messages) == 0 else error_messages)

def check_all_connection_ids_exist(protocol: list[tuple], definitions: Definitions) -> list[str] | None:
    data = protocol[0][0]['contents']
    error_messages = []

    one_another = lambda x: "input" if x == "output" else "output"

    process_ids = {}
    for key in ("input", "output"):
        process_ids[key] = {one_another(key): [x["id"] for x in data[key]], key: []}
    for x in data["processes"]:
        xdef = definitions.get_by_name(x["type"])
        port_ids = {"input": [port["id"] for port in xdef.get("input", ())], "output": [port["id"] for port in xdef.get("output", ())]}
        process_ids[x["id"]] = port_ids

    for i, connection in enumerate(data["connections"]):
        for key in ("input", "output"):
            process_id, port_id = connection[key]
            if process_id not in process_ids:
                err = f"connections.{i}.{key}.0: The process id '{process_id}' does not exist."
                logger.error(err)
                error_messages.append(err)
            elif port_id not in process_ids[process_id][one_another(key)]:
                err = f"connections.{i}.{key}.1: The process '{process_id}' does not have the {one_another(key)} port named '{port_id}'."
                logger.error(err)
                error_messages.append(err)

    return (None if len(error_messages) == 0 else error_messages)

def check_connection_type_matched(protocol: list[tuple], definitions: Definitions, loader: EntityTypeLoader) -> list[str] | None:
    data = protocol[0][0]['contents']
    error_messages = []

    process_ids = {}
    for key in ("input", "output"):
        process_ids[key] = {x["id"]: x["type"] for x in data[key]}    
    for x in data["processes"]:
        xdef = definitions.get_by_name(x["type"])
        port_types = {x["id"]: x["type"] for x in itertools.chain(xdef.get("input", ()), xdef.get("output", ()))}
        process_ids[x["id"]] = port_types

    for i, connection in enumerate(data["connections"]):
        input_type = process_ids[connection["input"][0]][connection["input"][1]]
        output_type = process_ids[connection["output"][0]][connection["output"][1]]
        if not loader.is_acceptable(input_type,  output_type):
            err = f"connections.{i}: The type of the connected ports is mismatched. '{input_type}' != '{output_type}'."
            logger.error(err)
            error_messages.append(err)

    return (None if len(error_messages) == 0 else error_messages)

def check_connection_integrity(protocol: list[tuple], definitions: Definitions, loader: EntityTypeLoader) -> list[str] | None:
    data = protocol[0][0]['contents']
    error_messages = []

    addresses = defaultdict(list)
    for i, connection in enumerate(data["connections"]):
        for key in ("input", "output"):
            addresses[tuple(connection[key])].append(f"connections.{i}.{key}")

    for key in ("input", "output"):
        for i, port in enumerate(data[key]):
            if len(addresses[(key, port["id"])]) > 1 and not (key == "input" and loader.is_data(port["type"])):
                err = f"{key}.{i}: The port '{port["id"]}' of user '{key}' has multiple connections, {join_and(addresses[((key, port["id"]))])}."
                logger.error(err)
                error_messages.append(err)
            elif len(addresses[(key, port["id"])]) == 0:
                err = f"{key}.{i}: The port '{port["id"]}' of user '{key}' has no connection."
                logger.error(err)
                error_messages.append(err)

    for i, x in enumerate(data["processes"]):
        xdef = definitions.get_by_name(x["type"])
        for key in ("input", "output"):
            for port in xdef.get(key, ()):
                if len(addresses[(x["id"], port["id"])]) > 1 and not (key == "output" and loader.is_data(port["type"])):
                    err = f"processes.{i}: The port '{port["id"]}' of process '{x["id"]}' has multiple connections, {join_and(addresses[((x["id"], port["id"]))])}."
                    logger.error(err)
                    error_messages.append(err)
                elif len(addresses[(x["id"], port["id"])]) == 0 and "default" not in port:
                    err = f"processes.{i}: The port '{port["id"]}' of process '{x["id"]}' has no connection."
                    logger.error(err)
                    error_messages.append(err)

    return (None if len(error_messages) == 0 else error_messages)

def validate_protocol(protocol_file: str | pathlib.Path | IO, definitions_file: str | pathlib.Path | IO) -> list[str] | None:
    # validate definitions
    definitions_, error_messages = load_yaml(definitions_file)
    if error_messages is not None:
        return error_messages
    
    error_messages = check_definitions_schema(definitions_)
    if error_messages is not None:
        return error_messages

    definitions = Definitions()
    definitions.load(os.path.join(importlib.resources.files("ofplang.base"), "builtin_definitions.yaml"))
    definitions.load(definitions_file)

    # Required for EntityTypeLoader
    for i, xdef in enumerate(definitions):
        if xdef["base"] not in EntityTypeLoader.BUILTIN_TYPES and not definitions.has(xdef["base"]):
            error_messages = [f"{i}.base: '{xdef["base"]}' is undefined."]
            logger.error(error_messages[0])
            return error_messages

    loader = EntityTypeLoader(definitions)

    error_messages = check_definitions_type(definitions, loader)
    if error_messages is not None:
        return error_messages

    # validate protocol
    protocol, error_messages = load_yaml(protocol_file)
    if error_messages is not None:
        return error_messages
    
    for check_function in (check_protocol_schema, check_unique_ids):
        error_messages = check_function(protocol)
        if error_messages is not None:
            return error_messages

    error_messages = check_each_type(protocol, loader)
    if error_messages is not None:
        return error_messages

    result = check_all_connection_ids_exist(protocol, definitions)
    if result is not None:
        return result

    for check_function in (check_connection_type_matched, check_connection_integrity):
        error_messages = check_function(protocol, definitions, loader)
        if error_messages is not None:
            return error_messages

    return None

#XXX: legacy

def check_protocol(protocol: Protocol, definitions: Definitions | None = None) -> None:
    check_unique_id(protocol)
    check_default(protocol)
    check_connection_port_exists(protocol)

    if definitions is not None:
        check_process_types(protocol, definitions)
        check_port_types(protocol, definitions)

def check_default(protocol: Protocol) -> None:
    is_valid = True
    for port in protocol.input():
        if port.default is None:
            continue
        elif not isinstance(port.default, dict):
            logger.error(f"Port default must be given as a dict [{port}].")
            is_valid = False
        else:
            if "type" not in port.default:
                logger.error(f"'type' for default not defined [{port}].")
                is_valid = False
            if "value" not in port.default:
                logger.error(f"'value' for default not defined [{port}].")
                is_valid = False
    assert is_valid, "Wrong default setting."

def check_unique_id(protocol: Protocol) -> None:
    is_valid = True
    id_list = []
    for port in itertools.chain(protocol.input(), protocol.output()):
        if port.id not in id_list:
            id_list.append(port.id)
        else:
            logger.error(f"Id [{port.id}] already exists.")
            is_valid = False

    id_list = ["input", "output"]
    for entity in protocol.processes():
        if entity.id not in id_list:
            id_list.append(entity.id)
        else:
            logger.error(f"Id [{entity.id}] already exists.")
            is_valid = False

    assert is_valid, "Id must be unique."

def check_connection_port_exists(protocol: Protocol) -> None:
    id_list = ["input", "output"]
    for entity in protocol.processes():
        if entity.id not in id_list:
            id_list.append(entity.id)

    is_valid = True
    for connection in protocol.connections():
        if connection.input.process_id not in id_list:
            logger.error(f"Id [{connection.input.process_id}] does not exist.")
            is_valid = False
        if connection.output.process_id not in id_list:
            logger.error(f"Id [{connection.output.process_id}] does not exist.")
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
        if "base" not in definition:
            logger.error(f"'base' not defined [{definition}].")
            is_valid = False

    for definition in definitions:
        if 'base' not in definition:
            continue
        if definition['base'] != "Process" and definition['base'] != "IOProcess":  #XXX
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

def check_process_types(protocol: Protocol, definitions: Definitions) -> None:
    type_manager = TypeManager(definitions)

    is_valid = True
    process_types = {}
    for entity in protocol.processes():
        if not definitions.has(entity.type):
            logger.error(f"Unknown process type [{entity.type}].")
            is_valid = False
            continue

        assert type_manager.has_definition(entity.type)
        entity_type = type_manager.eval_primitive_type(entity.type)
        if not issubclass(entity_type, Process):
            logger.error(f"[{entity.type}] is not Process.")
            is_valid = False
            continue

        definition = definitions.get_by_name(entity.type)
        process_types[entity.id] = definition

    for connection in protocol.connections():
        if connection.input.process_id in process_types:
            for port in process_types[connection.input.process_id].get("output", []):
                if connection.input.port_id == port["id"]:
                    break
            else:
                logger.error(f"Unknown port [{connection.input}].")
                is_valid = False
        else:
            assert connection.input.process_id == "input"

        if connection.output.process_id in process_types:
            for port in process_types[connection.output.process_id].get("input", []):
                if connection.output.port_id == port["id"]:
                    break
            else:
                logger.error(f"Unknown port [{connection.output}].")
                is_valid = False
        else:
            assert connection.output.process_id == "output"

    assert is_valid, "Invalid process type."

def check_port_types(protocol: Protocol, definitions: Definitions) -> None:
    type_manager = TypeManager(definitions)

    is_valid = True

    process_types = {}
    for entity, process_dict in protocol.processes_with_dict():
        definition = definitions.get_by_name(entity.type)

        port_defaults = {}
        for port_default in process_dict.get("input", ()):
            assert "id" in port_default
            assert "type" in port_default
            assert "value" in port_default
            port_defaults[port_default["id"]] = {"value": port_default["value"], "type": port_default["type"]}

        #XXX: This override the existing defaults.
        #XXX: The original defaults has no chance to be tested.
        for port in definition.get("input", ()):
            if port["id"] in port_defaults:
                port["default"] = port_defaults[port["id"]]
                del port_defaults[port["id"]]

        if len(port_defaults) > 0:
            for port_id in port_defaults.keys():
                logger.error(f"No corresponding port found for the given default [{(entity.id, port_id)}]")
            is_valid = False

        process_types[entity.id] = definition

        for port in definition.get("input", ()):
            cnt = 0
            for connection in protocol.connections():
                if connection.output == (entity.id, port["id"]):
                    cnt += 1
            if cnt == 0 and "default" not in port:
                logger.error(f"Missing connection [{(entity.id, port['id'])}]")
                is_valid = False

        for port in definition.get("output", ()):
            if type_manager.is_data(port["type"]):
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
        if connection.input.process_id in process_types:
            for port in process_types[connection.input.process_id].get("output", []):
                if connection.input.port_id == port["id"]:
                    port_types[connection.input] = port["type"]
                    input_port = port["type"]
                    break
        else:
            assert connection.input.process_id == "input"
            for port in protocol.input():
                if port.id == connection.input.port_id:
                    port_types[connection.input] = port.type
                    input_port = port.type
                    break

        if connection.output.process_id in process_types:
            for port in process_types[connection.output.process_id].get("input", []):
                if connection.output.port_id == port["id"]:
                    port_types[connection.output] = port["type"]
                    output_port = port["type"]
                    break
        else:
            assert connection.output.process_id == "output"
            for port in protocol.output():
                if port.id == connection.output.port_id:
                    port_types[connection.output] = port.type
                    output_port = port.type
                    break

        # if not type_manager.is_acceptable(input_port, output_port):
        #     logger.error(f"Type mismatch [{input_port} != {output_port}] in [{connection}]")
        #     is_valid = False

        if type_manager.is_object(input_port):
            connection_counts[connection.input].append(connection)
        if type_manager.is_object(output_port):
            connection_counts[connection.output].append(connection)

    for address, connections in connection_counts.items():
        if len(connections) == 0:
            logger.error(f"Object port [{address}] has no connection.")
            is_valid = False
        elif len(connections) > 1:
            logger.error(f"Object port [{address}] has multiple connections [{connections}].")
            is_valid = False

    assert is_valid, "Invalid port type."
