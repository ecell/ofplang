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

    definitions = Definitions(definitions_file)

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