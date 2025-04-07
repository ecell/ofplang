#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from collections.abc import Iterable, Iterator
from collections import OrderedDict

from ofplang.base.definitions import Definitions
from ofplang.base.protocol import EntityDescription, PortAddress, Port, PortConnection, Protocol

from ofplang.base.entity_type import EntityTypeLoader
from ofplang.base.validate import validate_definitions_post, validate_protocol_post

logger = getLogger(__name__)


class UntypedProcess:

    def __init__(self, entity: EntityDescription, definition: dict) -> None:
        self.__entity = entity
        self.__definition = definition

    def input(self) -> Iterable[tuple[PortAddress, Port]]:
        input = {
            PortAddress(self.__entity.id, port["id"]): Port(**port)
            for port in self.__definition.get("input", [])}
        return input.items()

    def output(self) -> Iterable[tuple[PortAddress, Port]]:
        output = {
            PortAddress(self.__entity.id, port["id"]): Port(**port)
            for port in self.__definition.get("output", [])}
        return output.items()

    def asentitydesc(self) -> EntityDescription:
        return self.__entity

    @property
    def id(self) -> str:
        return self.__entity.id

    @property
    def type(self) -> str:
        return self.__entity.type

    @property
    def definition(self) -> dict:
        return self.__definition.copy()  # deepcopy

class UntypedModel:

    def __init__(self, protocol: Protocol, definitions: Definitions) -> None:
        self.__protocol = protocol
        self.__definitions = definitions
        self.__load()

    @property
    def protocol(self) -> Protocol:
        return self.__protocol

    @property
    def definitions(self) -> Definitions:
        return self.__definitions
    
    def __load(self) -> None:
        self.__processes = OrderedDict()
        for process_desc, process_dict in self.__protocol.processes_with_dict():
            definition = self.__definitions.get_by_name(process_desc.type)

            if "input" in process_dict:
                input_defaults = {port["id"]: {"value": port["value"], "type": port["type"]} for port in process_dict["input"]}
                for port in definition["input"]:
                    if port["id"] in input_defaults:
                        port["default"] = input_defaults[port["id"]]

            process = UntypedProcess(process_desc, definition)
            self.__processes[process.id] = process

    def get_definition_by_name(self, name: str) -> dict:
        return self.__definitions.get_by_name(name)

    def get_by_id(self, id: str) -> UntypedProcess:
        return self.__processes[id]

    def connections(self) -> Iterator[PortConnection]:
        return self.__protocol.connections()

    def processes(self) -> Iterable[UntypedProcess]:
        return self.__processes.values()
    
    def input(self) -> Iterator[tuple[PortAddress, Port]]:
        #XXX: default?
        return ((PortAddress("input", port.id), port) for port in self.__protocol.input())

    def output(self) -> Iterator[tuple[PortAddress, Port]]:
        return ((PortAddress("output", port.id), port) for port in self.__protocol.output())

class Model(UntypedModel):

    def __init__(self, protocol: Protocol, definitions: Definitions) -> None:
        super().__init__(protocol, definitions)

        self.__loader = EntityTypeLoader(definitions)

        validate_definitions_post(definitions, self.__loader)
        validate_protocol_post(protocol, definitions, self.__loader)

    def issubclass(self, sub: str, sup: str) -> bool:
        sub_, sup_ = self.__loader.evaluate(sub), self.__loader.evaluate(sup)
        assert isinstance(sub_, type), repr(sub_)
        assert isinstance(sup_, type), repr(sub_)
        return issubclass(sub_, sup_)
    
    def is_acceptable(self, sub: str, sup: str) -> bool:
        return self.__loader.is_acceptable(sub, sup)
