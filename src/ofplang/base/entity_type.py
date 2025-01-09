#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from . import definitions
from ._entity_type import EntityType, Undefined, Object, Data, Process, Spread, Optional, Array, check_entity_type, is_data, is_object, is_acceptable

logger = getLogger(__name__)


class Number(Data): pass
class Float(Number): pass
class Integer(Number): pass
class String(Data): pass

class Labware(Object): pass
class SpotArray(Labware): pass

class IOProcess(Process): pass  #XXX
class BuiltinProcess(Process): pass
class RunScript(BuiltinProcess): pass

class TypeManager:

    def __init__(self, definitions: definitions.Definitions) -> None:
        self.__definitions = definitions

        self.__load_entity_types()

    def __load_entity_types(self) -> None:
        self.__primitive_types = {
            "Object": Object,
            "Data": Data,
            "Undefined": Undefined,
            "Process": Process,
            "Spread": Spread,
            "Optional": Optional,
            "Array": Array,

            "Number": Number,
            "Float": Float,
            "Integer": Integer,
            "String": String,

            "Labware": Labware,
            "SpotArray": SpotArray,

            "IOProcess": IOProcess,
            "BuiltinProcess": BuiltinProcess,
            "RunScript": RunScript,
            }

        for x in self.__definitions:
            name, base = x["name"], x["base"]
            assert name not in self.__primitive_types, name
            self.__primitive_types[name] = type(name, (self.__primitive_types[base], ), {})

    def has_definition(self, expression: str) -> bool:
        return (expression in self.__primitive_types)

    def eval_primitive_type(self, expression: str) -> type:
        assert self.has_definition(expression), f"Unknown entity type given [{expression}]."
        entity_type = self.__primitive_types[expression]
        assert issubclass(entity_type, (EntityType, )), f"[{expression}] is not an entity type."
        return entity_type

    def eval_entity_type(self, expression: str) -> type:
        # new_type = eval(expression, {"__builtins__": None}, self.__primitive_types)
        new_type = eval(expression, self.__primitive_types, {})
        check_entity_type(new_type)
        return new_type

    def is_data(self, one: str) -> bool:
        return is_data(self.eval_entity_type(one))

    def is_object(self, one: str) -> bool:
        return is_object(self.eval_entity_type(one))

    def is_acceptable(self, one: str, another: str) -> bool:
        return is_acceptable(self.eval_entity_type(one), self.eval_entity_type(another))

    def issubclass(self, one: str, another: str) -> bool:
        return issubclass(self.eval_primitive_type(one), self.eval_primitive_type(another))
