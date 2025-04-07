#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from typing import _GenericAlias, Type, Any  # type: ignore[attr-defined]

from . import definitions
from ._entity_type import Entity, Object, Data, Process, Spread, Optional, Array, check_entity_type, is_data, is_object, is_acceptable

logger = getLogger(__name__)


class Number(Data): pass
class Float(Number): pass
class Integer(Number): pass
class Boolean(Data): pass
class String(Data): pass

class Labware(Object): pass
# class SpotArray(Labware): pass

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
            # "Undefined": Undefined,
            "Process": Process,
            "Spread": Spread,
            "Optional": Optional,
            "Array": Array,

            "Number": Number,
            "Float": Float,
            "Integer": Integer,
            "Boolean": Boolean,
            "String": String,

            "Labware": Labware,
            # "SpotArray": SpotArray,

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
        assert issubclass(entity_type, (Entity, )), f"[{expression}] is not an entity type."
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

class EntityTypeLoader:

    BUILTIN_TYPES = dict(
        Entity=Entity, Object=Object, Data=Data, Process=Process,
        Spread=Spread, Optional=Optional, Array=Array,
        )

    def __init__(self, definitions: definitions.Definitions | None = None) -> None:
        self.__definitions = definitions

        self.__primitive_types = self.BUILTIN_TYPES.copy()
        self.__load_primitive_types()
    
    def __load_primitive_types(self) -> None:
        if self.__definitions is not None:
            for x in self.__definitions:
                name, base = x["name"].strip(), x["base"].strip()
                assert name not in self.__primitive_types, f"Type '{name}' is already defined"
                assert base in self.__primitive_types, f"The base type of '{name}', '{base}', is not defined."
                self.__primitive_types[name] = type(name, (self.__primitive_types[base], ), {})

    def __evaluate(self, expression: str) -> Any:
        return eval(expression, self.__primitive_types, {})

    def evaluate(self, expression: str) -> Type[Entity] | _GenericAlias:
        obj = self.__evaluate(expression)
        assert check_entity_type(obj), f"{repr(obj)}"
        return obj  # type: ignore

    def is_valid(self, expression: str, primitive: bool = False) -> bool:
        try:
            obj = self.__evaluate(expression)
        except Exception as e:
            logger.error(str(e))
            return False
        if not check_entity_type(obj):
            return False
        elif primitive and not isinstance(obj, type):
            logger.error(f"'{expression}' is an entity type, but not primitive.")
            return False
        return True

    def is_acceptable(self, sub: str, sup: str) -> bool:
        return is_acceptable(self.evaluate(sub), self.evaluate(sup))

    def is_data(self, expression: str) -> bool:
        return is_data(self.evaluate(expression))


if __name__ == "__main__":
    loader = EntityTypeLoader()

    print(repr(loader.evaluate("Object")))
    print(repr(loader.evaluate("Process")))
    print(repr(loader.evaluate("Array[Object]")))
    print(repr(loader.evaluate("Spread[Array[Object]]")))
