#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import definitions

import _entity_type
from _entity_type import Object, Data, Operation, Spread, Optional, Array

class TypeChecker:

    def __init__(self, definitions: definitions.Definitions) -> None:
        self.__definitions = definitions

        self.__load_entity_types()
    
    def __load_entity_types(self) -> None:
        self.__primitive_types = {
            "Object": Object,
            "Data": Data,
            "Operation": Operation,
            "Spread": Spread,
            "Optional": Optional,
            "Array": Array
            }
        
        for x in self.__definitions:
            name, ref = x["name"], x["ref"]
            assert name not in self.__primitive_types, name
            self.__primitive_types[name] = type(name, (self.__primitive_types[ref], ), {})

    def eval_entity_type(self, expression: str) -> type:
        new_type = eval(expression, {"__builtins__": None}, self.__primitive_types)
        _entity_type.check_entity_type(new_type)
        return new_type

    def is_data(self, one: str) -> bool:
        return _entity_type.is_data(self.eval_entity_type(one))

    def is_object(self, one: str) -> bool:
        return _entity_type.is_object(self.eval_entity_type(one))

    def is_acceptable(self, one: str, another: str) -> bool:
        return _entity_type.is_acceptable(self.eval_entity_type(one), self.eval_entity_type(another))
