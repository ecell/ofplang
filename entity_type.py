#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import typing, types, inspect


class EntityType(type): pass
class Object(EntityType): pass
class Data(EntityType): pass
class Operation(type): pass  #XXX

class Spread(typing.Generic[typing.TypeVar("T")]): pass
class Optional(typing.Generic[typing.TypeVar("T")]): pass  # Do not use `T | None`. Use `Optional` instead.
class Array(typing.Generic[typing.TypeVar("D")]): pass

def is_union(x):
    return isinstance(x, types.UnionType) or isinstance(x, typing._UnionGenericAlias)

def is_generic(x):
    return isinstance(x, typing._GenericAlias) and x.__origin__ in (Spread, Optional, Array)

def is_acceptable(one, another) -> bool:
    # Union
    if is_union(one):
        return all(is_acceptable(x, another) for x in one.__args__)
    elif not is_union(one) and is_union(another):
        return any(is_acceptable(one, x) for x in another.__args__)

    # typing._GenericAlias
    if is_generic(one):
        if is_generic(another):
            return (
                is_acceptable(one.__origin__, another.__origin__)
                and len(one.__args__) == len(another.__args__)
                and all(is_acceptable(x, y) for x, y in zip(one.__args__, another.__args__))
            )
        else:
            assert inspect.isclass(another), f"{another}"
            return issubclass(one.__origin__, another)
    else:
        if is_generic(another):
            assert inspect.isclass(one), f"{one}"
            return False
        else:
            # class
            assert inspect.isclass(one), f"{one}"
            assert inspect.isclass(another), f"{another}"
            return issubclass(one, another)
    assert False, "Never get here."

def is_object(one):
    if is_union(one) or is_generic(one):
        return any(is_object(x) for x in one.__args__)
    return issubclass(one, Object)

def is_data(one):
    if is_union(one) or is_generic(one):
        return all(is_data(x) for x in one.__args__)
    return issubclass(one, Data)

def check_entity_type(one):
    if is_union(one) or is_generic(one):
        tuple(check_entity_type(x) for x in one.__args__)
    else:
        assert issubclass(one, (EntityType, ))

import definitions

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
        check_entity_type(new_type)
        return new_type

    def is_data(self, one: str) -> bool:
        return is_data(self.eval_entity_type(one))

    def is_acceptable(self, one: str, another: str) -> bool:
        return is_acceptable(self.eval_entity_type(one), self.eval_entity_type(another))


if __name__ == "__main__":
    class Labware(Object): pass

    # assert is_acceptable(Array, Data)
    # assert not is_acceptable(Array, Object)
    assert is_acceptable(Array, Labware | Array)
    assert not is_acceptable(Labware | Array, Array)

    assert is_acceptable(Spread, Spread)
    assert is_acceptable(Spread[Array], Spread)
    assert is_acceptable(Spread[Array], Spread[Array])
    assert is_acceptable(Spread[Array], Spread[Labware | Array])
    assert is_acceptable(Spread[Array], Spread[Array])
    # assert not is_acceptable(Spread[Array], Data)
    # assert is_acceptable(Spread[Array], Spread[Data])
    assert is_acceptable(Spread[Labware], Spread)
    assert not is_acceptable(Spread[Labware], Object)
    assert is_acceptable(Spread[Labware], Spread[Object])

    assert is_acceptable(Optional[Labware], Optional[Labware])
    assert is_acceptable(Optional[Labware], Optional[Object])
    assert not is_acceptable(Optional[Labware], Labware)
    assert not is_acceptable(Optional[Labware], Object)

    assert not is_acceptable(Spread[Optional[Labware]], Labware | Spread[Labware])
    assert not is_acceptable(Spread[Optional[Labware]], Spread[Labware])
    assert not is_acceptable(Spread[Optional[Labware]], Labware)

    class Scalar(Data): pass
    class Boolean(Scalar): pass
    class Integer(Scalar): pass
    class Float(Scalar): pass
    Real = Integer | Float

    assert not is_acceptable(Optional[Float], Float)
    assert not is_acceptable(Float, Optional[Float])
    assert is_acceptable(Optional[Float], Optional[Float])
    assert is_acceptable(Optional[Float], Optional[Real])
    assert not is_acceptable(Optional[Float], Optional[Object])
    assert not is_acceptable(Optional[Float], Data)
    assert not is_acceptable(Optional[Float], Object)
    assert is_acceptable(Optional[Float], Optional)
    assert not is_acceptable(Optional[Float], Spread)

    assert Spread == Spread
    assert Spread[Float] != Spread
    assert Spread[Float] not in (Spread, )

    assert is_object(Spread[Labware])
    assert is_object(Optional[Labware])
    assert not is_data(Spread[Labware])
    assert not is_data(Optional[Labware])
    assert is_object(Spread[Optional[Labware]])
    assert is_object(Spread[Spread[Labware | Float]])
    assert is_object(Spread[Float | Spread[Labware]])
    assert is_data(Spread[Float | Array[Integer]])
