#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import typing
import types
import inspect

from . import definitions

logger = getLogger(__name__)


class Entity(type): pass
class Object(Entity): pass
class Data(Entity): pass
class Process(Entity): pass  #XXX

T = typing.TypeVar("T", covariant=True)
class Generic_(typing.Generic[T]): pass
class Spread(Generic_[T], typing.Generic[T]): pass
class Optional(Generic_[T], typing.Generic[T]): pass  # Do not use `T | None`. Use `Optional` instead.
class Array(Generic_[T], typing.Generic[T]): pass

def is_union(one: typing.Any) -> bool:
    #XXX: Object | Data: <class 'types.UnionType'>
    #XXX: typing.get_origin(Object | Data)=<class 'types.UnionType'>
    #XXX: Data | Array[Data] == typing.Union[Data, Array[Data]]: <class 'typing._UnionGenericAlias'>
    #XXX: typing.get_origin(Data | Array[Data])=typing.Union
    return isinstance(one, types.UnionType) or isinstance(one, typing._UnionGenericAlias)  # type: ignore[attr-defined]

def is_generic(one: typing.Any) -> bool:
    #XXX: typing._UnionGenericAlias is typing._GenericAlias,
    #XXX: but get_origin returns typing.Union, which doesn't work with issubclass.
    return (
        not isinstance(one, typing._UnionGenericAlias)  # type: ignore[attr-defined]
        and isinstance(one, typing._GenericAlias)  # type: ignore[attr-defined]
        and issubclass(typing.get_origin(one), Generic_)
    )

def check_entity_type(one: typing.Any) -> bool:
    # logger.info(f"one={repr(one)}: {type(one)}")
    if isinstance(one, type):
        return issubclass(one, Entity)
    elif is_union(one):
        # logger.info(f"origin={repr(typing.get_origin(one))}")
        return all(check_entity_type(x) for x in typing.get_args(one))
    elif is_generic(one):
        return all(check_entity_type(x) for x in typing.get_args(one))
    return False

def is_acceptable(sub: typing.Any, sup: typing.Any) -> bool:
    """Check if sub is a subtype of sup for generic types."""
    # Union
    if is_union(sub):
        return all(is_acceptable(x, sup) for x in typing.get_args(sub))
    elif is_union(sup):
        return any(is_acceptable(sub, x) for x in typing.get_args(sup))

    # typing._GenericAlias
    if is_generic(sub):
        if is_generic(sup):
            return (
                is_acceptable(sub.__origin__, sup.__origin__)
                and len(sub.__args__) == len(sup.__args__)
                and all(is_acceptable(x, y) for x, y in zip(sub.__args__, sup.__args__))
            )
        else:
            assert inspect.isclass(sup), f"{sup}"
            return issubclass(sub.__origin__, sup)
    else:
        if is_generic(sup):
            assert inspect.isclass(sub), f"{sub}"
            return False
        else:
            # class
            assert inspect.isclass(sub), f"{sub}"
            assert inspect.isclass(sup), f"{sup}"
            return issubclass(sub, sup)
    assert False, "Never get here."

def is_object(one: typing.Any) -> bool:
    if is_union(one) or is_generic(one):
        return any(is_object(x) for x in one.__args__)
    return issubclass(one, Object)

def is_data(one: typing.Any) -> bool:
    if is_union(one) or is_generic(one):
        return all(is_data(x) for x in one.__args__)
    return issubclass(one, Data)

def to_str(one: typing.Any) -> str:
    if is_union(one):
        return " | ".join(to_str(x) for x in typing.get_args(one))    
    elif is_generic(one):
        s = ", ".join(to_str(x) for x in typing.get_args(one))
        origin = typing.get_origin(one)
        assert issubclass(origin, Generic_), f"{origin}"
        return f"{origin.__name__}[{s}]"

    assert issubclass(one, Entity), f"{one}"
    return one.__name__

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

    def __evaluate(self, expression: str) -> typing.Any:
        return eval(expression, self.__primitive_types, {})

    def evaluate(self, expression: str) -> typing.Type[Entity] | typing._GenericAlias:  # type: ignore[name-defined]
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
    import logging
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(level=logging.INFO)

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

    assert to_str(Spread[Float | Array[Integer]]) == "Spread[Float | Array[Integer]]"

    assert check_entity_type(Float | Integer)
    assert check_entity_type(Float | Array[Float])
    
    # loader = EntityTypeLoader()
    # print(repr(loader.evaluate("Object")))
    # print(repr(loader.evaluate("Process")))
    # print(repr(loader.evaluate("Array[Object]")))
    # print(repr(loader.evaluate("Spread[Array[Object]]")))
