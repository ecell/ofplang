#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

logger = getLogger(__name__)

import typing, types, inspect


class _EntityTypeMeta(type):
    
    def __instancecheck__(self, obj):
        return super().__instancecheck__(obj)

    def __repr__(self):
        return f'{self.__module__}.{self.__qualname__}'
        # return super().__repr__()  # respect to subclasses

class EntityType(metaclass=_EntityTypeMeta):
    pass

class Any(EntityType, typing.Generic[typing.TypeVar("T")]): pass
class Spread(EntityType, typing.Generic[typing.TypeVar("T")]): pass
class Optional(EntityType, typing.Generic[typing.TypeVar("T")]): pass

# See typing._TupleType
class _StructType(typing._SpecialGenericAlias, _root=True):
    @typing._tp_cache
    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) >= 2 and params[-1] is ...:
            msg = "Struct[t, ...]: t must be a type."
            params = tuple(typing._type_check(p, msg) for p in params[:-1])
            return self.copy_with((*params, typing._TypingEllipsis))
        msg = "Struct[t0, t1, ...]: each t must be a type."
        params = tuple(typing._type_check(p, msg) for p in params)
        return self.copy_with(params)
class _Struct(EntityType): pass
Struct = _StructType(_Struct, -1, inst=False, name='Struct')

class Object(EntityType): pass
class Data(EntityType): pass
class Array(Data, typing.Generic[typing.TypeVar("T")]): pass

# def get_primitive_type_by_id(id: str) -> EntityType | None:
#     return {"Data": Data, "Object": Object}.get(id, None)

def is_union(x):
    return isinstance(x, types.UnionType) or isinstance(x, typing._UnionGenericAlias)

def is_any(x):
    return isinstance(x, typing._GenericAlias) and x.__origin__ == Any

def is_spread(x):
    return isinstance(x, typing._GenericAlias) and x.__origin__ == Spread

def is_optional(x):
    return isinstance(x, typing._GenericAlias) and x.__origin__ == Optional

def is_struct(x):
    return x is Struct or isinstance(x, typing._GenericAlias) and x.__origin__ == _Struct

def is_array(x):
    return isinstance(x, typing._GenericAlias) and x.__origin__ == Array

def is_acceptable(one, another) -> bool:
    # Union
    if is_union(one):
        return all(is_acceptable(x, another) for x in one.__args__)
    elif not is_union(one) and is_union(another):
        return any(is_acceptable(one, x) for x in another.__args__)

    # typing._GenericAlias
    if isinstance(one, typing._GenericAlias) and isinstance(another, typing._GenericAlias):
        if another.__origin__ == Any:
            # assert len(one.__args__) == 1, f"{one}"  #FIXME:
            # return is_acceptable(one.__args__[0], another)
            return any(is_acceptable(x, another) for x in one.__args__)
        else:
            return (
                is_acceptable(one.__origin__, another.__origin__)
                and len(one.__args__) == len(another.__args__)
                and all(is_acceptable(x, y) for x, y in zip(one.__args__, another.__args__))
            )
    elif isinstance(one, typing._GenericAlias) and not isinstance(another, typing._GenericAlias):
        assert inspect.isclass(another), f"{another}"
        return issubclass(one.__origin__, another)
    elif not isinstance(one, typing._GenericAlias) and isinstance(another, typing._GenericAlias):
        assert inspect.isclass(one), f"{one}"
        if another.__origin__ == Any:
            assert len(another.__args__) == 1, f"{another}"
            return is_acceptable(one, another.__args__[0])
        else:
            return False

    # class
    assert inspect.isclass(one), f"{one}"
    assert inspect.isclass(another), f"{another}"
    return issubclass(one, another)

def is_object(one):
    return is_acceptable(one, Any[Object])

def is_data(one):
    return is_acceptable(one, Any[Data])

def first_arg(x):
    assert isinstance(x, typing._GenericAlias), str(x)
    return x.__args__[0]


if __name__ == "__main__":
    class Labware(Object): pass

    assert is_acceptable(Array, Data)
    assert not is_acceptable(Array, Object)
    assert is_acceptable(Array, Labware | Array)
    assert not is_acceptable(Labware | Array, Array)

    assert is_acceptable(Spread, Spread)
    assert is_acceptable(Spread[Array], Spread)
    assert is_acceptable(Spread[Array], Spread[Array])
    assert is_acceptable(Spread[Array], Spread[Labware | Array])
    assert is_acceptable(Spread[Array], Spread[Array])
    assert not is_acceptable(Spread[Array], Data)
    assert is_acceptable(Spread[Array], Spread[Data])
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
    assert not is_object(Spread[Spread[Labware | Float]])
    assert not is_object(Spread[Float | Spread[Labware]])
    assert is_data(Spread[Float | Array[Integer]])

    assert is_struct(Struct)
    assert is_struct(Struct[Labware])
    assert is_acceptable(Struct[Labware, Array[Float]], _Struct)  #XXX:
    assert is_acceptable(Struct[Labware, Array[Float]], Struct[Labware, Array])
    assert not is_acceptable(Struct[Labware, Array[Float]], Struct[Labware])
    assert is_object(Struct[Labware, Array[Float]])
    assert is_data(Struct[Labware, Array[Float]])
