#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import uuid
from dataclasses import dataclass, field
from collections import defaultdict
import numpy


logger = getLogger(__name__)

@dataclass
class Labware:
    id: str

@dataclass
class Plate96(Labware):
    contents: defaultdict[str | int, numpy.ndarray] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

@dataclass
class Falcon50(Labware):
    contents: float = 0.0

class Deck:

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.__labwares: dict[str, Labware] = {}
        self.__addresses: dict[str, str] = {}
        self.__sites: dict[str, list[str]] = {}
        self.__stackable: dict[str, bool] = {}

    def __getitem__(self, key: str) -> str:
        #XXX
        return self.__sites[key][-1]
    
    def sites(self):
        return self.__sites.keys()
    
    def add_site(self, address: str, stackable: bool = False) -> None:
        assert address not in self.__sites
        self.__sites[address] = []
        self.__stackable[address] = stackable

    def get(self, id: str) -> Labware:
        return self.__labwares[id]
    
    def add(self, address: str, obj: Labware) -> None:
        assert obj.id not in self.__labwares
        assert address in self.__sites
        assert self.__stackable[address] or len(self.__sites[address]) == 0
        self.__labwares[obj.id] = obj
        self.__sites[address].append(obj.id)
        self.__addresses[obj.id] = address
    
    def where(self, id: str) -> str | None:
        return self.__addresses.get(id, None)

    def pop(self, address: str) -> Labware:
        labware_id = self.__sites[address].pop()
        del self.__addresses[labware_id]
        return self.__labwares.pop(labware_id)
    
    def move(self, src: str, dst: str) -> None:
        assert src in self.__sites and len(self.__sites[src]) > 0
        assert dst in self.__sites
        assert self.__stackable[dst] or len(self.__sites[dst]) == 0
        self.add(dst, self.pop(src))

class DeckView:

    def __init__(self, deck: Deck) -> None:
        self.__deck = deck

    def __getitem__(self, key: str) -> str:
        return self.__deck[key]
    
    def sites(self):
        return self.__deck.keys()

    def get(self, id: str) -> Labware:
        return self.__deck.get(id)
    
    def where(self, id: str) -> str | None:
        return self.where(id)

class DeckEditor:

    def __init__(self):
        self.__deck = Deck()
    
    def initialize(self):
        self.__deck.clear()

        self.__deck.add_site('7mm Nest_Riken[005]')
        for channel in range(6):
            address = f'50ml FalconTube 6pos[{channel+1:03d}]'
            self.__deck.add_site(address)

    def view(self) -> DeckView:
        return DeckView(self.__deck)

    def add_site(self, address: str, stackable: bool = False) -> None:
        self.__deck.add_site(address, stackable)

    def new_plate96(self, address: str, id: str | None = None) -> str:
        return self.__new_labware(Plate96, address, id)

    def new_falcon50(self, address: str, id: str | None = None) -> str:
        return self.__new_labware(Falcon50, address, id)

    def __new_labware(self, labware_type: type, address: str, id: str | None = None) -> str:
        labware_id = id or str(uuid.uuid4())
        assert self.__deck.where(labware_id) is None
        labware = labware_type(labware_id)
        self.__deck.add(address, labware)
        return labware_id
    
    def remove(self, id: str) -> None:
        address = self.__deck.where(id)
        assert address is not None, id
        _ = self.__deck.pop(address)

    def dispense_liquid_96wells(self, channel: str, address: str, volume) -> None:
        plate96_id = self.__deck[address]
        plate = self.__deck.get(plate96_id)
        assert isinstance(plate, Plate96)
        tube50_id = self.__deck[channel]
        falcon = self.__deck.get(tube50_id)
        assert isinstance(falcon, Falcon50)
        plate.contents[falcon.id] += volume
        falcon.contents -= sum(volume)
    
    def supply_liquid_falcon50(self, id: str, volume: float) -> None:
        falcon = self.__deck.get(id)
        assert isinstance(falcon, Falcon50)
        falcon.contents += volume