#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import uuid
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import numpy

from ..base.executor import OperationNotSupportedError
from ..base.model import Model
from ..base.protocol import EntityDescription

from .builtin import BuiltinExecutor

logger = getLogger(__name__)

@dataclass
class Labware:
    id: str

@dataclass
class Plate96(Labware):
    contents: defaultdict[int, numpy.ndarray] = field(default_factory=lambda: defaultdict(lambda: numpy.zeros(96, dtype=float)))

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

    def __getitem__(self, key: str) -> list[Labware]:
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

class DeckSimulator:

    def __init__(self):
        self.__deck = Deck()
    
    def initialize(self):
        self.__deck.clear()

        self.__deck.add_site('7mm Nest_Riken[005]')
        for channel in range(6):
            address = f'50ml FalconTube 6pos[{channel+1:03d}]'
            self.__deck.add_site(address)
    
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

class SimulatorBase(BuiltinExecutor):

    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        super().initialize()

        # global state
        self.__plates: dict[str, Plate96] = {}
        self.__liquids: defaultdict[int, float] = defaultdict(float)  # stock

    def new_plate(self, plate_id: str | None = None) -> str:
        plate_id = plate_id or str(uuid.uuid4())
        assert plate_id not in self.__plates
        self.__plates[plate_id] = Plate96(plate_id)
        return plate_id

    def get_plate(self, plate_id: str) -> Plate96:
        return self.__plates[plate_id]

    async def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        logger.info(f"execute: {(operation, inputs)}")

        try:
            outputs = await super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ServePlate96":
                plate_id = self.new_plate(None if outputs_training is None else outputs_training["value"]["value"]["id"])
                outputs["value"] = {"value": {"id": plate_id}, "type": "Plate96"}
            elif operation.type == "StoreLabware":
                pass
            elif operation.type in "DispenseLiquid96Wells":
                channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
                plate_id = inputs["in1"]["value"]["id"]
                assert len(volume) == 96, f"The length of volume must be 96. [{len(volume)}] was given."
                # if inputs["in1"]["type"] == "Plate96":
                #     assert len(volume) == 96, f"The length of volume must be 96. [{len(volume)}] was given."
                # else:
                #     indices = inputs["in1"]["value"]["indices"]
                #     assert len(volume) == len(indices), f"The length of volume have to be the same with indices [{len(volume)} != {len(indices)}]."
                #     volume, tmp = numpy.zeros(96), volume
                #     volume[indices] = tmp
                self.get_plate(plate_id).contents[channel] += volume
                self.__liquids[channel] += sum(volume)
                outputs["out1"] = inputs["in1"]
            elif operation.type in "DispenseLiquid96Wells__Optional":
                if inputs["in1"]["value"] is not None:
                    channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
                    plate_id = inputs["in1"]["value"]["id"]
                    assert len(volume) == 96, f"The length of volume must be 96. [{len(volume)}] was given."
                    # if inputs["in1"]["type"] == "Plate96":
                    #     assert len(volume) == 96, f"The length of volume must be 96. [{len(volume)}] was given."
                    # else:
                    #     indices = inputs["in1"]["value"]["indices"]
                    #     assert len(volume) == len(indices), f"The length of volume have to be the same with indices [{len(volume)} != {len(indices)}]."
                    #     volume, tmp = numpy.zeros(96), volume
                    #     volume[indices] = tmp
                    self.get_plate(plate_id).contents[channel] += volume
                    self.__liquids[channel] += sum(volume)
                outputs["out1"] = inputs["in1"]
            # elif operation.type == "Sleep":
            #     duration = inputs["duration"]["value"]
            #     await asyncio.sleep(duration)
            #     outputs["out1"] = inputs["in1"]
            # elif operation.type == "Gather":
            #     outputs["out1"] = inputs["in1"]
            #     outputs["out2"] = inputs["in2"]
            else:
                raise err
        return outputs
class Simulator(SimulatorBase):

    async def execute(self, model: 'Model', operation: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."

        try:
            outputs = await super().execute(model, operation, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if operation.type == "ReadAbsorbance3Colors":
                plate_id = inputs["in1"]["value"]["id"]
                # start = numpy.zeros(96, dtype=float)  # self.get_plate(plate_id).contents.default_factory()
                # contents = sum(self.get_plate(plate_id).contents.values(), start)
                # contents = self.get_plate(plate_id).contents[2]
                contents = self.get_plate(plate_id).contents
                print(contents)

                value1 = contents[1]
                value2 = contents[2]
                value3 = contents[3]
                # x = numpy.zeros(96, dtype=float)
                # if 1 in contents:
                #     x += contents[1] * 1.0
                # if 2 in contents:
                #     x += contents[2] * 1.0
                # value1 = 30 * numpy.cos(x / 10.0 * numpy.pi) + 50  # Cosine
                # value1 += numpy.random.normal(scale=0.1, size=value1.shape)

                # x = numpy.zeros(96, dtype=float)
                # if 1 in contents:
                #     x += contents[1] * 0.2
                # if 2 in contents:
                #     x += contents[2] * 1.8
                # value2: numpy.ndarray = 100 * x / (x + 180.0) + 50  # Sigmoid
                # value2 += numpy.random.normal(scale=0, size=value2.shape)

                # x = numpy.zeros(96, dtype=float)
                # if 1 in contents:
                #     x += contents[1] * 1.8
                # if 2 in contents:
                #     x += contents[2] * 0.2
                # value3: numpy.ndarray = 30 * (numpy.sin(x / 50.0 * numpy.pi) + 1.0) + 15  # Sin
                # value3 += numpy.random.normal(scale=0, size=value3.shape)

                # if inputs["in1"]["type"] == "Plate96":
                #     pass
                # else:
                #     assert inputs["in1"]["type"] == "SpotArray"
                #     indices = inputs["in1"]["value"]["indices"]
                #     value1, value2, value3 = value1[indices], value2[indices], value3[indices]

                outputs["value"] = {"value": [value1, value2, value3], "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]
            else:
                raise err
        return outputs