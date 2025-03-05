#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from typing import Any
import numpy
import asyncio

from ..base.executor import ProcessNotSupportedError, ExecutorBase
from ..base.model import Model
from ..base.protocol import EntityDescription

from .simulator import DeckSimulator

logger = getLogger(__name__)

OPERATIONS_QUEUED: 'asyncio.Queue[dict]' = asyncio.Queue()

async def tecan_fluent_operator(simulation: bool = True):
    operator = Operator(simulation)
    operator.initialize()

    while True:
        while OPERATIONS_QUEUED.empty():
            await asyncio.sleep(10)
        job = await OPERATIONS_QUEUED.get()
        await operator.execute(**job)
        OPERATIONS_QUEUED.task_done()

class Operator:

    def __init__(self, simulation: bool = True) -> None:
        self.simulation = simulation
        self.__deck = DeckSimulator()
    
    @property
    def deck(self) -> DeckSimulator:
        return self.__deck  #XXX
    
    def initialize(self) -> None:
        self.__deck.initialize()
        for channel in range(6):
            self.__deck.new_falcon50(f'50ml FalconTube 6pos[{channel+1:03d}]')
    
    async def execute(self, future: asyncio.Future, model: 'Model', operation: str, inputs: dict, address: str) -> None:
        logger.info(f"execute [{operation}].")
        
        if not self.simulation:
            from . import tecan

        outputs = {}

        if operation == "ServePlate96":
            plate_id = self.__deck.new_plate96('7mm Nest_Riken[005]')
            outputs["value"] = {"value": {"id": plate_id}, "type": "Plate96"}
        elif operation == "StoreLabware":
            self.__deck.remove(inputs["in1"]["value"]["id"])  #XXX
        elif operation == "ReadAbsorbance3Colors":
            params: dict[str, Any] = {}  # noqa: F841

            if self.simulation:
                await asyncio.sleep(3)
                data = numpy.zeros(96, dtype=float)
            else:
                (data, ), _ = tecan.read_absorbance_3colors(**params)

            assert inputs["in1"]["type"] == "Plate96"
            outputs["value"] = {"value": data, "type": "Spread[Array[Float]]"}
            outputs["out1"] = inputs["in1"]
        elif operation == "DispenseLiquid96Wells":
            channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
            assert len(volume) == 96

            volume = numpy.asarray(volume)
            volume = volume.astype(int)
            params = {'data': volume, 'channel': channel}  # noqa: F841

            if self.simulation:
                await asyncio.sleep(5)
            else:
                _ = tecan.dispense_liquid_96wells(**params)
            
            outputs["out1"] = inputs["in1"]
            self.__deck.dispense_liquid_96wells(f"50ml FalconTube 6pos[{channel+1:03d}]", '7mm Nest_Riken[005]', volume)
        else:
            future.set_exception(ProcessNotSupportedError(f"Undefined process given [{operation}]."))
            return

        future.set_result(outputs)

class TecanFluentController(ExecutorBase):

    def queue_operation(self, model: 'Model', process: EntityDescription, inputs: dict, address: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = {'future': future, 'model': model, 'operation': process.type, 'inputs': inputs, 'address': address}
        OPERATIONS_QUEUED.put_nowait(job)
        return future

    async def execute(self, model: 'Model', process: EntityDescription, inputs: dict, job_id: str) -> dict:
        logger.info(f"TecanFluentSimulator.execute <= [{process}] [{inputs}]")

        outputs = {}

        try:
            outputs = await super().execute(model, process, inputs, job_id)
        except ProcessNotSupportedError as err:  # noqa: F841
            operation_id = self.store.create_operation({})
            outputs = await self.queue_operation(model, process, inputs, self.store.get_operation_uri(operation_id))
            self.store.update_operation(operation_id, {})

        logger.info(f"TecanFluentSimulator.execute => [{process}] [{outputs}]")
        return outputs