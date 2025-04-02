#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger, StreamHandler, Formatter

import inspect
from typing import Any
from io import StringIO
import numpy
import asyncio

from ..base.executor import ProcessNotSupportedError, ExecutorBase
from ..base.model import Model
from ..base.protocol import EntityDescription

from .simulator import DeckEditor, DeckView

logger = getLogger(__name__)


class Operator:

    def __init__(self, simulation: bool = True, deck: DeckEditor | None = None) -> None:
        self.simulation = simulation
        self.__deck = deck or DeckEditor()
        self.__OPERATIONS_QUEUED: 'asyncio.Queue[dict]' = asyncio.Queue()

    def start(self) -> asyncio.Task:
        return asyncio.create_task(self.run())

    async def run(self) -> None:
        self.initialize()

        while True:
            while self.__OPERATIONS_QUEUED.empty():
                await asyncio.sleep(10)
            job = await self.__OPERATIONS_QUEUED.get()
            await self.execute(**job)
            self.__OPERATIONS_QUEUED.task_done()

    def put_nowait(self, job: dict) -> None:
        self.__OPERATIONS_QUEUED.put_nowait(job)

    @property
    def deck(self) -> DeckView:
        return self.__deck.view()
    
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
                await asyncio.sleep(10)
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
                await asyncio.sleep(15)
            else:
                _ = tecan.dispense_liquid_96wells(**params)
            
            outputs["out1"] = inputs["in1"]
            self.__deck.dispense_liquid_96wells(f"50ml FalconTube 6pos[{channel+1:03d}]", '7mm Nest_Riken[005]', volume)
        else:
            future.set_exception(ProcessNotSupportedError(f"Undefined process given [{operation}]."))
            return

        future.set_result(outputs)

class TecanFluentController(ExecutorBase):

    def __init__(self, operator: Operator | None = None) -> None:
        if operator is None:
            self.__operator = Operator()
            self.__operator.start()
        else:
            self.__operator = operator

    def queue_operation(self, model: 'Model', operation: str, inputs: dict, address: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = {'future': future, 'model': model, 'operation': operation, 'inputs': inputs, 'address': address}
        self.__operator.put_nowait(job)
        return future

    async def execute(self, model: 'Model', process: EntityDescription, inputs: dict, job_id: str, run_id: str) -> dict:
        logger.info(f"TecanFluentSimulator.execute <= [{process}] [{inputs}]")
        operation_id = self.store.create_operation(dict(process_id=job_id, name=process.type))

        mylogger = getLogger(f"{run_id}.{operation_id}")
        stream = StringIO()
        mylogger.addHandler(StreamHandler(stream))
        
        mylogger.info(f"process={str(process)}")
        mylogger.info(f"inputs={str(inputs)}")
        mylogger.info(f"job_id={job_id}")
        mylogger.info(f"run_id={run_id}")

        self.store.set_operation_attribute(operation_id, "log", stream.getvalue())

        outputs = {}

        try:
            outputs = await super().execute(model, process, inputs, job_id, run_id)
        except ProcessNotSupportedError as err:  # noqa: F841
            outputs = await self.queue_operation(model, process.type, inputs, self.store.get_operation_uri(operation_id))

        logger.info(f"TecanFluentSimulator.execute => [{process}] [{outputs}]")
        self.store.finish_operation(operation_id)

        mylogger.info(f"outputs={str(outputs)}")
        self.store.set_operation_attribute(operation_id, "log", stream.getvalue())  # => process
        self.artifact_store.log_text(stream.getvalue(), f"{self.store.get_operation_uri(operation_id)}/log.txt")

        return outputs