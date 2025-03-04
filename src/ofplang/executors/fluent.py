#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from typing import Any
import numpy
import asyncio

from ..base.executor import OperationNotSupportedError
from ..base.model import Model
from ..base.protocol import EntityDescription

from .simulator import SimulatorBase, DeckSimulator
from .builtin import BuiltinExecutor

logger = getLogger(__name__)

OPERATIONS_QUEUED: 'asyncio.Queue[dict]' = asyncio.Queue()

async def tecan_fluent_operator():
    while True:
        while OPERATIONS_QUEUED.empty():
            await asyncio.sleep(10)
        job = await OPERATIONS_QUEUED.get()
        await execute(**job)
        OPERATIONS_QUEUED.task_done()

async def execute(future: asyncio.Future, model: 'Model', process: EntityDescription, inputs: dict) -> None:
    logger.info(f"execute [{process.id}] [{process.type}].")
    # from . import tecan

    outputs = {}

    if process.type == "ReadAbsorbance3Colors":
        params: dict[str, Any] = {}  # noqa: F841
        # (data, ), _ = tecan.read_absorbance_3colors(**params)
        await asyncio.sleep(10)
        data = numpy.zeros(96, dtype=float)

        assert inputs["in1"]["type"] == "Plate96"
        outputs["value"] = {"value": data, "type": "Spread[Array[Float]]"}
        outputs["out1"] = inputs["in1"]
    elif process.type == "DispenseLiquid96Wells":
        channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
        assert len(volume) == 96

        volume = numpy.asarray(volume)
        volume = volume.astype(int)
        params = {'data': volume, 'channel': channel}  # noqa: F841
        # _ = tecan.dispense_liquid_96wells(**params)
        await asyncio.sleep(15)
        outputs["out1"] = inputs["in1"]
    else:
        future.set_exception(OperationNotSupportedError(f"Undefined process given [{process.id}, {process.type}]."))
        return

    future.set_result(outputs)

class TecanFluentSimulator(BuiltinExecutor):

    def __init__(self) -> None:
        self.__deck = DeckSimulator()

    def initialize(self) -> None:
        super().initialize()

        self.__deck.initialize()
        for channel in range(6):
            self.__deck.new_falcon50(f'50ml FalconTube 6pos[{channel+1:03d}]')

    async def execute(self, model: 'Model', process: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."
        logger.info(f"TecanFluentSimulator.execute <= [{process}] [{inputs}]")

        outputs = {}

        if process.type == "ServePlate96":
            plate_id = self.__deck.new_plate96('7mm Nest_Riken[005]')
            outputs["value"] = {"value": {"id": plate_id}, "type": "Plate96"}
        elif process.type == "StoreLabware":
            self.__deck.remove(inputs["in1"]["value"]["id"])  #XXX
        else:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            job = {'future': future, 'model': model, 'process': process, 'inputs': inputs}
            OPERATIONS_QUEUED.put_nowait(job)
            outputs = await future

            if process.type == "DispenseLiquid96Wells":
                channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
                self.__deck.dispense_liquid_96wells(f"50ml FalconTube 6pos[{channel+1:03d}]", '7mm Nest_Riken[005]', volume)

        logger.info(f"TecanFluentSimulator.execute => [{process}] [{outputs}]")
        return outputs

class TecanFluentController(SimulatorBase):

    async def execute(self, model: 'Model', process: EntityDescription, inputs: dict, outputs_training: dict | None = None) -> dict:
        assert outputs_training is None, "'teach' is not supported."
        from . import tecan

        try:
            outputs = await super().execute(model, process, inputs, outputs_training)
        except OperationNotSupportedError as err:
            outputs = {}
            if process.type == "ReadAbsorbance3Colors":
                params: dict[str, Any] = {}
                (data, ), _ = tecan.read_absorbance_3colors(**params)
                if inputs["in1"]["type"] == "Plate96":
                    pass
                else:
                    assert inputs["in1"]["type"] == "SpotArray"
                    indices = inputs["in1"]["value"]["indices"]
                    data = [data[0][indices], data[1][indices], data[2][indices]]
                outputs["value"] = {"value": data, "type": "Spread[Array[Float]]"}
                outputs["out1"] = inputs["in1"]
            elif process.type == "DispenseLiquid96Wells":
                channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
                if inputs["in1"]["type"] == "Plate96":
                    assert len(volume) == 96
                else:
                    indices = inputs["in1"]["value"]["indices"]
                    assert len(volume) == len(indices)
                    volume, tmp = numpy.zeros(96), volume
                    volume[indices] = tmp

                volume = volume.astype(int)
                params = {'data': volume, 'channel': channel}
                _ = tecan.dispense_liquid_96wells(**params)
            else:
                raise err
        return outputs
