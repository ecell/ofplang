#!/usr/bin/python
# -*- coding: utf-8 -*-

import uuid
from collections import defaultdict
import numpy

from runner import Runner, Token
from protocol import Entity, PortAddress


class Simulator:

    def __init__(self):
        pass

    def __call__(self, runner: Runner, tasks: list[tuple[Entity, dict]]) -> None:
        for operation, inputs in tasks:
            outputs = self.execute(operation, inputs)
            runner.add_tokens([
                Token(PortAddress(operation.id, key), value)
                for key, value in outputs.items()])
            
            if operation.type == "ServePlate96":  #XXX
                runner.deactivate(operation.id)

    def execute(self, operation: Entity, inputs: dict) -> None:
        outputs = {}
        if operation.type == "ServePlate96":
            value = {"id": str(uuid.uuid4()), "contents": defaultdict(lambda: numpy.zeros(96, dtype=float))}
            outputs["value"] = {"value": value, "type": "Plate96"}
        elif operation.type == "StoreLabware":
            pass
        elif operation.type == "DispenseLiquid96Wells":
            channel, volume = inputs["channel"]["value"], inputs["volume"]["value"]
            value = inputs["in1"]["value"]  # deepcopy?
            value["contents"][channel] += volume
            outputs["out1"] = {"value": value, "type": "Plate96"}
        elif operation.type == "ReadAbsorbance3Colors":
            outputs["out1"] = inputs["in1"]

            # value = [numpy.zeros(96, dtype=float)]  #XXX
            contents = sum(inputs["in1"]["value"]["contents"].values())
            value = contents ** 3 / (contents ** 3 + 100.0 ** 3)  # Sigmoid
            value += numpy.random.normal(scale=0.05, size=value.shape)
            
            outputs["value"] = {"value": [value], "type": "Spread[Array[Float]]"}
        else:
            raise RuntimeError(f"Undefined operation given [{operation.type}].")
        return outputs
