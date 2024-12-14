import io
from ofplang.prelude import *
from ofplang.base.validate import check_definitions, check_protocol

import pytest


@pytest.fixture
def init_definitions():
    definitions = Definitions(io.StringIO("""
        - name: Plate96
          ref: Labware
          description: A 96-well plate.
        - name: Unit
          ref: String
        - name: ServePlate96
          ref: IOProcess
          description: Provide an empty 96-well plate.
          output:
            - id: value
              type: Plate96
        - name: StoreLabware
          ref: IOProcess
          description: Store the 96-well plate in the designated location. Or simply discard.
          input:
            - id: in1
              type: Labware
            - id: where
              type: String
              default:
                value: ""
                type: String
        - name: LabwareToSpotArray
          ref: BuiltinProcess
          description: ""
          input:
            - id: in1
              type: Labware
            - id: indices
              type: Array[Integer]
          output:
            - id: out1
              type: SpotArray
        - name: DispenseLiquid96Wells
          ref: Process
          description: Dispense liquid into the 96-well plate.
          input:
            - id: in1
              type: SpotArray | Plate96
            - id: channel
              type: Integer
              default:
                value: 0
                type: Integer
            - id: volume
              type: Array[Float]
            - id: unit
              type: Unit
              default:
                value: microliter
                type: Unit
          output:
            - id: out1
              type: Plate96
        - name: ReadAbsorbance3Colors
          ref: Process
          description: Measure the absorbance of the 96-well plate.
          input:
            - id: in1
              type: SpotArray | Plate96
          output:
            - id: out1
              type: Plate96
            - id: value
              type: Spread[Array[Float]]
        """))
    return definitions

def test_definitions(init_definitions):
    check_definitions(init_definitions)

def test_protocol1(init_definitions):
    protocol = Protocol(io.StringIO("""
        name: test1
        contents:
          input: []
          output: []
          processes: []
          connections: []
        """))
    check_protocol(protocol, init_definitions)