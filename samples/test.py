from ofplang.tools.graph import ProtocolGraph


protocol = ProtocolGraph("samples/definitions.yaml")

serve_plate1 = protocol.ServePlate96()
dispense_liquid1 = protocol.DispenseLiquid96Wells(in1=serve_plate1.value, volume=protocol.input.volume1, channel=protocol.input.channel1)
dispense_liquid2 = protocol.DispenseLiquid96Wells(in1=dispense_liquid1.out1, volume=protocol.input.volume2, channel=protocol.input.channel2)
read_absorbance1 = protocol.ReadAbsorbance3Colors(in1=dispense_liquid2.out1)
_ = protocol.StoreLabware(in1=read_absorbance1.out1)
protocol.output.data = read_absorbance1.value

protocol.dump()
