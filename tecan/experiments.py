import logging as _logging
from logging import getLogger, INFO, StreamHandler, Formatter
logger = getLogger(__name__)
logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

import numpy
import pathlib
import time
import datetime

def wait_until(fluent, *status, interval=3):
    #RunModeRunFinished
    #RunModePreRunChecks
    #RunModePreparingRun
    #RunModeWaitingForSystem
    #RunModeBeginRun
    #RunModeRunning
    #EditMode
    #RunModeStopOnError  #TODO
    while fluent.state not in status:
        logger.debug(f"Waiting... [{fluent.state}]")
        time.sleep(interval)

def call_method(fluent, methodname):
    assert fluent.state in ("RunModeRunFinished", "EditMode")

    def state_changed_callback(state):
        # str
        logger.info("New state is " + str(state))
    state_subscription = fluent.subscribe_state(state_changed_callback)

    def progress_changed_callback(progress):
        # int
        logger.info("New progress is " + str(progress))
    progress_subscription = fluent.subscribe_progress(progress_changed_callback)

    fluent.prepare_method(methodname)
    wait_until(fluent, "RunModePreparingRun")

    fluent.run_method()
    wait_until(fluent, "RunModeRunning")
    wait_until(fluent, "RunModeRunFinished", "EditMode")

    state_subscription.cancel()
    progress_subscription.cancel()

def mlflow_tracking(experiment_name='experiment1'):
    def _mlflow_tracking(func):
        import mlflow

        def wrapper(**params):
            run_name = func.__name__
            logger.info(f'Start experiment [{run_name}]')
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name) as _:
                mlflow.log_dict(params, 'params.yml')
                res, opts = func(**params)
                mlflow.log_dict(opts, 'opts.yml')
                mlflow.log_dict({'results': res}, 'res.yml')
            logger.info(f'Finish experiment [{run_name}]')
            return (res, opts)
        return wrapper
    return _mlflow_tracking

def loadasc(ascfile, delim="\t"):
    if isinstance(ascfile, str):
        ascfile = pathlib.Path(ascfile)
    else:
        assert isinstance(ascfile, pathlib.Path)
    if not ascfile.is_file():
        return None
    data = []
    with ascfile.open() as f:
        header = f.readline().rstrip()
        for line in f:
            data.append(line.rstrip())
        footer = data.pop()
    data = numpy.asarray([[float(x) for x in line.split(delim)] for line in data])
    return data.T, (header, footer)

@mlflow_tracking()
def read_absorbance_3colors():
    call_method(fluent, "ReadAbsorbance")

    ascpath = pathlib.Path(r"C:\Users\Public\Documents\Tecan\Magellan Pro\asc")
    ascfiles = ascpath.glob("ThreeColorsAbsorbance*.asc")
    ascfile = max(ascfiles, key=lambda x: x.stat().st_ctime)

    logger.info(f"Loading ASCII file [{str(ascfile)}]...")
    assert ascfile.is_file()
    data, (header, footer) = loadasc(ascfile)
    return (data, ), {'header': header, 'footer': footer, 'filename': str(ascfile)}

@mlflow_tracking()
def dispense_liquid_96wells(*, data, channel=0, dropchip=1):
    filename = r"C:\Users\kaizu\Desktop\VARS.csv"

    logger.info(f"Writing file [{filename}]...")
    with open(filename, 'w') as f:
        now = datetime.datetime.now()
        f.write(now.strftime("%Y/%m/%d %H:%M:%S.%f\n"))
        f.write(f'{channel:d}\n')
        f.write(f'{dropchip:d}\n')
        for x in data:
            f.write(f"{x:.0f}\n")
    # with open(filename) as f:
    #     for i, line in enumerate(f):
    #         logger.info(f"LINE{i+1:03d}: {line.rstrip()}")

    call_method(fluent, "DispenseLiquid96Wells")
    return None, {}

def loadattr(attrfile):
    if isinstance(attrfile, str):
        attrfile = pathlib.Path(attrfile)
    else:
        assert isinstance(attrfile, pathlib.Path)
    if not attrfile.is_file():
        return None
    data = []
    with attrfile.open() as f:
        header = f.readline().rstrip()
        for line in f:
            line = line.rstrip()
            if line == '':
                break
            row = line.split(';')
            assert len(row) == 2, str(row)
            data.append(float(row[1]))
    data = numpy.asarray(data)
    return data, (header, )

@mlflow_tracking()
def measure_volume_96wells():
    call_method(fluent, "MeasureVolume96Wells")

    attrfile = pathlib.Path(r"C:\Users\kaizu\Desktop\WellVolume.csv")

    logger.info(f"Loading attribute file [{str(attrfile)}]...")
    assert attrfile.is_file()
    data, (header, ) = loadattr(attrfile)
    return (data, ), {'header': header, 'filename': str(attrfile)}

from tecan import Fluent
# fluent = Fluent("127.0.0.1", 50052)
fluent = Fluent.discover(10)
fluent.start_fluent() 
fluent.close_method()
wait_until(fluent, "RunModeRunFinished", "EditMode")


if __name__ == "__main__":
    # r = numpy.tile(numpy.arange(0, 80, 10), 8)
    # b = numpy.repeat(numpy.arange(0, 80, 10), 8)

    # r = numpy.arange(0, 160, 10)
    r = numpy.zeros(96)
    r[0: : 8] = numpy.arange(0, 120, 10)
    # b = numpy.zeros(len(r))
    # w = 160 - (r + b)

    # data = numpy.zeros(96, numpy.int64)
    # data[: len(r)] = r
    # params = {'data': data, 'channel': 0}
    # print(data[: len(r)])
    # _, opts = dispense_liquid_96wells(**params)

    # data = numpy.zeros(96, numpy.int64)
    # data[: len(b)] = b
    # params = {'data': data, 'channel': 2}
    # print(data[: len(b)])
    # _, opts = dispense_liquid_96wells(**params)

    # data = numpy.zeros(96, numpy.int64)
    # data[: len(w)] = w
    # params = {'data': data, 'channel': 3}
    # print(data[: len(w)])
    # _, opts = dispense_liquid_96wells(**params)

    # params = {}
    # (data, ), opts = measure_volume_96wells(**params)
    # print(data[: len(w)])

    params = {}
    (data, ), opts = read_absorbance_3colors(**params)
    print(data[:, : len(r)])