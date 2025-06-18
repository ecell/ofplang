#!/usr/bin/python
# -*- coding: utf-8 -*-

from logging import getLogger
import pathlib
import time
import datetime
import numpy
from .__Fluent import Fluent
# from tecan import Fluent  # type: ignore[import-untyped]

logger = getLogger(__name__)


MAGELLAN_PATH = r"C:\Users\Public\Documents\Tecan\Magellan Pro\asc"
# FLUENT_IO_PATH = r"C:\Users\kaizu\Desktop"
FLUENT_IO_PATH = r"C:\Users\Tecan\Desktop"

def wait_until(fluent, *status, interval=3, max_iter=0):
    #RunModeRunFinished
    #RunModePreRunChecks
    #RunModePreparingRun
    #RunModeWaitingForSystem
    #RunModeBeginRun
    #RunModeRunning
    #EditMode
    #RunModeStopOnError  #TODO
    cnt = 0
    while fluent.state not in status:
        logger.debug(f"Waiting... [{fluent.state}]")
        time.sleep(interval)
        cnt += 1

        if cnt > max_iter > 0:
            raise RuntimeError(f"Connection failed [{', '.join(status)}]")

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

# def mlflow_tracking(experiment_name='experiment1'):
#     def _mlflow_tracking(func):
#         import mlflow

#         def wrapper(**params):
#             run_name = func.__name__
#             logger.info(f'Start experiment [{run_name}]')
#             mlflow.set_experiment(experiment_name)
#             with mlflow.start_run(run_name=run_name) as _:
#                 mlflow.log_dict(params, 'params.yml')
#                 res, opts = func(**params)
#                 mlflow.log_dict(opts, 'opts.yml')
#                 mlflow.log_dict({'results': res}, 'res.yml')
#             logger.info(f'Finish experiment [{run_name}]')
#             return (res, opts)
#         return wrapper
#     return _mlflow_tracking

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

def read_absorbance_3colors(fluent):
    call_method(fluent, "ReadAbsorbance")

    ascpath = pathlib.Path(MAGELLAN_PATH)
    ascfiles = ascpath.glob("ThreeColorsAbsorbance*.asc")
    ascfile = max(ascfiles, key=lambda x: x.stat().st_ctime)

    logger.info(f"Loading ASCII file [{str(ascfile)}]...")
    assert ascfile.is_file()
    data, (header, footer) = loadasc(ascfile)
    return (data, ), {'header': header, 'footer': footer, 'filename': str(ascfile)}

def dispense_liquid_96wells(fluent, *, data, channel=0, dropchip=1):
    filename = f"{FLUENT_IO_PATH}\\VARS.csv"

    channelmap = {'A1': 0, 'B1': 1, 'A2': 2, 'B2': 3, 'A3': 4, 'B3': 5}
    if isinstance(channel, str):
        channel = channelmap[channel]
    assert isinstance(channel, int) and 0 <= channel and channel < 6

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

def measure_volume_96wells(fluent):
    call_method(fluent, "MeasureVolume96Wells")

    filename = f"{FLUENT_IO_PATH}\\WellVolume.csv"
    attrfile = pathlib.Path(filename)

    logger.info(f"Loading attribute file [{str(attrfile)}]...")
    assert attrfile.is_file()
    data, (header, ) = loadattr(attrfile)
    return (data, ), {'header': header, 'filename': str(attrfile)}

def setup() -> Fluent | None:
    # fluent = Fluent("127.0.0.1", 50052)
    # fluent = Fluent("10.5.1.22", 50052)
    fluent = Fluent.discover(15)
    fluent.start_fluent()
    fluent.close_method()
    wait_until(fluent, "RunModeRunFinished", "EditMode", max_iter=100)
    return fluent
