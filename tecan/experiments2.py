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
    while fluent.state not in status:
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

def experiment1(data, channel=0):
    filename = r"C:\Users\Tecan\Desktop\VARS.csv"

    logger.info(f"Writing file [{filename}]...")
    with open(filename, 'w') as f:
        now = datetime.datetime.now()
        f.write(now.strftime("%Y/%m/%d %H:%M:%S.%f\n"))
        f.write(f'{channel:d}\n')
        for x in data:
            f.write(f"{x:.0f}\n")
    with open(filename) as f:
        for i, line in enumerate(f):
            logger.info(f"LINE{i+1:03d}: {line.rstrip()}")

    call_method(fluent, "DispenseLiquid96Wells")

    # ascpath = pathlib.Path(r"C:\Users\Public\Documents\Tecan\Magellan Pro\asc")
    # ascfiles = ascpath.glob("ThreeColorsAbsorbance*.asc")
    # ascfile = max(ascfiles, key=lambda x: x.stat().st_ctime)

    # logger.info(f"Loading ASCII file [{str(ascfile)}]...")
    # assert ascfile.is_file()
    # data, (header, footer) = loadasc(ascfile)
    return None, {}

from tecan import Fluent
fluent = Fluent.discover(10)
fluent.start_fluent() 
fluent.close_method()
wait_until(fluent, "RunModeRunFinished", "EditMode")


if __name__ == "__main__":
    from mlflow import set_experiment, start_run, log_metric, log_param, log_artifact, log_dict

    data = numpy.zeros(96, numpy.int64)
    data[43] = 100
    data[52] = 100
    data[53] = 50
    params = {'data': data, 'channel': 1}
    _, opts = experiment1(**params)
    # set_experiment("experiment1")

    # params = {}
    # with start_run(run_name='test') as _:
    #     for k, v in params.items():
    #         log_param(k, v)

    #     data, opts = experiment1(**params)
    #     print(opts["filename"])
    #     print(opts["header"])
    #     print(opts["footer"])
    #     print(data)
    #     print(data.shape)

    #     log_dict(opts, 'opts.yml')
    #     for i in range(3):
    #         log_metric(f'mean label{i+1}', numpy.average(data[i]))
    #         log_metric(f'max label{i+1}', numpy.max(data[i]))
    #         log_metric(f'min label{i+1}', numpy.min(data[i]))
    #     log_artifact(opts["filename"])
