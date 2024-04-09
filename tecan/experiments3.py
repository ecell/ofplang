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

def experiment1():
    call_method(fluent, "MeasureVolume96Wells")

    attrfile = pathlib.Path(r"C:\Users\Tecan\Desktop\WellVolume.csv")

    logger.info(f"Loading attribute file [{str(attrfile)}]...")
    assert attrfile.is_file()
    data, (header, ) = loadattr(attrfile)
    return data, {'header': header, 'filename': str(attrfile)}

from tecan import Fluent
fluent = Fluent.discover(10)
fluent.start_fluent() 
fluent.close_method()
wait_until(fluent, "RunModeRunFinished", "EditMode")


if __name__ == "__main__":
    from mlflow import set_experiment, start_run, log_metric, log_param, log_artifact, log_dict

    params = {}
    data, opts = experiment1(**params)
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
