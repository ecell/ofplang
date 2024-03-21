#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from definitions import Definitions
from protocol import Protocol
from validate import check_definitions, check_protocol
from runner import Runner
from simulator import Simulator

definitions = Definitions('./manipulate.yaml')
check_definitions(definitions)
protocol = Protocol("./sample.yaml")
check_protocol(protocol, definitions)
runner = Runner(protocol, definitions)
executor = Simulator()
runner.executor = executor

import optuna

def objective(trial):
    assert 0 <= trial.number < 96
    x = trial.suggest_float('x', 0, 200)
    volume = numpy.zeros(96, dtype=float)
    volume[trial.number] = x
    inputs = {"volume": {"value": volume, "type": "Array[Float]"}}
    outputs = runner.run(inputs)
    y = outputs["data"]["value"][0][trial.number]
    return (y - 0.5) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=96)

print(study.best_params)  # E.g. {'x': 99.94241349236856}

import matplotlib.pyplot as plt
ax = optuna.visualization.matplotlib.plot_optimization_history(study)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('optimization_history.png', dpi=300)