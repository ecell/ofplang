#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy

from ofplang.definitions import Definitions
from ofplang.protocol import Protocol
from ofplang.validate import check_definitions, check_protocol
from ofplang.runner import Runner
from ofplang.executors import Simulator

definitions = Definitions('./manipulate.yaml')
check_definitions(definitions)
protocol = Protocol("./sample1.yaml")
check_protocol(protocol, definitions)
runner = Runner(protocol, definitions, executor=Simulator())

import optuna

def objective(trial):
    assert 0 <= trial.number < 96
    x = trial.suggest_float('x', 0, 200)
    volume = numpy.zeros(96, dtype=float)
    volume[trial.number] = x
    inputs = {"volume": {"value": volume, "type": "Array[Float]"}}
    outputs = runner.run(inputs).output
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