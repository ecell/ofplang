#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from ofplang.prelude import *

from ofplang.executors import Simulator

runner = Runner("./protocol1.yaml", "./definitions.yaml", executor=Simulator())

import optuna

def objective(trial):
    assert 0 <= trial.number < 96
    x = trial.suggest_float('x', 0, 200)
    y = trial.suggest_float('y', 0, 200)
    volume1 = numpy.zeros(96, dtype=float)
    volume1[trial.number] = x
    volume2 = numpy.zeros(96, dtype=float)
    volume2[trial.number] = y
    inputs = {"volume1": {"value": volume1, "type": "Array[Float]"}, "volume2": {"value": volume2, "type": "Array[Float]"}}
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
