from tecan import Fluent

import time
import logging
logging.basicConfig(level=logging.INFO)

# f = Fluent('127.0.0.1', 50052)
fluent = Fluent.discover(10)

def state_changed_callback(state):
    # str
    logging.info("New state is " + str(state))
state_subscription = fluent.subscribe_state(state_changed_callback)

def progress_changed_callback(progress):
    # int
    logging.info("New progress is " + str(progress))
progress_subscription = fluent.subscribe_progress(progress_changed_callback)

fluent.start_fluent() 
# print(dir(fluent))

print(fluent.get_all_runnable_methods()) 
fluent.close_method()

while fluent.state not in ("RunModeRunFinished", "EditMode"):
    time.sleep(3)

fluent.prepare_method("TestAPIMethod1")
# fluent.prepare_method("ReadAbsorbance")

while fluent.state != "RunModePreparingRun":
    time.sleep(3)

variable_names = fluent.get_variable_names()
print(variable_names)

fluent.run_method()

while fluent.state != "RunModeRunning":
    time.sleep(3)

while fluent.progress < 1:
    time.sleep(3)

fluent.transfer_labware("96 Well Flat FALCON[002]", "Nest7mm_Pos_3", 7) 
logging.info("transfer_labware")

print(fluent.get_variable_names())
print(fluent.variables)
print(fluent.variables.var1)
fluent.variables.var1 = "new value"
print(fluent.variables.var1)

# time.sleep(20)

fluent.finish_execution() 

while fluent.state not in ("RunModeRunFinished", "EditMode"):
    time.sleep(3)

state_subscription.cancel()
progress_subscription.cancel()

#RunModeRunFinished
#RunModePreRunChecks
#RunModePreparingRun
#RunModeWaitingForSystem
#RunModeBeginRun
#RunModeRunning
#EditMode