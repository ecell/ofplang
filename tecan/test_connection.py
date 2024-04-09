import time

def wait_until(fluent, *status, interval=3):
    #RunModeRunFinished
    #RunModePreRunChecks
    #RunModePreparingRun
    #RunModeWaitingForSystem
    #RunModeBeginRun
    #RunModeRunning
    #EditMode
    while fluent.state not in status:
        print(f"Waiting... [{fluent.state}]")
        time.sleep(interval)

from tecan import Fluent
# fluent = Fluent("127.0.0.1", 5353)
fluent = Fluent.discover(10)
fluent.start_fluent() 
fluent.close_method()
wait_until(fluent, "RunModeRunFinished", "EditMode")
print("Connected")