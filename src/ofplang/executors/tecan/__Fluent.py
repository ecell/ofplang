#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This file is derived from [Python/src/tecan/__Fluent.py], originally part of the [fluent-sila2-connector] project,
and is redistributed here under the terms of the BSD License.

Original source: https://gitlab.com/tecan/fluent-sila2-connector (commit 72296334b6550a6956e2e1a467958ac2bf20517c)

Copyright (c) 2019, Tecan Trading AG

Modifications may have been made from the original version.


Wrapper for the SiLAFluentController.
"""

from logging import getLogger
from enum import Enum

from sila2.client import SilaClient
from sila2.discovery import SilaDiscoveryBrowser
import threading

logger = getLogger(__name__)


class Fluent:
    """
    Python wrapper to remote Fluent.
    """

    def __init__(self, server_ip: str = "127.0.0.1", server_port: int = 50052, client: SilaClient = None, **kwargs):
        """Connects to a SilaFluentServer an creates a Fluent object able to control an instance of FluentControl.
        :param server_ip:
        :type server_ip: str
        :param server_port:
        :type server_port: int
        """
        if client is not None:
            self.__client = client
        else:
            self.__client = SilaClient(server_ip, server_port, **kwargs)
        if "SilaFluentController" not in dir(self.__client):
            raise AttributeError("The connected server is not a FluentControl SiLA2 Server")
        self.variables = _VariableContainer(self)
        logger.info("successfully connected to the server")

    def add_labware(
        self,
        labware_name: str,
        labware_type: str,
        target_location: str,
        position=0,
        rotation=0,
        has_lid=False,
        barcode="",
    ):
        """adds labware to the worktable
        :param barcode:
        :type barcode: str
        :param labware_name:
        :type labware_name: str
        :param labware_type:
        :type labware_type: str
        :param target_location: labware-name of the target
        :type target_location: str
        :param position: defaults to 0
        :type position: int, optional
        :param rotation: defaults to 0
        :type rotation: int, optional
        :param has_lid: defaults to False
        :type has_lid: bool, optional
        """
        parameters = (barcode, has_lid, labware_name, labware_type, target_location, position, rotation)
        self.__client.SilaFluentController.AddLabware(parameters)
        logger.info("Labware added successfully")

    def remove_labware(self, labware_name: str):
        """removes labware from worktable
        :param labware_name: name of labware to remove
        :type labware_name: str
        """
        self.__client.SilaFluentController.RemoveLabware(labware_name)
        logger.info("Labware removed successfully")

    def set_location(self, labware: str, rotation: int, target_location: str, target_site: int):
        """
        :param labware: name of the labware
        :type labware: str
        :param rotation:
        :type rotation: int
        :param target_location:
        :type target_location: str
        :param target_site:
        :type target_site: int
        """
        parameters = (labware, rotation, target_location, target_site)
        self.__client.SilaFluentController.SetLocation(parameters)
        logger.info("Location successfully set")

    def transfer_labware(
        self, labware_to_location: str, target_location: str, target_position=0, only_use_selected_site=True
    ):
        """transfers labware on the worktable.
        :param labware_to_location:
        :type labware_to_location: str
        :param target_location:
        :type target_location: str
        :param target_position: defaults to 0
        :type target_position: int, optional
        :param only_use_selected_site: , defaults to True
        :type only_use_selected_site: bool, optional
        """
        parameters = (labware_to_location, only_use_selected_site, target_location, target_position)
        self.__client.SilaFluentController.TransferLabware(parameters)
        logger.info("labware successfully transferred")

    def transfer_labware_back_to_base(self, labware_name: str):
        """transfers labware back to the spot where it was added to the worktable.
        :param labware_name:
        :type labware_name: str
        """
        self.__client.SilaFluentController.TransferLabwareBackToBase(labware_name)
        logger.info("labware successfully transferred")

    def get_fingers(self, device_alias: str, gripper_fingers: str):
        """
        :param device_alias:
        :type device_alias: str
        :param gripper_fingers:
        :type gripper_fingers: str
        """
        parameters = (device_alias, gripper_fingers)
        self.__client.SilaFluentController.GetFingers(parameters)
        logger.info("successfully mounted fingers")

    def user_prompt(self, text: str):
        """
        :param text:
        :type text: str
        """
        self.__client.SilaFluentController.UserPrompt(text)

    def get_tips(self, airgap_volume: int, airgap_speed: int, diti_type):
        """
        :param airgap_volume:
        :type airgap_volume: int
        :param airgap_speed:
        :type airgap_speed: int
        :param diti_type:
        :type diti_type: DiTi
        """
        parameters = (airgap_volume, airgap_speed, diti_type.value)
        self.__client.SilaFluentController.GetTips(parameters)
        logger.info("done")

    def aspirate(self, volume: int, labware: str, liquid_class: str, well_offset=0):
        """
        :param volume:
        :type volume: int
        :param labware:
        :type labware: str
        :param liquid_class:
        :type liquid_class: str
        :param well_offset: , defaults to 0
        :type well_offset: int, optional
        """
        parameters = (volume, labware, liquid_class, well_offset)
        logger.info("aspirating...")
        self.__client.SilaFluentController.Aspirate(parameters)
        logger.info("finished aspirating")

    def dispense(self, volume: int, labware: str, liquid_class: str, well_offset=0):
        """
        :param volume:
        :type volume: int
        :param labware:
        :type labware: str
        :param liquid_class:
        :type liquid_class: str
        :param well_offset: , defaults to 0
        :type well_offset: int, optional
        """
        parameters = (volume, labware, liquid_class, well_offset)
        self.__client.SilaFluentController.Dispense(parameters)
        logger.info("finished dispensing")

    def drop_tips(self, labware: str):
        """
        :param labware:
        :type labware: str
        """
        self.__client.SilaFluentController.DropTips(labware)
        logger.info("dropped")

    def prepare_method(self, to_prepare: str):
        """Prepare a method so that you can run it later.
        :param to_prepare:
        :type to_prepare: str
        """
        logger.info("preparing method...")
        self.__client.SilaFluentController.PrepareMethod(to_prepare)
        logger.info("method ready to run")

    def run_method(self):
        """Runs a method. you have to prepare it first by prepare_method()."""
        self.__client.SilaFluentController.RunMethod()
        """
        variables = self.get_variable_names()
        for name in variables:
            globals()[name].__setattr__(name, self.get_variable_value(name))
        """
        logger.info("method running")

    def pause_run(self):
        """Pauses a method run."""
        self.__client.SilaFluentController.PauseRun()

    def resume_run(self):
        """Resumes a method run that was paused before."""
        self.__client.SilaFluentController.ResumeRun()
        logger.info("method paused")

    def stop_method(self):
        """Stops a method run. For a soft stop better use finish_execution()."""
        self.__client.SilaFluentController.StopMethod()
        logger.info("method stopped")

    def close_method(self):
        """closes a method."""
        self.__client.SilaFluentController.CloseMethod()
        logger.info("method closed")

    def set_variable_value(self, variable_name: str, value: str):
        """
        :param variable_name:
        :type variable_name: str
        :param value:
        :type value: str
        """
        parameters = (variable_name, str(value))
        self.__client.SilaFluentController.SetVariableValue(parameters)
        logger.info("Value successfully set")

    def get_variable_names(self) -> list:
        """
        only works if you have a method prepared but not yet running
        :return:
        :rtype: str
        """
        return self.__client.SilaFluentController.GetVariableNames().ReturnValue

    def get_variable_value(self, variable_name: str) -> str:
        """
        :param variable_name:
        :type variable_name: str
        :return:
        :rtype: str
        """
        return self.__client.SilaFluentController.GetVariableValue(variable_name).ReturnValue

    def get_all_runnable_methods(self):
        """returns a list of all methods that are executable from python. can only show methods, that are visible in touchtools.
        :return:
        :rtype: list
        """
        return self.__client.SilaFluentController.GetAllRunnableMethods().ReturnValue

    def finish_execution(self):
        """Stops a method run softly. Best practice to end a method run containing an API-channel."""
        self.__client.SilaFluentController.FinishExecution()
        logger.info("successfully finished execution.")

    def start_fluent(self, username: str = None, password: str = None, simulation_mode=False):
        """Starts Fluent Controll or attaches to a running instance.
        :param username: Your Fluent USM username. When set, password has to be set as well., defaults to None
        :type username: str, optional
        :param password: Set only, when Username is set, defaults to None
        :type password: str, optional
        :param simulation_mode: Starts Fluent in simulation mode when set True, defaults to False
        :type simulation_mode: bool, optional
        :raises AttributeError: Raised if only one of username or password was set.
        """
        if username is None:
            if password is not None:
                raise AttributeError("username missing")
            logger.info("starting Fluent...")
            if simulation_mode:
                self.__client.SilaFluentController.StartFluentInSimulationMode()
            else:
                self.__client.SilaFluentController.StartFluentOrAttach()
            logger.info("started")
        elif password is None:
            raise AttributeError("password missing")
        else:
            parameters = (username, password)
            logger.info("starting Fluent...")
            self.__client.SilaFluentController.StartFluentAndLogin(parameters)
            logger.info("started!")

    def shutdown(self, timeout: int):
        """Shuts down a running FluentControl instance.

        :param timeout: [description]
        :type timeout: int
        """
        self.__client.SilaFluentController.Shutdown(timeout)

    def __progress(self):
        return self.__client.SilaFluentStatusProvider.Progress.get()

    def __state(self):
        return self.__client.SilaFluentStatusProvider.State.get()

    def __lastError(self):
        return self.__client.SilaFluentStatusProvider.LastError.get()

    def subscribe_progress(self, progress_callback):
        """Subscribes to the current progress with the given callback"""
        subscription = self.__client.SilaFluentStatusProvider.Progress.subscribe()
        return _CallbackSubscription(subscription, progress_callback)

    def subscribe_state(self, state_callback):
        """Subscribes to the state with the given callback"""
        subscription = self.__client.SilaFluentStatusProvider.State.subscribe()
        return _CallbackSubscription(subscription, state_callback)

    def subscribe_error(self, error_callback):
        """Subscribes to the error messages with the given callback"""
        subscription = self.__client.SilaFluentStatusProvider.LastError.subscribe()
        return _CallbackSubscription(subscription, error_callback)

    progress = property(__progress, doc="Gets the value of the last progress update")
    state = property(__state, doc="Gets the state of FluentControl")
    lastError = property(__lastError, doc="Gets the last error message")

    @staticmethod
    def discover(timeout: float = 0):
        """Uses zeroconf to find a server that is running in the local network"""
        browser = SilaDiscoveryBrowser()
        client = browser.find_server(timeout=timeout)
        if client is not None:
            return Fluent(client=client)

class _CallbackSubscription:
    def __init__(self, subscription, callback) -> None:
        self.__subscription = subscription
        self.__callback = callback
        thread = threading.Thread(target=self.__run)
        thread.start()
        self.__thread = thread

    def __run(self):
        for item in self.__subscription:
            if item is None or item is "":
                continue
            try:
                self.__callback(item)
            except Exception as e:
                logger.error("Subscription callback raised an error", e)

    def cancel(self):
        self.__subscription.cancel()
        self.__thread.join()

class _VariableContainer:
    """Encapsulates FluentControl variables"""

    variables = {}

    def __init__(self, client: Fluent):
        """Creates an encapsulation of the variables in FluentControl"""
        self.__client = client

    def __getattr__(self, name):
        if name == "_VariableContainer__client":
            return super().__getattr__(name)
        return self.__client.get_variable_value(name)

    def __setattr__(self, name, value=None):
        if name == "_VariableContainer__client":
            super().__setattr__(name, value)
        elif value is not None:
            self.__client.set_variable_value(name, value)
        else:
            raise AttributeError("value must not be None")

    def __dir__(self):
        return self.__client.get_variable_names()


class DiTi(Enum):
    """
    Enum for DiTi types
    """

    FCA_200_UL_FILTERED_SBS = "TOOLTYPE:LiHa.TecanDiTi/TOOLNAME:FCA, 200ul Filtered SBS"
    FCA_5000_UL_FLIERED_SBS = "TOOLTYPE:LiHa.TecanDiTi/TOOLNAME:FCA, 5000ul Filtered SBS"
