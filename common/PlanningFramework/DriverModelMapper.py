import os
import numpy as np
from typing import List, Tuple
from lib.ISimulationDataProvider import ISimulationDataProvider
from lib.IVehicle import IVehicle
from common.utilities import dynamic_import
import json
from collections import deque
import queue
import threading


class DriverModelMapper:
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    DriverModelMapper class maps from a given integer index to a concrete
    implementation of the IDriverModel interface.
    The index to implementation mapping is specified in the Config.ini file and
    can be used to assign a specific algorithm to Car objects by setting the attribute
    driverModelId in the Car class.
    The mapping is implemented by the dictionary attribute driverModelMapping
    with the indexes as key values and the concrete implementation instances as values.
    """

    def __init__(self):

        self._driver_model_mapping = dict()
        self.queue_handler = None
        self.simulation_data_provider = None

    def dispatch_tasks(self, behaviour_data, vehicles, queue_reference, loop_idx, concurrent: bool, total_time,
                       time_out=None):
        """
        Dispatches incoming planning requests to configured driver models.
        Uses the concurrent flag to determine if new processes should be used for driver models.
        If concurrent is set to False, plan_execution may be halted in driver model default.
        """
        if concurrent is False:
            queue_reference, loop_idx = self.start_driver_models(behaviour_data, vehicles, total_time,
                                                                 self.simulation_data_provider.get_time_step_size(),
                                                                 queue_reference, loop_idx)
            return queue_reference, loop_idx
        else:
            # Use new thread to start driver models
            t = threading.Thread(target=self.start_driver_models_async,
                                 args=(behaviour_data, vehicles, total_time,
                                       self.simulation_data_provider.get_time_step_size(), time_out))
            t.start()

    def start_driver_models(self, behaviour_data, vehicles, total_time, sim_time_step, queue_reference, loop_idx):
        """
        Starts and executes driver models synchronously. Control flow may be halted in
        driver model plan_execution.
        """

        # Start driver models synchronously
        data = []
        for index in self._driver_model_mapping:
            assignedVehicles, indexesInVehicles = self.get_assigned_vehicles(vehicles, driverModelId=index)
            # Call driverModel to get assigned vehicles behaviours
            if assignedVehicles:
                # Create data store
                q = queue.Queue()
                if not queue_reference:
                    queue_reference = []
                # Run driver model to completion
                queue_reference, loop_idx = self._driver_model_mapping[index].run(q, assignedVehicles, vehicles,
                                                                                  total_time, queue_reference, loop_idx)

                # Extract behaviour data from queue
                columns = []
                while True:
                    row = q.get()
                    if row is None:
                        break
                    else:
                        columns.append(row)
                data.append(np.array(columns))

        # Concatenate data and add to deque
        behaviour_data.extend(deque(np.concatenate([block for block in data], axis=1)))
        return queue_reference, loop_idx

    def get_mapping(self):
        return self._driver_model_mapping

    def set_mapping_from_config(self, simulationDataProvider: ISimulationDataProvider):
        """
        Loads Driver Model configuration from config file and stores
        the mapping from indexes to instances in a dictionary.
        """
        self.simulation_data_provider = simulationDataProvider
        path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/Config.json"

        with open(path, 'r') as fp:
            config = json.load(fp)
            driver_models = config['driver_models']
            for el in driver_models:
                # Get driver model information from json
                class_name = el['name']
                rel_path = el['rel_path']
                cl = dynamic_import(rel_path, class_name)
                id = el['id']
                # Create instance and add to mapping
                obj = cl.create(simulationDataProvider)
                self._driver_model_mapping[id] = obj

    def get_assigned_vehicles(self, vehicles: List[IVehicle], driverModelId: int) -> Tuple[List[IVehicle], List[int]]:
        """
        Takes a list of car objects as input and extract a sublist containing
        all objects which driverModelIds match the input argument driverModelId.
        Returns the resulting sublist and a list containing the positions of
        extracted objects in the original list.
        """
        assignedVehicles = list()
        positions = list()
        for position, vehicle in enumerate(vehicles):
            if int(vehicle.get_driver_model()) == int(driverModelId):
                assignedVehicles.append(vehicle)
                positions.append(position)
        return assignedVehicles, positions
