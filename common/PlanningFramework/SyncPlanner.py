from lib.IBehaviourProvider import IBehaviourProvider
from lib.IVehicle import IVehicle
from lib.Behaviour import Behaviour
from typing import List
from common.PlanningFramework.DriverModelMapper import DriverModelMapper
from collections import deque
from common.StateLogger import StateLogger
import common.user_settings as cfg


class SyncPlanner(IBehaviourProvider):
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    Pops behavior data from behavior row and starts new planning process if behavior queue is empty.
    """
    LOGGING = True

    def __init__(self):
        self.driver_model_mapper = DriverModelMapper()
        self.behaviour_data = deque()

        self.planning_time = 0.1

        self.simulation_time_step = None

    def set_up(self, simulation_data_provider):
        self.driver_model_mapper.set_mapping_from_config(simulation_data_provider)
        self.simulation_time_step = round(simulation_data_provider.get_time_step_size(), 2)

    def get_behaviours(self, queue_reference, loop_idx, vehicles: List[IVehicle]) -> List[Behaviour]:
        """
        Method gets called by calculation_loop of file SimulationCore.py.
        Splits list of vehicles according to the driverModelIds and
        calls the getAssignedVehicleBehaviour methods of the DriverModel modules
        with appropriate sublists.
        Returns the then merged list of tuples (Car, Behaviour) of all vehicles.
        """

        # Look for available data first
        try:
            behaviour_row = self.behaviour_data.popleft()
        except IndexError:  # no data available
            # run driver models synchronously
            queue_reference, loop_idx = self.driver_model_mapper.dispatch_tasks(self.behaviour_data, vehicles,
                                                                                queue_reference, loop_idx,
                                                                                concurrent=False,
                                                                                total_time=self.planning_time)

            # Retry getting data
            behaviour_row = self.behaviour_data.popleft()

        # Sort behaviours by vehicle ids to ensure match with vehicle list
        # Convert to normal list type
        behaviour_row = list(behaviour_row)
        behaviour_row.sort(key=lambda x: x.vehicle_id)
        return behaviour_row, queue_reference, loop_idx
