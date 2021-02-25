from SimulationEnvironment.SimulationEnvironment import SimulationEnvironment
from pathlib import Path
from common.PlanningFramework.SyncPlanner import SyncPlanner
import common.user_settings as cfg
from SimulationEnvironment.Setting import GD


class ServiceManager:
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    Factory class which implements the singleton pattern for services.
    """

    def __init__(self, scene, params):

        self._behaviour_provider = None
        self._simulationService = None
        self.conf_path = Path(__file__).parent.joinpath('Config.json')
        self.scene = scene
        GD.ext_params = params

    def get_behaviour_provider(self):
        if not self._behaviour_provider:
            if cfg.execution_mode == 'sync':
                self._behaviour_provider = SyncPlanner()
            else:
                raise ValueError('Invalid execution mode in user_settings.py')
        return self._behaviour_provider

    def get_simulation_service(self):
        GD.all_cars_object_list = []
        GD.all_cars_data_array = []
        GD.number_of_cars = 0
        if not self._simulationService:
            self._simulationService = SimulationEnvironment(self.scene)
        return self._simulationService
