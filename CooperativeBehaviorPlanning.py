from common.ServiceManager import ServiceManager
import common.user_settings as cfg
from common.Visualize_Scenario import plot_scenario_from_file
import os
import time


class CooperativeBehaviorPlanning:
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    Set up cooperative behavior planning process.
    """

    def __init__(self):
        self.service_manager = None
        self.simulation_service = None

    @staticmethod
    def main(scene=None, params=None):
        """
        Main function to setup and run planning process.

        :param scene: Parameter for controlling simulation externally. If not specified (None), parameters will be
                        read from Cars.txt and Roadway.txt defined in "user_settings.py". Scene is a tuple of
                        (roadway_list, cars_list). Both lists consist of one string for each line in
                        Cars.txt/Roadway.txt.
        :param params: Dictionary to control simulation parameters externally. If not specified, standard parameters
                        will be used. Params is a dictionary of parameters {"parameter_name": parameter_value, "..."}.
        :return: Pandas dataframe with all result variables over time.
        """
        controller = CooperativeBehaviorPlanning()
        controller.initialize(scene, params)
        sim_result = controller.start()
        return sim_result

    def initialize(self, scene, params):
        """
        Initializes modules and configuration parameters.

        :param scene: Parameter for controlling simulation externally. If not specified (None), parameters will be
                        read from Cars.txt and Roadway.txt defined in "user_settings.py". Scene is a tuple of
                        (roadway_list, cars_list). Both lists consist of one string for each line in
                        Cars.txt/Roadway.txt.
        :param params: Dictionary to control simulation parameters externally. If not specified, standard parameters
                        will be used. Params is a dictionary of parameters {"parameter_name": parameter_value, "..."}.
        """
        # Create ServiceManager
        self.service_manager = ServiceManager(scene, params)
        # Configure main modules
        self.simulation_service = self.service_manager.get_simulation_service()
        behaviour_provider = self.service_manager.get_behaviour_provider()
        self.simulation_service.setUp(behaviour_provider=behaviour_provider)
        behaviour_provider.set_up(self.simulation_service.get_simulation_data_provider())

    def start(self):
        """
        Start simulation.
        :return: Pandas dataframe with all result variables over time.
        """
        sim_result = self.simulation_service.run(cfg.sim_time)
        return sim_result


if __name__ == '__main__':
    # take time
    start_time = time.time()
    # plan scenario
    result = CooperativeBehaviorPlanning.main()
    # print calculation time
    print("Calculation time: " + str(time.time() - start_time))
    # plot results
    path_results = os.path.dirname(os.path.realpath(__file__)) + '/SimulationEnvironment/memory/' + cfg.scene + '.csv'
    path_roadway = os.path.dirname(
        os.path.realpath(__file__)) + '/SimulationEnvironment/init_data/Scenes/' + cfg.scene + '/Roadway.txt'
    plot_scenario_from_file(path_results, path_roadway)
