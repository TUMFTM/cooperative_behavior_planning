import abc
from typing import List
from lib.IVehicle import IVehicle
from lib.Behaviour import Behaviour


class IBehaviourProvider:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_behaviours(self, queue_reference, loop_idx, vehicles: List[IVehicle]) -> List[Behaviour]:
        """
        Created by:
        Marius Gruber

        Description:
        This method gets called by the simulation service to obtain behaviour data
        for the current time step and given vehicles.

        :param queue_reference: Reference queue as input for trajectory planning (not implemented).
        :param loop_idx: Loop index of simulation environment.
        :param vehicles: Input IVehicle objects.
        :return:
        """

    @abc.abstractmethod
    def set_up(self, simulation_data_provider):
        """
        Created by:
        Marius Gruber

        Description:
        This method is called at initialization time and should fully configure the implementing service.
        :param simulation_data_provider: Provides access to simulation data.
        """
        raise NotImplementedError
