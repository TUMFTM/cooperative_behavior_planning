import abc
from typing import List
from lib.Behaviour import Behaviour
from lib.ISimulationDataProvider import ISimulationDataProvider
from lib.IVehicle import IVehicle


class IDriverModel:
    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractmethod
    def create(cls, dataProviderFactory: ISimulationDataProvider):
        """
        Created by:
        Marius Gruber

        Description:
        Returns an initialized instance of the IDriverModel implementation.
        """
        pass

    @abc.abstractmethod
    def run(self, queue, assigned_vehicles, all_vehicles, total_time, queue_reference, loop_idx, time_out=None):
        raise NotImplementedError
