import abc
from .IRoadwayModel import IRoadwayModel


class ISimulationDataProvider:
    """
    Created by:
    Marius Gruber

    Description:
    The IDataProviderFactory is an abstract factory providing
    relevant data structures to the DriverModels. @see{IDataProviderFactory}
    It needs to be implemented by the simulation framework and is propagated
    to the DriverModels by the DriverModelMapper.
    """
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_roadway_model(self) -> IRoadwayModel:
        """
        This method returns an instance of a roadway model which
        implements the IRoadwayModel interface.
        """
        pass

    @abc.abstractmethod
    def get_time_step_size(self):
        """
        :return: Simulation time step size in seconds
        """
        pass


