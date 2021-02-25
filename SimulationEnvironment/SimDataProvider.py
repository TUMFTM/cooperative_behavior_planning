from lib.ISimulationDataProvider import ISimulationDataProvider
from SimulationEnvironment.Setting import GD
from .GDRoadwayModel import GDRoadwayModel


class SimDataProvider(ISimulationDataProvider):
    """
    Created by:
    Marius Gruber

    Description:
    Provides access to simulation data.
    """

    def __init__(self, globalData: GD):
        self.globalData = globalData

    def get_roadway_model(self):
        roadway_model = GDRoadwayModel(lanes=self.globalData.roadway.lane_matrix,
                                       entrances=self.globalData.roadway.list_entrances,
                                       exits=self.globalData.roadway.list_exits)
        return roadway_model

    def get_time_step_size(self):
        return self.globalData.time_per_step
