from lib.Direction import Direction


class Behaviour:
    """
    Created by:
    Marius Gruber

    Description:
    Define behavior data structure (vehicle id, acceleration, lane change, direction of lane change).
    """

    def __init__(self, vehicle_id: int, acc: float, lane_change: bool, direction: Direction):
        self.vehicle_id = vehicle_id
        self.acc = acc
        self.lane_change = lane_change
        self.direction = direction
