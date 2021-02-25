import abc
from typing import List


class IRoadwayModel:
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_right_most_lane(self, pos: float) -> int:
        """
        Created by:
        Marius Gruber

        Description: Get rightmost lane of roadway
        :param pos: absolute position on roadway
        :return: lane number of right most lane
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_left_most_lane(self, pos: float) -> int:
        """
        Created by:
        Marius Gruber

        Description:  Get leftmost lane of roadway
        :param pos: absolute position on roadway
        :return: lane number of left most lane
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_lanes(self, pos: float) -> List[int]:
        """
        Created by:
        Marius Gruber

        Description: Get number of lanes of roadway
        :param pos: absolute position on roadway
        :return: list containing all lane-IDs available at given position
        """
        raise NotImplementedError

