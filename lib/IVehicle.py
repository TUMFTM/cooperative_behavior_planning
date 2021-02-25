import abc


class IVehicle:
    """
    Created by:
    Marius Gruber

    Description:
    Interface for Vehicle class.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_pos(self) -> float:
        """
        property get
        use global coordinates that match roadway model's coordinates
        :return: vehicle's current position [m]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_vel(self) -> float:
        """
        property get
        :return: vehicle's current velocity [m/s]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_a(self) -> float:
        """
        property get
        :return: vehicle's current acceleration [m/s^2]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_lane(self) -> int:
        """
        property get
        left most lane-ID available is 0
        right most lane-ID available is 10
        :return: lane occupied by vehicle [-]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_destination_lane(self) -> int:
        """
        property get
        left most lane-ID available is 0
        right most lane-ID available is 10
        :return: lane, the vehicle is heading at
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_is_changing_lane(self) -> bool:
        """
        property get
        :return: true, if vehicle is currently changing lane; false if not
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_timeout_for_lane_change(self) -> float:
        """
        property get
        :return: time remaining until lane change is terminated [s]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_driver_model(self) -> int:
        """
        property get
        :return: driver model ID that is ought to plan for this vehicle
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_target_velocity(self) -> float:
        """
        property get
        :return: target velocity that vehicle wishes to travel with, if possible [m/s]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_id(self):
        """
        for mapping purposes
        :return: vehicle's unique id
        """
        raise NotImplementedError
