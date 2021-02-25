import pandas as pd
import numpy as np
from SimulationEnvironment.Car_Class import IDM_Variables
from DriverModels.A_star.actions import calculate_a_IIDM
from SimulationEnvironment.Setting import GD
from scipy.optimize import minimize


class ParameterEstimator:
    """
    Created by:
    Christian Knies

    Description:
    Class estimates unknown parameters for all vehicles.
    """

    def __init__(self, roadway_model):
        self.vehicle_data = pd.DataFrame()
        self.vehicle_data_history = []
        self.columnlist = ["id", "position", "max_velocity", "velocity", "acceleration", "lane", "is_changing_lane",
                           "destination_lane", "length", "width", "lc_duration", "lc_timer", "IDM_param",
                           "front_distance", "v_leader", "position_leader"]
        self.index = "id"
        self.roadway = roadway_model

    def initialize(self, all_vehicle_objects):
        """
        Initialize dataframe in first run.

        :param all_vehicle_objects: All vehicle objects in planning process.
        :return:
        """
        # initialize data frame
        carlist = []
        for car in all_vehicle_objects:
            carlist.append(
                [car.get_id(), car.get_pos(), car.get_vel(), car.get_vel(), car.get_a(), car.get_lane(),
                 car.get_is_changing_lane(), car.get_destination_lane(), car.get_len(), car.get_width(),
                 self.estimate_lc_duration(car.get_len()), 0, IDM_Variables(), -1, -1, -1])
        self.vehicle_data = pd.DataFrame(carlist)
        # define column names
        self.vehicle_data.columns = self.columnlist
        self.vehicle_data.set_index(self.index)

    def update(self, all_vehicle_objects, dt):
        """
        Update vehicle data for planning new time step (new planning process).

        :param all_vehicle_objects: All vehicle objects in planning process.
        :param dt: Planning time step.
        :return:
        """
        # initialize if data frame is empty
        if self.vehicle_data.empty:  # if dataframe is empty in first run
            self.initialize(all_vehicle_objects)
        else:
            # calculate distance to front vehicle and difference speed
            for car in all_vehicle_objects:
                leaders_in_lane = self.vehicle_data[
                    (abs(self.vehicle_data["lane"] - self.vehicle_data.at[car.get_id(), "lane"]) <= 0.5) &
                    self.vehicle_data["position"] > self.vehicle_data.at[car.get_id(), "position"]]
                if not leaders_in_lane.empty:
                    leader_id = leaders_in_lane["position"].idxmin()
                    self.vehicle_data.at[car.get_id(), "front_distance"] = leaders_in_lane.at[leader_id, "position"] - \
                                                                           self.vehicle_data.at[
                                                                               car.get_id(), "position"] - car.get_len()
                    self.vehicle_data.at[car.get_id(), "v_leader"] = leaders_in_lane.at[leader_id, "velocity"]
                    self.vehicle_data.at[car.get_id(), "position_leader"] = leaders_in_lane.at[leader_id, "position"]
                else:
                    self.vehicle_data.at[car.get_id(), "front_distance"] = 1000
                    self.vehicle_data.at[car.get_id(), "v_leader"] = 1000
                    self.vehicle_data.at[car.get_id(), "position_leader"] = self.vehicle_data.at[
                                                                                car.get_id(), "position"] + 1000
            # copy dataframe to history
            self.vehicle_data_history.append(self.vehicle_data.copy(deep=True))
        # update all time dependent values
        for car in all_vehicle_objects:
            self.vehicle_data.at[car.get_id(), "position"] = car.get_pos()
            self.vehicle_data.at[car.get_id(), "max_velocity"] = max(self.vehicle_data.at[car.get_id(), "max_velocity"],
                                                                     car.get_vel())
            self.vehicle_data.at[car.get_id(), "velocity"] = car.get_vel()
            self.vehicle_data.at[car.get_id(), "acceleration"] = car.get_a()
            self.vehicle_data.at[car.get_id(), "lane"] = car.get_lane()
            self.vehicle_data.at[car.get_id(), "is_changing_lane"] = car.get_is_changing_lane()
            self.vehicle_data.at[car.get_id(), "destination_lane"] = car.get_destination_lane()
            self.vehicle_data.at[car.get_id(), "lc_timer"] = self.estimate_lc_timer(car)
            # set method for estimating desired velocity v0: estimated or received (true) v0
            self.estimate_IDM_param_static(car, self.estimate_v0, dt)

    def get_parameter(self, car_object):
        """
        Provide parameters from perspective of ego vehicle (true values for ego vehicle, estimated values for
        other vehicles).
        :param car_object: Ego vehicle object.
        :return: Vehicle data dataframe.
        """
        # find all cars in range
        cars_in_range = self.vehicle_data[
            abs(self.vehicle_data["position"] - self.vehicle_data.at[car_object.get_id(), "position"]) < 400]
        env_cars_in_range = cars_in_range[cars_in_range["id"] != car_object.get_id()]
        ego_car = self.get_ground_truth(car_object)
        return ego_car.append(env_cars_in_range)

    def get_ground_truth(self, car):
        """
        Provide tue data for car object.

        :param car: Ego vehicle object.
        :return: Data frame with true values.
        """
        ground_truth_data = [car.get_id(), car.get_pos(), car.IDM_v.V_0, car.get_vel(), car.get_a(), car.get_lane(),
                             car.get_is_changing_lane(), car.get_destination_lane(), car.get_len(), car.get_width(),
                             car.get_lc_duration(), car.get_timeout_for_lane_change(), car.IDM_v, -1, -1, -1]
        ego_vehicle = pd.DataFrame(ground_truth_data).T
        ego_vehicle.columns = self.columnlist
        ego_vehicle.set_index(self.index)
        return ego_vehicle

    def estimate_lc_duration(self, length):
        """
        Estimate lane change duration based on car length (length < 10 meter --> car, truck otherwise).

        :param length: Vehicle length.
        :return: Lane change duration.
        """
        if length < 10:  # if length < 10 m -> Car
            lc_duration = 4  # [s]
        else:  # if length > 10 m -> Truck
            lc_duration = 6  # [s]
        return lc_duration

    def estimate_lc_timer(self, car_object):
        """
        Estimate time until lane change is completed.

        :param car_object:  Vehicle object of lane changing car.
        :return: Estimated time until lane change is finished.
        """
        if not car_object.is_changing_lane:  # car not in lane change
            lc_timeout = 0
        else:  # car in lane change
            lc_timeout = self.estimate_lc_duration(car_object.get_len()) - (
                        car_object.get_lc_duration() - car_object.get_timeout_for_lane_change())
        return lc_timeout

    def estimate_v0(self, car_object):
        """
        Estimate desired velocity v0 for vehicle object based on max. observed velocity, lane of vehicle
        (on entry lane or not) and distance to leader vehicle (tailgating).
        :param car_object: Vehicle object.
        :return: Estimated desired velocity.
        """
        # standard parameters
        tailgate_ttc = 5.8
        tailgate_v_plus = 2
        tailgate_thw = 1
        # overwrite parameter if externally set
        if GD.ext_params is not None:
            if "tailgate_ttc" in GD.ext_params.keys():
                tailgate_ttc = GD.ext_params["tailgate_ttc"]
            if "tailgate_thw" in GD.ext_params.keys():
                tailgate_thw = GD.ext_params["tailgate_thw"]
            if "tailgate_v_plus" in GD.ext_params.keys():
                tailgate_v_plus = GD.ext_params["tailgate_v_plus"]
        # distance to lane end
        dist_lane_end = max(self.roadway.get_dist_to_lane_end(self.vehicle_data.at[car_object.get_id(), "position"],
                                                              self.vehicle_data.at[car_object.get_id(), "lane"]) -
                            self.vehicle_data.at[car_object.get_id(), "length"], 0.1)
        # if vehicle on entry lane: v0 = 36 m/s for cars and 23 m/s for trucks
        if car_object.get_len() < 10 and dist_lane_end < 400:  # Car on ending lane
            v0_entry = 36
        elif car_object.get_len() > 10 and dist_lane_end < 400:  # Truck on ending lane
            v0_entry = 23
        else:  # vehicle not on ending lane
            v0_entry = 0
        # increase v0 if a vehicle is tailgating (real v0 might be higher than former max_velocity
        leaders_in_lane = self.vehicle_data[
            (self.vehicle_data["position"] > self.vehicle_data.at[car_object.get_id(), "position"]) & (
                        self.vehicle_data["lane"] == self.vehicle_data.at[car_object.get_id(), "lane"])]
        if leaders_in_lane.empty is True:
            # no leader car
            v0_tailgate = 0
        elif (min(leaders_in_lane["position"]) - self.vehicle_data.at[car_object.get_id(), "position"] -
              self.vehicle_data.at[car_object.get_id(), "length"]) / self.vehicle_data.at[
            car_object.get_id(), "velocity"] <= tailgate_thw or (
                min(leaders_in_lane["position"]) - self.vehicle_data.at[car_object.get_id(), "position"] -
                self.vehicle_data.at[car_object.get_id(), "length"]) / max((self.vehicle_data.at[
                                                                                car_object.get_id(), "velocity"] -
                                                                            leaders_in_lane.at[leaders_in_lane[
                                                                                                   "position"].idxmin(), "velocity"]),
                                                                           0.1) <= tailgate_ttc:
            # tailgating vehicle: v0 expected to be higher than actual velocity
            v0_tailgate = self.vehicle_data.at[car_object.get_id(), "velocity"] + tailgate_v_plus
        else:
            # car not tailgating
            v0_tailgate = 0
        # v0 is max of former max. velocity, v0_entry and v0_tailgate
        desired_velocity = max(self.vehicle_data.at[car_object.get_id(), "max_velocity"], v0_entry, v0_tailgate)
        return desired_velocity

    def receive_v0(self, car_object):
        """
        Read true value of desired velocity v0 for vehicle object (e.g. received via CAM message).
        :param car_object: Vehicle object.
        :return: Estimated desired velocity.
        """
        # distance to lane end
        dist_lane_end = max(self.roadway.get_dist_to_lane_end(self.vehicle_data.at[car_object.get_id(), "position"],
                                                              self.vehicle_data.at[car_object.get_id(), "lane"]) -
                            self.vehicle_data.at[car_object.get_id(), "length"], 0.1)
        # if vehicle on entry lane: v0 = 36 m/s for cars and 23 m/s for trucks
        if car_object.get_len() < 10 and dist_lane_end < 400:  # Car on ending lane
            v0_entry = 36
        elif car_object.get_len() > 10 and dist_lane_end < 400:  # Truck on ending lane
            v0_entry = 23
        else:  # vehicle not on ending lane
            v0_entry = 0
        # read true desired velocity from car_object
        desired_velocity = max(car_object.IDM_v.V_0, car_object.get_vel(), v0_entry)
        return desired_velocity

    def estimate_IDM_param_static(self, car_object, get_v0, dt):
        """
        Estimate IIDM values of vehicle object based on vehicle length (<10 m: car, truck otherwise).

        :param car_object: Vehicle object to estimate values for.
        :param get_v0: Method for determinig v0: self.estimate_v0 or self.receive_v0
        :param dt: Planning time step.
        :return:
        """
        # set standard parameters
        IIDM_param = {}
        IIDM_param["Car"] = {}
        IIDM_param["Car"]["T"] = 0.5
        IIDM_param["Car"]["param_a"] = 1.4
        IIDM_param["Car"]["param_b"] = 2.0
        IIDM_param["Car"]["param_delta"] = 4
        IIDM_param["Car"]["param_p"] = 0.2
        IIDM_param["Car"]["s0"] = 2
        IIDM_param["Car"]["v0"] = get_v0(car_object)
        IIDM_param["Truck"] = {}
        IIDM_param["Truck"]["T"] = 0.5
        IIDM_param["Truck"]["param_a"] = 0.7
        IIDM_param["Truck"]["param_b"] = 2
        IIDM_param["Truck"]["param_delta"] = 4
        IIDM_param["Truck"]["param_p"] = 0.2
        IIDM_param["Truck"]["s0"] = 4
        IIDM_param["Truck"]["v0"] = get_v0(car_object)

        # overwrite parameter from file
        if GD.ext_params is not None:
            if "truck_delta" in GD.ext_params.keys():
                IIDM_param["Truck"]["param_delta"] = GD.ext_params["truck_delta"]
            if "car_delta" in GD.ext_params.keys():
                IIDM_param["Car"]["param_delta"] = GD.ext_params["car_delta"]
            if "truck_T" in GD.ext_params.keys():
                IIDM_param["Truck"]["T"] = GD.ext_params["truck_T"]
            if "car_T" in GD.ext_params.keys():
                IIDM_param["Car"]["T"] = GD.ext_params["car_T"]
            if "p" in GD.ext_params.keys():
                IIDM_param["Truck"]["param_p"] = GD.ext_params["p"]
            if "p" in GD.ext_params.keys():
                IIDM_param["Car"]["param_p"] = GD.ext_params["p"]
            if "car_a" in GD.ext_params.keys():
                IIDM_param["Car"]["param_a"] = GD.ext_params["car_a"]
            if "car_b" in GD.ext_params.keys():
                IIDM_param["Car"]["param_b"] = GD.ext_params["car_b"]
            if "truck_a" in GD.ext_params.keys():
                IIDM_param["Truck"]["param_a"] = GD.ext_params["truck_a"]
            if "truck_b" in GD.ext_params.keys():
                IIDM_param["Truck"]["param_b"] = GD.ext_params["truck_b"]

        # decide whether car or truck
        if car_object.get_len() < 10:  # if length < 10 m -> Car
            self.vehicle_data.at[car_object.get_id(), "IDM_param"].set_Variables(
                V_0=IIDM_param["Car"]["v0"],
                T=IIDM_param["Car"]["T"],
                a=IIDM_param["Car"]["param_a"],
                b=IIDM_param["Car"]["param_b"],
                delta=IIDM_param["Car"]["param_delta"],
                s0=IIDM_param["Car"]["s0"],
                p=IIDM_param["Car"]["param_p"])
        else:  # if length > 10 m -> Truck
            self.vehicle_data.at[car_object.get_id(), "IDM_param"].set_Variables(
                V_0=IIDM_param["Truck"]["v0"],
                T=IIDM_param["Truck"]["T"],
                a=IIDM_param["Truck"]["param_a"],
                b=IIDM_param["Truck"]["param_b"],
                delta=IIDM_param["Truck"]["param_delta"],
                s0=IIDM_param["Truck"]["s0"],
                p=IIDM_param["Truck"]["param_p"])
