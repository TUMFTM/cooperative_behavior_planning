from SimulationEnvironment.Car_Class import Car, Data_line
from lib.Behaviour import Behaviour
from lib.IDriverModel import IDriverModel
from typing import List
from SimulationEnvironment.Setting import GD
from lib.Direction import Direction


class AsymmetricIDM(IDriverModel):
    """
    Created by:
    Matthias Blum

    Modified by:
    Christian Knies

    Description:
    Class for calculating behavior according to IIDM and MOBIL model (doi:10.1098/rsta.2010.0084).
    """

    def __init__(self, simulation_time_step):
        self.simulation_time_step = round(simulation_time_step, 2)

    @classmethod
    def create(cls, simulationProviderFactory=None, simulation_time_step=None):
        """
        Returns an initialized instance of an IDriverModel implementation.
        """
        if simulationProviderFactory:
            # roadway_model = simulationProviderFactory.get_roadway_model()
            simulation_time_step = simulationProviderFactory.get_time_step_size()
        else:
            if not simulation_time_step:
                raise ValueError('no valid parameter combination is given')

        return AsymmetricIDM(simulation_time_step)

    def run(self, queue, assignedVehicles, all_vehicles, total_time, queue_reference, loop_idx, time_out=None):
        """
        Start behavior generation process with IIDM and MOBIL.

        :param queue: Queue of behaviors to fill
        :param assignedVehicles: List of vehicles assigned to this behavior model.
        :param all_vehicles: List of all vehicles in Simulation.
        :param total_time: Time step of simulation.
        :param queue_reference: Reference queue as input for trajectory planning (not implemented).
        :param loop_idx: Loop index of simulation environment.
        :param time_out: External control for time_out of planning process (not implemented).
        :return:
        """
        # calculate behaviour for all assigned vehicles
        behaviourlist = list()
        for vehicle in assignedVehicles:
            behaviourlist.append(self.calculate_behaviour(vehicle))

        # extend behavior for current time step to length of queue (total_time/simulation_time_step)
        queuelength = int(total_time / self.simulation_time_step)
        for i in range(0, queuelength):
            queue.put(behaviourlist)
        # Send sentinel when finished
        queue.put(None)
        # return empty behaviours
        return [], loop_idx

    def calculate_behaviour(self, car: Car) -> Behaviour:
        """
        Decide over lane changes and choose acceleration accordingly.

        :param car: Vehicle object for which behavior is calculated.
        :return: List of behaviors.
        """

        # parameters
        v_crit = 60 / 3.6  # [m/s]
        changing_threshold = 0.1
        bias_for_right_lane = 0.0
        b_safe_follower = -2
        # calculate all accelerations required in MOBIL model
        self.calc_IDM_values(car)
        # Set default value for direction
        direction = None

        if car.is_changing_lane:
            # car in lane change
            leader_car_a = car.a_center_car_leader
            # set car Parameters
            acceleration = leader_car_a
            a_changing_lane = None
            car_wants_to_change = None
        else:  # car not in lane change
            # gather data of ego vehicle and infrastructure
            vel_car = car.get_vel()
            ego_state_lane, ego_lane_len = car.proof_lane(car.get_lane())
            state_lane_to_the_left, lane_len_left = car.proof_lane(car.get_lane() - 1)
            state_lane_to_the_right, lane_len_right = car.proof_lane(car.get_lane() + 1)
            # gather data of left leader
            left_leader_car_data = car.environment.get_left_leader_car_data()
            left_leader_id = Data_line.get_id(left_leader_car_data)
            left_leader_car_vel = Data_line.get_vel(left_leader_car_data)
            left_leader_car_a = Data_line.get_a(left_leader_car_data)
            # gather data of leader car
            leader_car_data = car.environment.get_leader_car_data()
            leader_car_id = Data_line.get_id(leader_car_data)
            vel_leader = Data_line.get_vel(leader_car_data)

            # check left lane, lane change only if target lane longer than 400 m
            if (
                    state_lane_to_the_left == GD.ending_lane and lane_len_left > 400) or state_lane_to_the_left == GD.endless_lane:
                # left lane exists
                rop_obstruction = 0  # is vehicle obstructed by right overtaking prohibition?
                # check influence of left leader car (right overtaking prohibition)
                if left_leader_id > -1:
                    # left_leader exists
                    if (vel_car > left_leader_car_vel) and (left_leader_car_vel > v_crit) and (ego_lane_len > 400):
                        a_assym_stay = min(car.a_center_car_left_leader, car.a_center_car_leader)
                        if car.a_center_car_leader > car.a_center_car_left_leader + changing_threshold:  # obstruction by left leader
                            rop_obstruction = 1
                    else:
                        a_assym_stay = car.a_center_car_leader  # traffic jam, right overtaking allowed
                else:
                    a_assym_stay = car.a_center_car_leader  # no left_leader

                # incentive for lane change to the left
                if (car.a_left_follower_center_car > b_safe_follower) and (car.a_center_car_left_leader > GD.b_safe):
                    # safety criterion for a change to the left fullfilled
                    if ego_state_lane == GD.ending_lane:  # special case on ending lane: car wants always to change
                        car.incentive_left_change = 5
                    elif rop_obstruction == 1:  # change to left lane behind obstructing vehicle to force lane change
                        car.incentive_left_change = 2
                    else:
                        car.incentive_left_change = (car.a_center_car_left_leader - a_assym_stay +
                                                     car.IDM_v.p * (
                                                             car.a_left_follower_center_car - car.a_left_follower_left_leader))
                        car.incentive_left_change -= bias_for_right_lane
                else:
                    car.incentive_left_change = 0
            else:
                # left lane doesn't exist
                car.incentive_left_change = 0

            # check right lane, lane change only if target lane longer than 400 m
            if (
                    state_lane_to_the_right == GD.ending_lane and lane_len_right > 400) or state_lane_to_the_right == GD.endless_lane:
                # right lane exists
                if (leader_car_id > -1) or (left_leader_id > -1):
                    # leader_car or left leader car exists
                    if max(vel_leader, vel_car) > v_crit:
                        # right overtaking prohibited
                        if left_leader_id == -1 or left_leader_car_vel > vel_car:
                            # no left leader to consider
                            a_assym_change = min(car.a_center_car_right_leader, car.a_center_car_leader)
                        elif leader_car_id == -1:
                            # no leader in same lane
                            a_assym_change = min(car.a_center_car_right_leader, car.a_center_car_left_leader)
                        else:
                            # leaders in left and center lane
                            a_assym_change = min(car.a_center_car_right_leader, car.a_center_car_left_leader,
                                                 car.a_center_car_leader)
                    else:
                        a_assym_change = car.a_center_car_right_leader  # traffic jam, right overtaking allowed
                else:
                    a_assym_change = car.a_center_car_right_leader

                # incentive for lane change to the right
                if ((car.a_right_follower_center_car > b_safe_follower) and (
                        car.a_center_car_right_leader > GD.b_safe)):
                    # safety criterion for a change to the right fullfilled
                    if ego_state_lane == GD.ending_lane:  # special case on ending lane: car wants always to change
                        car.incentive_left_change = 4
                    else:
                        car.incentive_right_change = (
                                a_assym_change - car.a_center_car_leader + car.IDM_v.p * (car.a_follower_and_leader
                                                                                          - car.a_follower_center_car))
                        car.incentive_right_change += bias_for_right_lane
                else:
                    car.incentive_right_change = 0
            else:
                # right lane doesn't exist
                car.incentive_right_change = 0

            # decide if a lane change should be performed and what direction
            if (car.incentive_right_change <= changing_threshold) and (car.incentive_left_change <= changing_threshold):
                # no lane change, because the incentives are too low
                car_wants_to_change = False
                # calculate acceleration
                if left_leader_id > -1:  # right overtaking prohibition
                    if (vel_car > left_leader_car_vel) and (left_leader_car_vel > v_crit) and (ego_lane_len > 400):
                        if car.a_center_car_left_leader > -4:
                            # obey right overtaking prohibition only if acceleration > -4 m/s^2
                            acceleration = min(car.a_center_car_left_leader, car.a_center_car_leader)
                        else:
                            acceleration = car.a_center_car_leader
                    else:  # traffic jam or ending lane --> no right overtaking prohibition
                        acceleration = car.a_center_car_leader
                else:
                    acceleration = car.a_center_car_leader
            else:
                if car.incentive_right_change > car.incentive_left_change:
                    # change to the right
                    car_wants_to_change = True
                    direction = Direction.right
                    acceleration = min(car.a_center_car_right_leader, car.a_center_car_leader)
                else:
                    # change to the left
                    car_wants_to_change = True
                    direction = Direction.left
                    acceleration = min(car.a_center_car_left_leader, car.a_center_car_leader)
        # Create behaviour object
        behaviour = Behaviour(vehicle_id=car.get_id(), acc=acceleration,
                              lane_change=car_wants_to_change,
                              direction=direction)
        return behaviour

    def calc_IDM_values(self, car: Car):
        """
        Calculate all accelerations required in MOBIL model

        :param car: Vehicle object
        :return:
        """
        id_center_car = int(car.get_id())
        lane_center_car = car.get_lane()
        id_leader = car.id_leader
        id_follower = car.id_follower
        id_left_leader = car.id_left_leader
        id_left_follower = car.id_left_follower
        id_right_leader = car.id_right_leader
        id_right_follower = car.id_right_follower
        lane = car.get_lane()
        left_lane = lane - 1
        right_lane = lane + 1

        a_left_follower_left_leader = self.IIDM_calculate_a(id_left_follower,
                                                            id_left_leader, left_lane)  # b'f'

        a_center_car_left_leader = self.IIDM_calculate_a(id_center_car,
                                                         id_left_leader, left_lane)  # c f'

        a_left_follower_center_car = self.IIDM_calculate_a(id_left_follower,
                                                           id_center_car, left_lane)  # b'c

        a_center_car_leader = self.IIDM_calculate_a(id_center_car,
                                                    id_leader, lane)  # cf

        a_follower_and_leader = self.IIDM_calculate_a(id_follower,
                                                      id_leader, lane)  # bf

        a_follower_center_car = self.IIDM_calculate_a(id_follower,
                                                      id_center_car, lane)  # bc

        a_right_follower_center_car = self.IIDM_calculate_a(id_right_follower,
                                                            id_center_car, right_lane)  # b''c

        a_right_follower_right_leader = self.IIDM_calculate_a(id_right_follower,
                                                              id_right_leader, right_lane)  # b''f''

        a_center_car_right_leader = self.IIDM_calculate_a(id_center_car,
                                                          id_right_leader, right_lane)  # cf''

        car.a_left_follower_left_leader = a_left_follower_left_leader
        car.a_center_car_left_leader = a_center_car_left_leader
        car.a_left_follower_center_car = a_left_follower_center_car
        car.a_center_car_leader = a_center_car_leader
        car.a_follower_and_leader = a_follower_and_leader
        car.a_follower_center_car = a_follower_center_car
        car.a_right_follower_center_car = a_right_follower_center_car
        car.a_right_follower_right_leader = a_right_follower_right_leader
        car.a_center_car_right_leader = a_center_car_right_leader

        if id_left_leader > -1:
            car.x_left_leader = GD.all_cars_object_list[id_left_leader].get_pos()
        if id_left_follower > -1:
            car.x_left_follower = GD.all_cars_object_list[id_left_follower].get_pos()
        if id_center_car > -1:
            car.x_center_car = GD.all_cars_object_list[id_center_car].get_pos()
        if id_leader > -1:
            car.x_leader = GD.all_cars_object_list[id_leader].get_pos()
        if id_follower > -1:
            car.x_follower = GD.all_cars_object_list[id_follower].get_pos()
        if id_right_follower > -1:
            car.x_right_follower = GD.all_cars_object_list[id_right_follower].get_pos()
        if id_right_leader > -1:
            car.x_right_leader = GD.all_cars_object_list[id_right_leader].get_pos()

    def IIDM_calculate_a(self, id_follower_car: int, id_leader_car: int, lane_id):
        """"
        Calculate acceleration of ego (follower) vehicle according to IIDM model.

        :param id_follower_car: ID ego vehicle.
        :param id_leader_car: ID leader vehicle.
        :param lane_id: Lane ID.

        :return: Acceleration a
        """

        lane_status, lane_len = GD.all_cars_object_list[id_follower_car].proof_lane(
            lane_id)  # use env_lane of following car

        if (lane_status == GD.no_lane) or (lane_status == GD.future_lane):
            a = -10
        else:
            if id_follower_car == -1:
                # no follower car
                a = 0
                return a
            else:
                IDM_var_follower = GD.all_cars_object_list[id_follower_car].IDM_v
                data_follower_car = GD.all_cars_object_list[id_follower_car].data
                len_follower = Data_line.get_len(data_follower_car)
                v_follower = Data_line.get_vel(data_follower_car)
                pos_follower = Data_line.get_pos(data_follower_car)

                if id_leader_car == -1:
                    # no leader vehicle
                    if lane_status == GD.endless_lane:  # check if lane ends
                        # no leader vehicle
                        a = IDM_var_follower.a * (1 - (v_follower / IDM_var_follower.V_0) ** IDM_var_follower.delta)
                        return a
                    if lane_status == GD.ending_lane:
                        # the lane will end, so it is a non moving obstacle
                        v_leader = 0
                        # desired behavior: accelerate until braking distance before lane ending
                        brake_distance = abs(-0.5 / GD.max_break * v_follower ** 2)
                        if brake_distance < lane_len - len_follower:
                            delta_x = 1000
                            delta_v = 0
                        else:
                            delta_x = max(lane_len - len_follower, 0.1)
                            delta_v = v_follower - v_leader

                        s_star = IDM_var_follower.s0 + max(
                            IDM_var_follower.T * v_follower + v_follower * delta_v / IDM_var_follower.C1, 0)
                        z = s_star / delta_x

                        if z >= 1:
                            a = IDM_var_follower.a * (1 - (z ** 2))
                        else:
                            a_f = max(0.01, IDM_var_follower.a * (
                                        1 - (v_follower / IDM_var_follower.V_0) ** IDM_var_follower.delta))
                            a = a_f * (1 - (z ** (2 * IDM_var_follower.a / a_f)))
                        return a

                else:  # there are leader and follower
                    data_leader_car = GD.all_cars_object_list[id_leader_car].data
                    v_leader = Data_line.get_vel(data_leader_car)
                    pos_leader = Data_line.get_pos(data_leader_car)
                    delta_v = v_follower - v_leader
                    delta_x = max(pos_leader - pos_follower - len_follower, 0.1)

                    if pos_follower + len_follower - pos_leader >= 0:
                        # special case if vehicle next to each other on neighbor lanes
                        return GD.a_crash  # vehicles would collide in case of a lane change
                    else:
                        s_star = IDM_var_follower.s0 + max(
                            IDM_var_follower.T * v_follower + v_follower * delta_v / IDM_var_follower.C1, 0)
                        z = s_star / delta_x

                        if z >= 1:
                            a = IDM_var_follower.a * (1 - (z ** 2))
                        else:
                            a_f = max(0.000001, IDM_var_follower.a * (
                                    1 - (v_follower / IDM_var_follower.V_0) ** IDM_var_follower.delta))
                            a = a_f * (1 - (z ** (2 * IDM_var_follower.a / a_f)))

                    # test if there is a closer lane-finish:
                    if lane_status == GD.ending_lane:
                        # compare if the lane end affects the driver more than the leader car
                        # desired behavior: accelerate until braking distance before lane ending
                        brake_distance = abs(-0.5 / GD.max_break * v_follower ** 2)
                        if brake_distance < lane_len - len_follower:
                            delta_x = 1000
                            delta_v = 0
                        else:
                            delta_x = max(lane_len - len_follower, 0.1)
                            delta_v = v_follower - v_leader

                        s_star = IDM_var_follower.s0 + max(
                            IDM_var_follower.T * v_follower + v_follower * delta_v / IDM_var_follower.C1, 0)
                        z = s_star / delta_x

                        if z >= 1:
                            a_ending_lane = IDM_var_follower.a * (1 - (z ** 2))
                        else:
                            a_f = max(0.000001, IDM_var_follower.a * (
                                    1 - (v_follower / IDM_var_follower.V_0) ** IDM_var_follower.delta))
                            a_ending_lane = a_f * (1 - (z ** (2 * IDM_var_follower.a / a_f)))
                        a = min(a, a_ending_lane)
        return a
