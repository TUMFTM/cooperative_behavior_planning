from itertools import product
from DriverModels.A_star.costs import *
from functools import partial
from DriverModels.A_star.actions import *


class Node:
    """
    Created by:
    Christian Knies

    Description:
    A node class for A-Star behavior planning
    """

    def __init__(self, roadway, vehicle_id, planning_car_id, length, lc_duration, parent=None, position=None,
                 velocity=None, acceleration=None, lane=None, isChangingLane=0,
                 destinationLane=None, lc_timer=None, IDM_param=None, max_a={}):
        self.parent = parent
        self.id = 0
        self.vehicle_id = vehicle_id
        self.planning_car_id = planning_car_id
        self.max_a = max_a
        self.roadway = roadway
        self.length = length
        self.lc_duration = lc_duration
        self.IDM_param = IDM_param
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.lane = lane
        self.isChangingLane = isChangingLane
        self.destinationLane = destinationLane
        self.lc_timer = lc_timer
        self.combination = []
        if self.parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1
        self.cost_so_far = 0
        self.cost_heuristic = 0
        self.priority = 0

    def get_children(self, roadway, action_dict_car, action_dict_truck, prediction_dict_car, prediction_dict_truck,
                     planning_time_step, planning_horizon, pool):
        """
        Expand all child nodes of the node "self" (all possible combinations of action sets).

        :param roadway: Roadway model.
        :param action_dict_car: Possible actions of planning cars.
        :param action_dict_truck: Possible actions of planning trucks.
        :param prediction_dict_car: Possible actions (only one predicted action) of prediction car
        (vehicles for which there is no planning.
        :param prediction_dict_truck: Possible actions (only one predicted action) of prediction truck
        (vehicles for which there is no planning.
        :param planning_time_step: Planning time step.
        :param planning_horizon: Planning horizon in time steps.
        :param pool: Pool for multiprocessing if multiprocessing was set active.
        :return: List of all child nodes.
        """
        num_vehicles = len(self.position)
        actionsetlist = []  # list of actions for each vehicle [[action1_vehicle1, action2_veh1],[action1]]
        # find available actions for each vehicle
        for vehicle in range(0, num_vehicles):
            # vehicle is a car with full action set
            if self.length[vehicle] < 10 and self.vehicle_id[vehicle] in self.planning_car_id:
                actionset = list(action_dict_car.keys())
                action_dict = action_dict_car
                max_acceleration = self.max_a["car"]
            # vehicle is a car with prediction set
            elif self.length[vehicle] < 10 and self.vehicle_id[vehicle] not in self.planning_car_id:
                actionset = list(prediction_dict_car.keys())
                action_dict = prediction_dict_car
                max_acceleration = self.max_a["car"]
            # vehicle is truck with full action set
            elif self.length[vehicle] > 10 and self.vehicle_id[vehicle] in self.planning_car_id:
                actionset = list(action_dict_truck.keys())
                action_dict = action_dict_truck
                max_acceleration = self.max_a["truck"]
            # vehicle is truck with prediction set
            else:
                actionset = list(prediction_dict_truck.keys())
                action_dict = prediction_dict_truck
                max_acceleration = self.max_a["truck"]
            # remove impossible lane changes to left (no lane or no gap)
            if 'LCL' in actionset:
                if roadway.get_left_most_lane(self.position[vehicle]) >= self.lane[vehicle] or self.isChangingLane[
                                vehicle] == 1:
                    actionset.remove('LCL')
                else:
                    lc_direction = -1
                    forbid_lcl = self.check_lc(vehicle, lc_direction)
                    if forbid_lcl >= 1:
                        actionset.remove('LCL')
            # remove impossible lane changes to left (no lane or no gap)
            if 'LCR' in actionset:
                if roadway.get_right_most_lane(self.position[vehicle]) <= self.lane[vehicle] or self.isChangingLane[
                                vehicle] == 1:
                    actionset.remove('LCR')
                else:
                    lc_direction = 1
                    forbid_lcr = self.check_lc(vehicle, lc_direction)
                    if forbid_lcr >= 1:
                        actionset.remove('LCR')
            # exclude accelerations near IIDM acceleration
            if self.vehicle_id[vehicle] in self.planning_car_id:
                a_thresh = 0.1
                IIDM_acceleration = IIDM(self, planning_time_step, 0, vehicle)[2]
                coast_acceleration = coast(self, planning_time_step, 0, vehicle)[2]
                for action in actionset:
                    if action_dict[action][0] == const_acceleration and abs(
                            action_dict[action][1] - IIDM_acceleration) < a_thresh:
                        actionset.remove(action)
                    elif action_dict[action][0] == coast and abs(coast_acceleration - IIDM_acceleration) < a_thresh:
                        actionset.remove(action)
            # list of possible actions for all vehicles
            actionsetlist.append(actionset)

        # generate all children nodes (karthesian product of all actions)
        if pool is False:
            # single processing
            children = []
            for combination in product(*actionsetlist):
                # generate new child
                new_child = self.generate_new_child(combination, action_dict_car, action_dict_truck,
                                                    prediction_dict_car, prediction_dict_truck, planning_time_step,
                                                    planning_horizon, num_vehicles)
                # append children list with new_child
                children.append(new_child)
            return children
        else:
            # multi processing
            new_child = partial(self.generate_new_child, action_dict_car=action_dict_car,
                                action_dict_truck=action_dict_truck, prediction_dict_car=prediction_dict_car,
                                prediction_dict_truck=prediction_dict_truck, planning_time_step=planning_time_step,
                                planning_horizon=planning_horizon,
                                num_vehicles=num_vehicles)
            children = pool.map(new_child, product(*actionsetlist))
            return children

    def generate_new_child(self, combination, action_dict_car, action_dict_truck, prediction_dict_car,
                           prediction_dict_truck, planning_time_step, planning_horizon, num_vehicles):
        """
        Generate one new child with specified actions.

        :param combination: Combination of actions to perform for new node.
        :param action_dict_car: Possible actions of planning cars.
        :param action_dict_truck: Possible actions of planning trucks.
        :param prediction_dict_car: Possible actions (only one predicted action) of prediction car
        (vehicles for which there is no planning.
        :param prediction_dict_truck: Possible actions (only one predicted action) of prediction truck
        (vehicles for which there is no planning.
        :param planning_time_step: Planning time step.
        :param planning_horizon: Planning horizon in time steps.
        :param num_vehicles: Number of vehicles in planning process.
        :return: New child node.
        """
        # initialize state of new child
        new_parent = self
        new_position = np.zeros(num_vehicles)
        new_velocity = np.zeros(num_vehicles)
        new_acceleration = np.zeros(num_vehicles)
        new_lane = np.zeros(num_vehicles)
        new_isChangingLane = np.zeros(num_vehicles)
        new_destinationLane = np.zeros(num_vehicles)
        new_lc_timer = np.zeros(num_vehicles)
        # expand all actions of one combination
        for vehicle in range(0, num_vehicles):
            # vehicle is car with full action set
            if self.length[vehicle] < 10 and self.vehicle_id[vehicle] in self.planning_car_id:
                action, param = action_dict_car[combination[vehicle]]
            # vehicle is car with prediction set
            elif self.length[vehicle] < 10 and self.vehicle_id[vehicle] not in self.planning_car_id:
                action, param = prediction_dict_car[combination[vehicle]]
            # vehicle is truck with full action set
            elif self.length[vehicle] >= 10 and self.vehicle_id[vehicle] in self.planning_car_id:
                action, param = action_dict_truck[combination[vehicle]]
            # vehicle is truck with prediction set
            elif self.length[vehicle] >= 10 and self.vehicle_id[vehicle] not in self.planning_car_id:
                action, param = prediction_dict_truck[combination[vehicle]]
            new_position[vehicle], new_velocity[vehicle], new_acceleration[vehicle], new_lane[vehicle], \
            new_isChangingLane[vehicle], new_destinationLane[vehicle], new_lc_timer[vehicle] = action(self,
                                                                                                      planning_time_step,
                                                                                                      param, vehicle)
        # create new child
        new_child = Node(self.roadway, self.vehicle_id, self.planning_car_id, self.length, self.lc_duration, new_parent,
                         new_position,
                         new_velocity, new_acceleration, new_lane, new_isChangingLane,
                         new_destinationLane, new_lc_timer, self.IDM_param, self.max_a)
        # set combination of actions
        new_child.combination = combination
        # calculate cost_so_far, heuristic and priority = cost_so_far + heuristic
        cost_evaluator(new_child, planning_time_step, planning_horizon)
        return new_child

    def check_lc(self, vehicle, lc_direction):
        """
        Check if lane change is possible  to exclude it if not (prune tree).

        :param vehicle: Number of vehicle.
        :param lc_direction: Direction of lane change.
        :return: Safety cost of lane change (1: lane change not safe, 0 otherwise).
        """
        t_thresh = 0.3
        # check leader car on target lane
        lane = self.lane[vehicle] + lc_direction
        leaders_in_target_lane = np.where(
            (self.position >= self.position[vehicle]) & (lane - 0.5 <= self.lane) & (self.lane <= lane + 0.5))
        if len(leaders_in_target_lane[0]) == 0:  # no leader car
            safety_cost_front = 0
        else:
            leader_car = np.argmin(self.position[leaders_in_target_lane])
            d_le = self.position[leaders_in_target_lane][leader_car] - self.position[vehicle] - self.length[vehicle]
            if d_le < 0:
                safety_cost_front = 1
            else:
                v_le = self.velocity[leaders_in_target_lane][leader_car]
                safety_cost_front = evaluate_safety(d_le, v_le, self.velocity[vehicle], t_thresh)
        # check follower car on target lane
        followers_in_target_lane = np.where(
            (self.position < self.position[vehicle]) & (lane - 0.5 <= self.lane) & (self.lane <= lane + 0.5))
        if len(followers_in_target_lane[0]) == 0:  # no follower car
            safety_cost_rear = 0
        else:
            follower_car = np.argmax(self.position[followers_in_target_lane])
            d_fo = self.position[vehicle] - self.position[followers_in_target_lane][follower_car] - \
                   self.length[followers_in_target_lane][follower_car]
            if d_fo < 0:
                safety_cost_rear = 1
            else:
                v_fo = self.velocity[followers_in_target_lane][follower_car]
                safety_cost_rear = evaluate_safety(d_fo, self.velocity[vehicle], v_fo, t_thresh)
        return max(safety_cost_front, safety_cost_rear)

    def copy_node(self):
        """
        Copy current node (deepcopy).

        :return: Copied instance of node.
        """
        out_node = Node(self.roadway, self.vehicle_id, self.planning_car_id, self.length, self.lc_duration, None,
                        self.position.copy(), self.velocity.copy(), self.acceleration, self.lane, self.isChangingLane,
                        self.destinationLane, self.lc_timer, self.IDM_param, self.max_a)
        return out_node
