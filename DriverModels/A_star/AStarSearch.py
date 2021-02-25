from lib.IDriverModel import IDriverModel
from DriverModels.A_star.PriorityQueue import *
from DriverModels.A_star import actions
from DriverModels.A_star.Node import Node
import numpy as np
from lib.Behaviour import Behaviour
from lib.Direction import Direction
import common.user_settings as cfg
import time
from multiprocessing import Pool


class AStarSearch(IDriverModel):
    """
    Created by:
    Christian Knies

    Description:
    Class for central cooperative behavior planning with A-Star search
    """

    def __init__(self, simulation_time_step, roadway_model):
        self.simulation_time_step = round(simulation_time_step, 2)
        self.roadway = roadway_model

    @classmethod
    def create(cls, simulationProviderFactory=None, simulation_time_step=None, roadway_model=None):
        """
        Returns an initialized instance of an IDriverModel implementation.
        """
        if simulationProviderFactory:
            roadway_model = simulationProviderFactory.get_roadway_model()
            simulation_time_step = simulationProviderFactory.get_time_step_size()
        else:
            if not simulation_time_step or not roadway_model:
                raise ValueError('no valid parameter combination is given')
        return AStarSearch(simulation_time_step, roadway_model)

    def run(self, queue, assignedVehicles, all_vehicles, total_time, queue_reference, loop_idx, time_out=None):
        """
        Start central A-Star planning process

        :param queue: Queue of behaviors to fill
        :param assignedVehicles: List of vehicles assigned to this behavior model.
        :param all_vehicles: List of all vehicles in Simulation.
        :param total_time: Time step of simulation.
        :param queue_reference: Reference queue as input for trajectory planning (not implemented).
        :param loop_idx: Loop index of simulation environment.
        :param time_out: External control for time_out of planning process (not implemented).
        :return:
        """
        # check if all vehicles are assigned to the planner
        if assignedVehicles != all_vehicles:
            raise Exception('all vehicles must be assigned to the global planner')

        # parameters
        planning_time_step = 2.5  # [s]
        planning_horizon = int(cfg.sim_time / planning_time_step)  # in time steps

        # create start node in first run
        start_position = np.array([])
        start_velocity = np.array([])
        start_acceleration = np.array([])
        start_lane = np.array([])
        IDM_param = np.array([])
        length = np.array([])
        lc_duration = np.array([])
        is_changing_lane = np.zeros(len(all_vehicles))
        destination_lane = np.zeros(len(all_vehicles))
        lc_timer = np.zeros(len(all_vehicles))
        vehicle_id = np.array([])
        for car in all_vehicles:
            start_position = np.append(start_position, car.get_pos())
            start_velocity = np.append(start_velocity, car.get_vel())
            start_acceleration = np.append(start_acceleration, car.get_a())
            start_lane = np.append(start_lane, car.get_lane())
            IDM_param = np.append(IDM_param, car.IDM_v)
            length = np.append(length, car.get_len())
            lc_duration = np.append(lc_duration, car.get_lc_duration())
            vehicle_id = np.append(vehicle_id, car.get_id())
        start = Node(self.roadway, vehicle_id, vehicle_id, length, lc_duration, None, start_position, start_velocity,
                     start_acceleration, start_lane, is_changing_lane, destination_lane, lc_timer, IDM_param,
                     {"car": 0, "truck": 0})

        # calculate best behaviour
        behaviourlist = self.a_star_search(start, planning_horizon, planning_time_step, self.roadway)

        # extend behavior for current time step to length of queue (total_time/simulation_time_step)
        queuelength = int(planning_time_step / self.simulation_time_step)
        while behaviourlist:  # as long as behaviourlist is not empty
            current_behaviour = behaviourlist.pop(0)
            for i in range(0, queuelength):
                queue.put(current_behaviour)
        # Send sentinel when finished
        queue.put(None)
        # return behaviours
        return [], loop_idx

    def a_star_search(self, start, planning_horizon, planning_time_step, roadway):
        """
        A-Star planning process

        :param all_vehicles: List of all vehicles in Simulation.
        :param start: Root node of planning process.
        :param planning_horizon: Planning horizon of A-Star search in time steps.
        :param planning_time_step: Time between two planning time steps.
        :param roadway: Roadway model.
        :return: List of behaviors for execution in simulation environment.
        """
        # settings
        use_multiprocessing = False
        # Initialization
        starting_time = time.time()
        timeout_interval = planning_horizon * planning_time_step * 100  # 100 times length of simulated scenario
        timeout = timeout_interval
        current_depth = 0
        top_node = start
        previous_depth = current_depth
        node_id = 0
        frontier = PriorityQueue()
        frontier.put(start, 0, node_id)  # node id if costs are equal
        # define action set for planning vehicles
        action_dict_car = {'const_acc0': (actions.const_acceleration, 0),
                           'const_acc_p14': (actions.const_acceleration, 1.4),
                           'const_acc_n2': (actions.const_acceleration, -2.0),
                           'coast': (actions.coast, -0.5),
                           'IIDM_acc': (actions.IIDM, 0),
                           'LCL': (actions.lane_change, (-1, "IIDM")),
                           'LCR': (actions.lane_change, (1, "IIDM"))}
        action_dict_truck = {'const_acc0': (actions.const_acceleration, 0),
                             'const_acc_p0.7': (actions.const_acceleration, 0.7),
                             'const_acc_n2': (actions.const_acceleration, -2),
                             'coast': (actions.coast, -0.5),
                             'IIDM_acc': (actions.IIDM, 0),
                             'LCL': (actions.lane_change, (-1, "IIDM")),
                             'LCR': (actions.lane_change, (1, "IIDM"))}
        # define action set of prediction vehicles
        prediction_dict_car = {'IIDM_acc': (actions.IIDM, 0)}
        prediction_dict_truck = {'IIDM_acc': (actions.IIDM, 0)}
        # find max acceleration of cars
        for action in action_dict_car:
            if action_dict_car[action][0] == actions.const_acceleration:
                start.max_a["car"] = max(start.max_a["car"], action_dict_car[action][1])
            elif action_dict_car[action][0] == actions.IIDM:
                for car in range(0, len(start.vehicle_id)):
                    if start.length[int(car)] < 10:
                        start.max_a["car"] = max(start.max_a["car"], start.IDM_param[int(car)].a)
        # find max acceleration of trucks
        for action in action_dict_truck:
            if action_dict_truck[action][0] == actions.const_acceleration:
                start.max_a["truck"] = max(start.max_a["truck"], action_dict_truck[action][1])
            elif action_dict_truck[action][0] == actions.IIDM:
                for truck in range(0, len(start.vehicle_id)):
                    if start.length[int(truck)] > 10:
                        start.max_a["truck"] = max(start.max_a["truck"], start.IDM_param[int(truck)].a)

        # start pool for multiprocessing and wrap node expansion
        if use_multiprocessing is True:
            pool = Pool()  # start computing pool
        else:
            pool = False

        # iteration loop of A*search
        while not frontier.empty():
            current_time = time.time() - starting_time
            current = frontier.get()

            if current.level == planning_horizon:  # best solution found
                # behaviourlist contains planned behaviour of all vehicles in planning step size
                behaviourlist = self.reconstruct_behaviour(current, start)
                print('Node ' + str(node_id) + ' calculation time ' + '{:.2f}'.format(current_time) + ' depth ' + str(
                    current_depth))
                break
            elif current_time > timeout:  # delete frontier (priority queue and start from best node on top level
                # delete frontier
                frontier = PriorityQueue()
                # initialize new stat node
                new_start_node = top_node.parent
                # put best node on top level in queue
                frontier.put(new_start_node, new_start_node.priority, new_start_node.id)
                # extend timeout
                timeout += timeout_interval
                print("Timeout interval exceeded. Fixed top node.")

            children = current.get_children(roadway, action_dict_car, action_dict_truck, prediction_dict_car,
                                            prediction_dict_truck, planning_time_step, planning_horizon, pool)
            for child in children:
                node_id += 1
                child.id = node_id
                child.parent = current
                # for optimal solution, only cost_so_far at level of planning_horizon is relevant
                if child.level == planning_horizon:
                    frontier.put(child, child.cost_so_far, child.id)  # node number if costs are equal
                else:
                    frontier.put(child, child.priority, child.id)  # node number if costs are equal
                current_depth = max(current_depth, child.level)
                if current_depth > previous_depth:
                    top_node = child
                    print(
                        'Node ' + str(node_id) + ' calculation time ' + '{:.2f}'.format(current_time) + ' depth ' + str(
                            current_depth))
                previous_depth = current_depth
                # new best node on current_depth
                if child.level == current_depth and child.priority < top_node.priority:
                    top_node = child

        # close computing pool
        if use_multiprocessing is True:
            pool.close()
            pool.join()
        return behaviourlist

    def reconstruct_behaviour(self, current, start):
        """
        Reads actions from nodes in correct order (A-Star search delivers nodes in wrong order).

        :param current: Last node of planning process (goal node).
        :param start: Root node of planning process.
        :return: List of behaviors
        """
        behaviourlist = []  # contains list of all vehicles behaviours of all time steps
        while current != start:
            behaviour = []  # contains list of all vehicles behaviours of one time step
            for vehicle in range(0, len(current.acceleration)):
                if (current.isChangingLane[vehicle]) == 1 and (current.parent.isChangingLane[vehicle] == 0):
                    lc = True
                    if current.lane[vehicle] < current.destinationLane[vehicle]:
                        direction = Direction.right
                    else:
                        direction = Direction.left
                else:
                    lc = False
                    direction = None
                behaviour.append(Behaviour(vehicle_id=current.vehicle_id[vehicle], acc=current.acceleration[vehicle],
                                           lane_change=lc, direction=direction))
            behaviourlist.append(behaviour)
            current = current.parent
        behaviourlist.reverse()
        return behaviourlist
