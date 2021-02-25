from lib.IDriverModel import IDriverModel
from DriverModels.A_star.PriorityQueue import *
from DriverModels.A_star import actions
from DriverModels.A_star.Node import Node
import numpy as np
from lib.Behaviour import Behaviour
from SimulationEnvironment.Setting import GD
from lib.Direction import Direction
import common.user_settings as cfg
import time
from multiprocessing import Pool
from DriverModels.A_star.ParameterEstimator import ParameterEstimator
from common.Visualize_Scenario import plot_scenario_from_node


class DecentralizedAStarSearch(IDriverModel):
    """
    Created by:
    Christian Knies

    Description:
    Class for decentral cooperative behavior planning with A-Star search.
    """

    def __init__(self, simulation_time_step, roadway_model):
        self.simulation_time_step = round(simulation_time_step, 2)
        self.roadway = roadway_model
        self.parameter_provider = ParameterEstimator(self.roadway)

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

        return DecentralizedAStarSearch(simulation_time_step, roadway_model)

    def create_start_node(self, car_object):
        """
        Create root node for A-Star search.

        :param car_object: Vehicle object for which behavior shall be planned.
        :return: Root node.
        """

        vehicle_data = self.parameter_provider.get_parameter(car_object)
        start_node = Node(self.roadway, vehicle_data.id.to_numpy(), np.array([car_object.get_id()]),
                          vehicle_data.length.to_numpy(), vehicle_data.lc_duration.to_numpy(), None,
                          vehicle_data.position.to_numpy(), vehicle_data.velocity.to_numpy(),
                          vehicle_data.acceleration.to_numpy(), vehicle_data.lane.to_numpy(),
                          vehicle_data.is_changing_lane.to_numpy(), vehicle_data.destination_lane.to_numpy(),
                          vehicle_data.lc_timer.to_numpy(), vehicle_data.IDM_param.to_numpy(),
                          {"car": 0, "truck": 0})
        return start_node

    def run(self, queue, assignedVehicles, all_vehicles, total_time, queue_reference, loop_idx, time_out=None):
        """
        Start decentral A-Star planning process

        :param queue: Queue of behaviors to fill
        :param assignedVehicles: List of vehicles assigned to this behavior model.
        :param all_vehicles: List of all vehicles in Simulation.
        :param total_time: Time step of simulation.
        :param queue_reference: Reference queue as input for trajectory planning (not implemented).
        :param loop_idx: Loop index of simulation environment.
        :param time_out: External control for time_out of planning process (not implemented).
        :return:
        """
        # parameters
        planning_time_step = 2.5  # [s]
        planning_horizon = min(5, np.ceil((cfg.sim_time - (
                loop_idx * self.simulation_time_step)) / planning_time_step))  # int(cfg.sim_time / planning_time_step)  # in time steps
        replanning_time_step = 2.5  # [s]

        # update parameter_provider
        self.parameter_provider.update(all_vehicles, replanning_time_step)

        # calculate behaviour for all assigned vehicles
        behaviorlist = list()  # lists behavior of all vehicles
        for vehicle in assignedVehicles:
            # construct start node
            start = self.create_start_node(vehicle)
            # calculate behavior
            behaviorlist.append(
                self.decentralized_a_star_search(start, planning_horizon, planning_time_step, self.roadway))
        # reformat behaviorlist to list of time steps from list of vehicles
        behaviorlist = list(map(list, zip(*behaviorlist)))

        # extend behavior for current time step to length of queue (replanning_time_step/simulation_time_step)
        replanning_behaviorlist = list()
        queuelength = int(planning_time_step / self.simulation_time_step)  # number of behaviors of one planning step
        while behaviorlist:  # as long as behaviourlist is not empty
            current_behaviour = behaviorlist.pop(0)
            for i in range(0, queuelength):
                replanning_behaviorlist.append(current_behaviour)
        # fill queue with replanning_behaviorlist
        for n in range(0, int(replanning_time_step / self.simulation_time_step)):
            queue.put(replanning_behaviorlist.pop(0))
        # Send sentinel when finished
        queue.put(None)
        # return empty behaviours
        return [], loop_idx

    def decentralized_a_star_search(self, start, planning_horizon, planning_time_step, roadway):
        """
        A-Star planning process

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
        timeout_interval = 5  # stop search after x [s]
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
                           'LCR': (actions.lane_change, (1, "IIDM"))
                           }
        action_dict_truck = {'const_acc0': (actions.const_acceleration, 0),
                             'const_acc_p0.7': (actions.const_acceleration, 0.7),
                             'const_acc_n2': (actions.const_acceleration, -2.0),
                             'coast': (actions.coast, -0.5),
                             'IIDM_acc': (actions.IIDM, 0),
                             'LCL': (actions.lane_change, (-1, "IIDM")),
                             'LCR': (actions.lane_change, (1, "IIDM"))
                             }
        # define action set of prediction vehicles
        prediction_dict_car = {'IIDM_acc': (actions.IIDM, 1)}
        prediction_dict_truck = {'IIDM_acc': (actions.IIDM, 1)}
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
                    if start.length[int(truck)] >= 10:
                        start.max_a["truck"] = max(start.max_a["truck"], start.IDM_param[int(truck)].a)

        # start pool for multiprocessing and wrap node expansion
        if use_multiprocessing is True:
            pool = Pool()  # start computing pool
        else:
            pool = False

        # iteration loop of A*search
        print("Vehicle ID: " + str(start.planning_car_id))
        while not frontier.empty():
            current_time = time.time() - starting_time
            current = frontier.get()

            if current.level == planning_horizon:  # best solution found
                # behaviourlist contains planned behaviour of all vehicles in planning step size
                behaviourlist = self.reconstruct_decentralized_behaviour(current, start)
                print(
                    'Node ' + str(node_id) + ' calculation time ' + '{:.2f}'.format(current_time) + ' depth ' + str(
                        current_depth))
                # plot current plan for debugging
                # if start.planning_car_id == 1:
                #     print(str(current.priority))
                #     plot_scenario_from_node(current, planning_time_step)
                break
            elif current_time > timeout:  # fix top node and fill rest rest with full break
                # reconstruct behaviourlist till top node
                behaviourlist_short = self.reconstruct_decentralized_behaviour(top_node, start)
                behaviourlist = self.overtime_behaviour(behaviourlist_short, planning_horizon, top_node)
                print('Node ' + str(node_id) + ' calculation time ' + '{:.2f}'.format(current_time) + ' depth ' + str(
                    top_node.level))
                print("Timeout interval exceeded. Fixed top node.")
                # plot current plan for debugging
                # plot_scenario_from_node(top_node, planning_time_step)
                break

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
                previous_depth = current_depth
                # new best node on current_depth
                if child.level == current_depth and child.priority < top_node.priority:
                    top_node = child
                # plot current plan for debugging
                    # if start.planning_car_id == 2 and child.level >= 4:
                    #     print(str(child.priority))
                    #     plot_scenario_from_node(top_node, planning_time_step)
                    #     print("test")
                # if start.planning_car_id == 2 and child.level >= 0:
                #     print(str(child.priority))
                #     plot_scenario_from_node(child, planning_time_step)

        # close computing pool
        if use_multiprocessing is True:
            pool.close()
            pool.join()
        return behaviourlist

    def reconstruct_decentralized_behaviour(self, current, start):
        """
        Reads actions from nodes in correct order (A-Star search delivers nodes in wrong order).

        :param current: Last node of planning process (goal node).
        :param start: Root node of planning process.
        :return: List of behaviors
        """
        behaviourlist = []  # contains list of all vehicles behaviours of all time steps
        vehicle = np.where(current.planning_car_id == current.vehicle_id)[0][0]
        while current != start:
            if (current.isChangingLane[vehicle]) == 1 and (current.parent.isChangingLane[vehicle] == 0):
                lc = True
                if current.lane[vehicle] < current.destinationLane[vehicle]:
                    direction = Direction.right
                else:
                    direction = Direction.left
            else:
                lc = False
                direction = None
            behaviourlist.append(Behaviour(vehicle_id=current.vehicle_id[vehicle], acc=current.acceleration[vehicle],
                                           lane_change=lc, direction=direction))
            current = current.parent
        behaviourlist.reverse()
        return behaviourlist

    def overtime_behaviour(self, behaviour, planning_horizon, node):
        """
        If planning process is aborted because of time out and behavior list is shorter than planning horizon
        (number of behavior values required by simulation environment), perform overtime behavior
        (max break no lane change).

        :param behaviour: Short (timed out) behavior list
        :param planning_horizon: Planning horizon in time steps.
        :param node: Current planning node.
        :return: Behavior list.
        """
        while len(behaviour) < planning_horizon:
            behaviour.append(
                Behaviour(vehicle_id=node.planning_car_id, acc=GD.max_break, lane_change=False, direction=None))
        return behaviour
