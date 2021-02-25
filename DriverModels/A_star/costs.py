import numpy as np
import math
from SimulationEnvironment.Setting import GD


def cost_evaluator(node, dt, planning_horizon):
    """
    Created by:
    Christian Knies

    Description:
    Evaluate cooperation costs overall and sets costs_so_far (current node and all parent nodes), heuristic and
    priority (costs so far + heuristic) in node class.

    :param node: Current planning node.
    :param dt: Planning time step.
    :param planning_horizon: Overall Planning horizon.
    :return: Cost value of current node.
    """
    # Parameters
    w_safety = 9e100
    w_roadway = 1e100
    w_energy = 1.7e-6
    w_time = 1
    w_lateral = 0.1
    w_right = 9e10
    w_speed_limit = 9e10
    t_thresh_safety = 0.5
    t_thresh_right = 0.5
    alpha = 0
    num_vehicles = len(node.position)
    cost_safety = 0
    cost_right_overtaking = 0
    cost_energy = 0
    cost_time = 0
    cost_lateral = 0
    cost_heuristic = 0
    cost_roadway = 0
    cost_speed_limit = 0
    for vehicle in range(0, num_vehicles):
        # decide if symmetric or asymmetric cost function
        if node.vehicle_id[vehicle] not in node.planning_car_id and len(node.planning_car_id) == 1:
            # prediction vehicle -> asymmetric cost function
            p_costs = 1.0  # 0.8
        else:
            # this is the only planning vehicle or more than one planning vehicle -> symmetric cost function
            p_costs = 1
            # check hard constraints only if vehicle is planning vehicle, because the behavior of prediction vehicles can`t be influenced (Exception lane changes: already checked in Node class)
            # calculate leader variables and desired velocity
            d_le, v_le, d_left_le, v_left_le, d_le_pre, v_le_pre = calc_auxiliaries(node, vehicle)
            # costs for traffic rule violation
            cost_roadway += node.roadway.get_off_track(node.position[vehicle], node.lane[vehicle])
            # consider states between current and previous time step if vehicle ended lane change
            if not all(node.lane == node.parent.lane):
                # vehicle ended lane change
                cost_safety += max(evaluate_safety(d_le, v_le, node.velocity[vehicle], t_thresh_safety),
                                   evaluate_safety(d_le_pre, v_le_pre, node.velocity[vehicle], t_thresh_safety))
            else:
                cost_safety += evaluate_safety(d_le, v_le, node.velocity[vehicle], t_thresh_safety)
            # right overtaking prohibition
            if v_left_le < node.velocity[vehicle]:
                cost_right_overtaking += evaluate_safety(d_left_le, v_left_le, node.velocity[vehicle], t_thresh_right)
            else:
                cost_right_overtaking += 0
            # obey truck maximum velocity of 25 m/s or desired velocity if > 25 m/s
            if (node.length[vehicle] >= 10) and (node.velocity[vehicle] > max(25, node.IDM_param[vehicle].V_0 + 0.1)):
                cost_speed_limit += 1
        # costs for maneuver planning: consider prediction vehicles with p value
        cost_energy += p_costs * evaluate_brake_energy(node, node.acceleration[vehicle], node.velocity[vehicle], alpha,
                                                       dt, vehicle)
        cost_time += p_costs * evaluate_time_loss(node.velocity[vehicle], node.IDM_param[vehicle].V_0, dt)
        cost_heuristic += p_costs * heuristic(node.velocity[vehicle], node.IDM_param[vehicle].V_0, node, vehicle)
        cost_lateral += p_costs * evaluate_lane_changes(node.isChangingLane[vehicle],
                                                        node.parent.isChangingLane[vehicle])
    # combined cooperation metric
    cost_cooperation = w_roadway * cost_roadway + w_safety * cost_safety + w_right * cost_right_overtaking + w_energy * cost_energy + w_time * cost_time + w_lateral * cost_lateral + w_speed_limit * cost_speed_limit
    # set costs in node class
    node.cost_so_far = cost_cooperation + node.parent.cost_so_far
    node.cost_heuristic = cost_heuristic
    node.priority = node.cost_so_far + node.cost_heuristic
    return cost_cooperation


def calc_auxiliaries(node, vehicle):
    """
    Calculate distance and velocity to leader car in this and previous time step.

    :param node: Current node.
    :param vehicle: Vehicle number.
    :return: Distances and velocities to leader car in this and previous time step
    """
    v_crit = 60 / 3.6  # right overtaking prohibition only over 60 km/h

    # calculate d_le and v_le for leader car in lane or in lane change to or from lane
    lane = node.lane[vehicle]
    leaders_in_lane = np.where((node.position > node.position[vehicle]) & (math.floor(lane) - 0.5 <= node.lane) & (
            node.lane <= math.ceil(lane) + 0.5))
    if len(leaders_in_lane[0]) == 0:  # no leader car
        d_le = 1000
        v_le = 1000
    else:
        leader_car = np.argmin(node.position[leaders_in_lane])
        d_le = max(node.position[leaders_in_lane][leader_car] - node.position[vehicle] - node.length[vehicle], 0)
        v_le = node.velocity[leaders_in_lane][leader_car]

    # calculate d_le and v_le between current and previous time step if a lane change ended in between
    lane_pre = node.parent.lane[vehicle]
    leaders_in_lane_pre = np.where(
        (node.position > node.position[vehicle]) & (math.floor(lane_pre) - 0.5 <= node.parent.lane) & (
                node.parent.lane <= math.ceil(lane_pre) + 0.5))
    if not all(node.lane == node.parent.lane) and len(leaders_in_lane_pre[0]) != 0:
        # lane change ended between current and previous time step
        leader_car_pre = np.argmin(node.position[leaders_in_lane_pre])
        d_le_pre = max(
            node.position[leaders_in_lane_pre][leader_car_pre] - node.position[vehicle] - node.length[vehicle], 0)
        v_le_pre = node.velocity[leaders_in_lane_pre][leader_car_pre]
    else:
        d_le_pre = 1000
        v_le_pre = 1000

    # calculate d and v for leader cars on left lanes for right overtaking prohibition
    leaders_left = np.where((node.position > node.position[vehicle]) & (node.lane < lane))
    if (len(leaders_left[0]) == 0) | (node.velocity[vehicle] < v_crit):  # no leader car or v < v_crit
        d_left_le = 1000
        v_left_le = 1000
    else:
        left_leader_car = np.argmin(node.position[leaders_left])
        d_left_le = max(node.position[leaders_left][left_leader_car] - node.position[vehicle] - node.length[vehicle], 0)
        v_left_le = node.velocity[leaders_left][left_leader_car]
    return d_le, v_le, d_left_le, v_left_le, d_le_pre, v_le_pre


def heuristic(v, v0, node, vehicle):
    """
    Calculates heuristic (min future costs) based on current and desired velocity considering max possible acceleration.

    :param v: List of velocities of all vehicles.
    :param v0: List of desired velocities all vehicles.
    :param node: Current node.
    :param vehicle: Number of vehicle for which to compute the heuristic.
    :return: Minimum future costs.
    """
    # heuristic dependent from velocity (greater/smaller than v0)
    if v0 > v:
        if node.length[vehicle] < 10:  # vehicle is a car
            max_a = node.max_a["car"]  # maximum possible acceleration
        else:  # vehicle is a truck
            max_a = node.max_a["truck"]  # maximum possible acceleration
        # time until v0 is reached with max acceleration
        t_max = (v0 - v) / max_a
        # time costs for maximum acceleration towards v0
        min_future_costs = (1 - (v / v0)) * t_max - (0.5 * max_a / v0) * (t_max ** 2)
    else:
        max_a = GD.max_break  # maximum deceleration
        # time until v0 is reached with min acceleration
        t_max = (v0 - v) / max_a
        # time costs for maximum deceleration towards v0
        min_future_costs = ((v / v0) - 1) * t_max + (0.5 * max_a / v0) * (t_max ** 2)
    return min_future_costs


def evaluate_reaction_time(d_le, v_le, v_fo):
    """
    Calculate necessary reaction time in current state.

    :param d_le: Distance to leader.
    :param v_le: Velocity of leader.
    :param v_fo: Velocity of ego vehicle.
    :return: Necessary reaction time.
    """
    # Parameter
    d_br = 2  # m Restbremsabstand nach Bremsung
    max_deceleration = 7.5  # m/s^2 maximale Verzögerung
    if v_fo <= 0:
        v_fo = 0.1
    # Calculation
    # t_re = 1/max(v_fo, 0.1) * (d_le - d_br + (0.5*np.power(v_le, 2)/a_max) - (0.5*np.power(v_fo, 2)/a_max))
    t_re = 1 / v_fo * (d_le - d_br + (0.5 * (v_le ** 2) / max_deceleration) - (0.5 * (v_fo ** 2) / max_deceleration))
    return t_re


def evaluate_safety(d_le, v_le, v_fo, t_thresh):
    """
    Evaluate safety based on reaction time and threshold.

    :param d_le: Distance of leader vehicle.
    :param v_le: Velocity of leader vehicle.
    :param v_fo: Velocity of ego vehicle.
    :param t_thresh: Threshold for reaction time under which a scenario is classified as unsafe.
    :return: Safety evaluation (bool: 1: unsafe, 0: safe)
    """
    # calculate reaction time
    t_re = evaluate_reaction_time(d_le, v_le, v_fo)
    # decide on safety
    if t_re >= t_thresh:
        J_safety = 0
    else:
        J_safety = 1 + t_thresh - t_re
    return J_safety


def evaluate_brake_energy(node, a, v, alpha, dt, vehicle):
    """
    Evaluate brake energy in current time step.

    :param node: Current planning node.
    :param a: Vehicle acceleration.
    :param v: Vehicle velocity.
    :param alpha: Inclination of road in rad.
    :param dt: Planning time step.
    :param vehicle: Number of vehicle.
    :return: Energy converted into heat by the brakes.
    """
    # Parameter
    if node.length[vehicle] < 10:  # vehicle is a car
        m = 1545  # vehicle mass [kg]
        g = 9.81  # gravitation constant [m/s^2]
        fr = 0.01  # rolling friction coefficient
        rho = 1.2  # air density [kg/m^3]
        A = 2.25  # vehicle front surface [m^2]
        cw = 0.3  # cw value
    else:  # vehicle is a truck
        m = 24400  # vehicle mass [kg]
        g = 9.81  # gravitation constant [m/s^2]
        fr = 0.008  # rolling friction coefficient
        rho = 1.2  # air density [kg/m^3]
        A = 8  # vehicle front surface [m^2]
        cw = 0.65  # cw value
    # coasting acceleration
    # a_coast = -(m*g*np.sin(alpha) + m*g*np.cos(alpha)*fr + 0.5*rho*A*cw*np.power(v, 2))/m
    a_coast = -(m * g * math.sin(alpha) + m * g * math.cos(alpha) * fr + 0.5 * rho * A * cw * (v ** 2)) / m
    # braking force must be positive, otherwise no braking
    F_br = m * max(a_coast - a, 0)
    J_energy_brake = F_br * v * dt
    return J_energy_brake


def evaluate_time_loss(v, v0, dt):
    """
    Evaluate time loss based on current and desired velocity.

    :param v: Current velocity.
    :param v0: Desired velocity.
    :param dt: Time step.
    :return: Time loss in seconds.
    """
    relative_time_loss = abs((v0 - v) / v0)
    absolute_time_loss = relative_time_loss * dt
    return absolute_time_loss


def evaluate_lane_changes(is_changing_lane, is_changing_lane_pre):
    """
    Evaluate of vehicle started lane change between last and current time step.

    :param is_changing_lane: Is vehicle changing lanes in current time step (1: yes, 0: no).
    :param is_changing_lane_pre: Was vehicle in lane change in last time step (1: yes, 0: no).
    :return: Did vehicle start lane change (0: no, 1: yes).
    """
    return int((is_changing_lane - is_changing_lane_pre) > 0)


if __name__ == "__main__":
    # Validation of heuristic
    a_max = 1.5
    v = 25
    v0 = 40
    t = 0
    dt = 0.1
    time_loss = 0
    v_loop = v
    while v_loop <= v0:
        relative_time_loss = abs((v0 - v_loop) / v0)
        absolute_time_loss = relative_time_loss * dt
        time_loss += absolute_time_loss
        v_loop = v_loop + a_max * dt
        t += dt
    print(time_loss)

    # define dummy class vehicle
    class dummy_node_class:

        def __init__(self, a_max):
            self.length = []
            self.length.append(3)
            self.max_a = {"car": a_max, "truck": a_max}


    dummy_node = dummy_node_class(a_max)

    min_future_costs = heuristic(v, v0, dummy_node, 0)
    print(min_future_costs)
