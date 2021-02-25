import numpy as np
from SimulationEnvironment.Setting import GD
import math
from common.Visualize_Scenario import plot_scenario_from_node

"""
Created by:
Christian Knies

Description:
File comprises all actions the vehicles can perform. In- and Outputs are always the same.

:param node: Current A-Star planning node.
:param dt: Time step/length of the action to perform.
:param param: Parameter, individual meaning for each action.
:param vehicle: Number of vehicle, which performs the action. 
:return: Values for next time step of position, velocity, acceleration, lane, isChangingLane, destinationLane and 
lc_timer (how long will lane change will take from current time). 
"""


def const_acceleration(node, dt, param, vehicle):
    """
    Perform constant acceleration over time step dt.
    """
    new_acceleration = param
    new_velocity = node.velocity[vehicle] + new_acceleration * dt
    new_position = node.position[vehicle] + node.velocity[vehicle] * dt + new_acceleration * 0.5 * (dt ** 2)
    # increase lane change timer
    new_destinationLane, new_isChangingLane, new_lane, new_lc_timer = progress_lane_change(node, dt, vehicle)
    return new_position, new_velocity, new_acceleration, new_lane, new_isChangingLane, new_destinationLane, new_lc_timer


def lane_change(node, dt, param, vehicle):
    """
    Start lane change. Acceleration either constant velocity (param[1] == "const") or IIDM (param[1] == "IIDM")
    """
    sim_steps = 3
    v_crit = 60 / 3.6
    # lateral
    new_lc_timer = node.lc_duration[vehicle] - dt  # in new node one dt has already passed
    new_destinationLane = node.lane[vehicle] + param[0]  # param +1/-1 for lane change right/left
    new_lane = (node.lane[vehicle] + new_destinationLane) / 2
    new_isChangingLane = 1
    # longitudinal
    if param[1] == "const":
        new_acceleration = 0  # node.acceleration[vehicle]   # use old acceleration
        new_velocity = node.velocity[vehicle] + new_acceleration * dt
        new_position = node.position[vehicle] + node.velocity[vehicle] * dt + new_acceleration * 0.5 * (dt ** 2)
    else:
        # longitudinal movement: find minimal acceleration of both involved lanes
        start_lane = node.lane[vehicle]
        dest_lane = new_destinationLane
        v_crit = 1000
        # consider acceleration on start_lane only if start lane is on track
        if node.roadway.get_off_track(node.position[vehicle], start_lane) == 1:
            acc_start_lane = 1000
        else:
            acc_start_lane = get_relevant_leader_acceleration(node, vehicle, start_lane, dt, v_crit)
        acc_dest_lane = get_relevant_leader_acceleration(node, vehicle, dest_lane, dt, v_crit)
        new_acceleration = max(min(acc_start_lane, acc_dest_lane), 0)
        new_position, new_velocity, new_acceleration = smooth_acceleration(node, vehicle, new_acceleration,
                                                                           new_isChangingLane, new_lane,
                                                                           new_destinationLane, sim_steps, v_crit, dt)
    return new_position, new_velocity, new_acceleration, new_lane, new_isChangingLane, new_destinationLane, new_lc_timer


def coast(node, dt, param, vehicle):
    """
    Perform coasting action. Vehicle decelerates according to driving resistance forces.
    """
    # Parameter
    if node.length[vehicle] < 10:  # vehicle is car
        m = 1545  # vehicle mass [kg]
        g = 9.81  # gravitation constant [m/s^2]
        fr = 0.01  # rolling friction coefficient
        rho = 1.2  # air density [kg/m^3]
        A = 2.25  # vehicle front surface [m^2]
        cw = 0.3  # cw value
        alpha = 0  # road inclination
    else:  # vehicle is truck
        m = 24400  # vehicle mass [kg]
        g = 9.81  # gravitation constant [m/s^2]
        fr = 0.008  # rolling friction coefficient
        rho = 1.2  # air density [kg/m^3]
        A = 8.0  # vehicle front surface [m^2]
        cw = 0.65  # cw value
        alpha = 0  # road inclination

    # increase lane change timer
    new_destinationLane, new_isChangingLane, new_lane, new_lc_timer = progress_lane_change(node, dt, vehicle)
    # calculate new acceleration, velocity and position
    new_acceleration = min(-(m * g * math.sin(alpha) + m * g * math.cos(alpha) * fr + 0.5 * rho * A * cw * (
            node.velocity[vehicle] ** 2)) / m, param)
    new_velocity = node.velocity[vehicle] + new_acceleration * dt
    new_position = node.position[vehicle] + node.velocity[vehicle] * dt + new_acceleration * 0.5 * (dt ** 2)
    return new_position, new_velocity, new_acceleration, new_lane, new_isChangingLane, new_destinationLane, new_lc_timer


def aim_v0(node, dt, param, vehicle):
    """
    Choose acceleration to reach desired velocity v0 in next time step.
    """
    # increase lane change timer
    new_destinationLane, new_isChangingLane, new_lane, new_lc_timer = progress_lane_change(node, dt, vehicle)
    new_acceleration = (node.IDM_param[vehicle].V_0 - node.velocity[vehicle]) / dt
    new_velocity = node.velocity[vehicle] + new_acceleration * dt
    new_position = node.position[vehicle] + node.velocity[vehicle] * dt + new_acceleration * 0.5 * (dt ** 2)
    return new_position, new_velocity, new_acceleration, new_lane, new_isChangingLane, new_destinationLane, new_lc_timer


def IIDM(node, dt, param, vehicle):
    """
    Perform IIDM acceleration over time step dt. If vehicle is in lane change, consider both relevant lanes. If
    vehicle is not in lane change, the lane change decision model MOBIL can be used (param == 1) or the acceleration
    can be computed without lane changes (param == 0).
    """
    # settings
    v_crit = 60 / 3.6  # right overtaking prohibition only over 60 km/h
    changing_threshold_left = 0.23  # 1.0
    changing_threshold_right = 1.05  # 0.4
    bias_for_right_lane = 0.0
    b_safe_follower = -2
    sim_steps = 3

    # overwrite parameter from file
    if GD.ext_params is not None:
        if "lc_thresh_left" in GD.ext_params.keys():
            changing_threshold_left = GD.ext_params["lc_thresh_left"]
        if "lc_thresh_right" in GD.ext_params.keys():
            changing_threshold_right = GD.ext_params["lc_thresh_right"]
        if "lc_b_safe" in GD.ext_params.keys():
            b_safe_follower = GD.ext_params["lc_b_safe"]
        if "sim_steps" in GD.ext_params.keys():
            sim_steps = int(round(GD.ext_params["sim_steps"]))

    # differentiation between lane change and no lane change
    if node.isChangingLane[vehicle] == 1:  # if vehicle is in lane change
        # longitudinal movement: find minimal acceleration of both involved lanes
        start_lane = (2 * node.lane[vehicle]) - node.destinationLane[vehicle]
        dest_lane = node.destinationLane[vehicle]
        # consider acceleration on start_lane only if start lane is on track
        if node.roadway.get_off_track(node.position[vehicle], start_lane) == 1:
            acc_start_lane = 1000
        else:
            acc_start_lane = get_relevant_leader_acceleration(node, vehicle, start_lane, dt, v_crit)
        acc_dest_lane = get_relevant_leader_acceleration(node, vehicle, dest_lane, dt, v_crit)
        new_acceleration = min(acc_start_lane, acc_dest_lane)
        # lateral movement
        new_lc_timer = max(node.lc_timer[vehicle] - dt, 0)
        if new_lc_timer == 0:  # if lane change is finished in current time step
            new_lane = node.destinationLane[vehicle]
            new_destinationLane = None
            new_isChangingLane = 0
        else:  # vehicle is still in lane change
            new_lane = node.lane[vehicle]
            new_destinationLane = node.destinationLane[vehicle]
            new_isChangingLane = node.isChangingLane[vehicle]
    else:  # currently no lane change
        if param == 0:  # IIDM action without lane changes
            # longitudinal movement in current lane
            new_acceleration = get_relevant_leader_acceleration(node, vehicle, node.lane[vehicle], dt, v_crit)
            new_lc_timer = node.lc_timer[vehicle]
            new_lane = node.lane[vehicle]
            new_destinationLane = node.destinationLane[vehicle]
            new_isChangingLane = node.isChangingLane[vehicle]
        else:  # IIDM action with MOBIL lane changes
            # structure environment
            leaders_in_lane = np.where(
                (node.position > node.position[vehicle]) & (node.lane[vehicle] - 0.5 <= node.lane) & (
                        node.lane <= node.lane[vehicle] + 0.5))
            followers_in_lane = np.where(
                (node.position < node.position[vehicle]) & (node.lane[vehicle] - 0.5 <= node.lane) & (
                        node.lane <= node.lane[vehicle] + 0.5))
            leaders_left = np.where(
                (node.position > node.position[vehicle]) & (node.lane[vehicle] - 1.5 <= node.lane) & (
                        node.lane <= node.lane[vehicle] - 0.5))
            followers_left = np.where(
                (node.position < node.position[vehicle]) & (node.lane[vehicle] - 1.5 <= node.lane) & (
                        node.lane <= node.lane[vehicle] - 0.5))
            leaders_right = np.where(
                (node.position > node.position[vehicle]) & (node.lane[vehicle] + 0.5 <= node.lane) & (
                        node.lane <= node.lane[vehicle] + 1.5))
            followers_right = np.where(
                (node.position < node.position[vehicle]) & (node.lane[vehicle] + 0.5 <= node.lane) & (
                        node.lane <= node.lane[vehicle] + 1.5))
            # define roles and calculate accelerations
            if len(leaders_left[0]) == 0:  # no left leader car
                left_leader = -1
                x_left_leader = node.position[vehicle] + 1000
                v_left_leader = 1000
            else:
                left_leader_in_group = np.argmin(node.position[leaders_left])
                x_left_leader = node.position[leaders_left][left_leader_in_group]
                v_left_leader = node.velocity[leaders_left][left_leader_in_group]
                left_leader = np.where(node.vehicle_id[leaders_left][left_leader_in_group] == node.vehicle_id)[0][0]
            if len(followers_left[0]) == 0:  # no left follower car
                left_follower = -1
                x_left_follower = node.position[vehicle] - 1000
                v_left_follower = 0
            else:
                left_follower_in_group = np.argmax(node.position[followers_left])
                x_left_follower = node.position[followers_left][left_follower_in_group]
                v_left_follower = node.velocity[followers_left][left_follower_in_group]
                left_follower = np.where(node.vehicle_id[followers_left][left_follower_in_group] == node.vehicle_id)[0][
                    0]
            if len(leaders_in_lane[0]) == 0:  # no center leader car
                center_leader = -1
                x_center_leader = node.position[vehicle] + 1000
                v_center_leader = 1000
            else:
                center_leader_in_group = np.argmin(node.position[leaders_in_lane])
                x_center_leader = node.position[leaders_in_lane][center_leader_in_group]
                v_center_leader = node.velocity[leaders_in_lane][center_leader_in_group]
                center_leader = \
                    np.where(node.vehicle_id[leaders_in_lane][center_leader_in_group] == node.vehicle_id)[0][0]
            if len(followers_in_lane[0]) == 0:  # no follower in lane
                center_follower = -1
                x_center_follower = node.position[vehicle] - 1000
                v_center_follower = 0
            else:
                center_follower_in_group = np.argmax(node.position[followers_in_lane])
                x_center_follower = node.position[followers_in_lane][center_follower_in_group]
                v_center_follower = node.velocity[followers_in_lane][center_follower_in_group]
                center_follower = \
                    np.where(node.vehicle_id[followers_in_lane][center_follower_in_group] == node.vehicle_id)[0][0]
            if len(leaders_right[0]) == 0:  # no right leader
                right_leader = -1
                x_right_leader = node.position[vehicle] + 1000
                v_right_leader = 1000
            else:
                right_leader_in_group = np.argmin(node.position[leaders_right])
                x_right_leader = node.position[leaders_right][right_leader_in_group]
                v_right_leader = node.velocity[leaders_right][right_leader_in_group]
                right_leader = np.where(node.vehicle_id[leaders_right][right_leader_in_group] == node.vehicle_id)[0][0]
            if len(followers_right[0]) == 0:  # no right follower
                right_follower = -1
                x_right_follower = node.position[vehicle] - 1000
                v_right_follower = 0
            else:
                right_follower_in_group = np.argmax(node.position[followers_right])
                x_right_follower = node.position[followers_right][right_follower_in_group]
                v_right_follower = node.velocity[followers_right][right_follower_in_group]
                right_follower = \
                    np.where(node.vehicle_id[followers_right][right_follower_in_group] == node.vehicle_id)[0][0]
            # calculate incentives
            incentive_left = 0
            incentive_right = 0
            a_ego_center_leader = calculate_a_IIDM(x_center_leader - node.position[vehicle] - node.length[vehicle],
                                                   v_center_leader, node.velocity[vehicle],
                                                   node.IDM_param[vehicle], dt)
            a_ego_left_leader = calculate_a_IIDM(x_left_leader - node.position[vehicle] - node.length[vehicle],
                                                 v_left_leader, node.velocity[vehicle], node.IDM_param[vehicle],
                                                 dt)
            if node.roadway.get_left_most_lane(node.position[vehicle]) < node.lane[vehicle] and max(
                    node.roadway.get_dist_to_lane_end(node.position[vehicle], node.lane[vehicle] - 1) - node.length[
                        vehicle], 0.1) > 400:
                # left lane exists and is longer than 400 m
                a_left_follower_ego = calculate_a_IIDM(
                    node.position[vehicle] - x_left_follower - node.length[left_follower],
                    node.velocity[vehicle] + min(a_ego_center_leader, a_ego_left_leader) * dt,
                    node.velocity[left_follower], node.IDM_param[left_follower], dt)
                if a_left_follower_ego > b_safe_follower and a_ego_left_leader > GD.b_safe:
                    # safety criterion fulfilled
                    if max(node.roadway.get_dist_to_lane_end(node.position[vehicle], node.lane[vehicle]) - node.length[
                                    vehicle], 0.1) < 400:
                        # current lane ends in less than 400 m -> incentive for lane change left high
                        incentive_left = 5
                    elif a_ego_center_leader > a_ego_left_leader + changing_threshold_left and node.velocity[
                        vehicle] > max(v_crit, v_left_leader):
                        # car is blocked by left overtaking prohibition -> incentive for lane change left high
                        incentive_left = 2
                    else:
                        # normal changing incentive
                        a_left_follower_left_leader = calculate_a_IIDM(
                            x_left_leader - x_left_follower - node.length[left_follower], v_left_leader,
                            node.velocity[left_follower], node.IDM_param[left_follower], dt)
                        incentive_left = a_ego_left_leader - min(a_ego_left_leader, a_ego_center_leader) + \
                                         node.IDM_param[vehicle].p * (a_left_follower_ego - a_left_follower_left_leader)
                        incentive_left -= bias_for_right_lane
            if node.roadway.get_right_most_lane(node.position[vehicle]) > node.lane[vehicle] and max(
                    node.roadway.get_dist_to_lane_end(node.position[vehicle], node.lane[vehicle] + 1) - node.length[
                        vehicle], 0.1) > 400:
                # right lane exists and is longer than 400 m
                a_ego_right_leader = calculate_a_IIDM(x_right_leader - node.position[vehicle] - node.length[vehicle],
                                                      v_right_leader, node.velocity[vehicle],
                                                      node.IDM_param[vehicle], dt)
                a_right_follower_ego = calculate_a_IIDM(
                    node.position[vehicle] - x_right_follower - node.length[right_follower],
                    node.velocity[vehicle] + min(a_ego_center_leader, a_ego_right_leader) * dt,
                    node.velocity[right_follower], node.IDM_param[right_follower], dt)
                if a_right_follower_ego > b_safe_follower and a_ego_right_leader > GD.b_safe:
                    # safety criterion fulfilled
                    if max(node.roadway.get_dist_to_lane_end(node.position[vehicle], node.lane[vehicle]) - node.length[
                                    vehicle], 0.1) < 400:
                        # current lane ends in less than 400 m -> incentive for lane change left high
                        incentive_right = 4
                    elif node.IDM_param[vehicle].V_0 < node.IDM_param[center_follower].V_0:
                        # let center follower pass if incentive big enough
                        a_center_follower_center_leader = calculate_a_IIDM(
                            x_center_leader - x_center_follower - node.length[center_follower], v_center_leader,
                            node.velocity[center_follower],
                            node.IDM_param[center_follower], dt)
                        a_center_follower_left_leader = calculate_a_IIDM(
                            x_left_leader - x_center_follower - node.length[center_follower], v_left_leader,
                            node.velocity[center_follower],
                            node.IDM_param[center_follower], dt)
                        a_center_follower_ego = calculate_a_IIDM(
                            node.position[vehicle] - x_center_follower - node.length[center_follower],
                            node.velocity[vehicle], node.velocity[center_follower],
                            node.IDM_param[center_follower], dt)
                        incentive_right = min(a_ego_right_leader, a_ego_center_leader) - min(a_ego_center_leader,
                                                                                             a_ego_left_leader) + \
                                          node.IDM_param[vehicle].p * (min(a_center_follower_center_leader,
                                                                           a_center_follower_left_leader) - a_center_follower_ego)
                        incentive_right += bias_for_right_lane
            # decide over lane change
            if incentive_right <= changing_threshold_right and incentive_left <= changing_threshold_left:
                # no lane change
                new_acceleration = get_relevant_leader_acceleration(node, vehicle, node.lane[vehicle], dt, v_crit)
                new_lc_timer = node.lc_timer[vehicle]
                new_lane = node.lane[vehicle]
                new_destinationLane = node.destinationLane[vehicle]
                new_isChangingLane = node.isChangingLane[vehicle]
            else:
                if incentive_right > incentive_left:
                    # lane change to right
                    start_lane = node.lane[vehicle]
                    new_destinationLane = start_lane + 1  # param +1/-1 for lane change right/left

                else:
                    # lane change to left
                    start_lane = node.lane[vehicle]
                    new_destinationLane = start_lane - 1  # param +1/-1 for lane change right/left
                # lateral: set parameters for lane change
                new_lc_timer = node.lc_duration[vehicle] - dt  # in new node one dt has already passed
                new_lane = (start_lane + new_destinationLane) / 2
                new_isChangingLane = 1
                # longitudinal movement: find minimal acceleration of both involved lanes
                acc_start_lane = get_relevant_leader_acceleration(node, vehicle, start_lane, dt, v_crit,
                                                                  new_isChangingLane)
                acc_dest_lane = get_relevant_leader_acceleration(node, vehicle, new_destinationLane, dt, v_crit)
                new_acceleration = min(acc_start_lane, acc_dest_lane)
    # calculate inter time steps if sim_steps > 1
    new_position, new_velocity, new_acceleration = smooth_acceleration(node, vehicle, new_acceleration,
                                                                       new_isChangingLane, new_lane,
                                                                       new_destinationLane, sim_steps, v_crit, dt)
    return new_position, new_velocity, new_acceleration, new_lane, new_isChangingLane, new_destinationLane, new_lc_timer


def smooth_acceleration(node, vehicle, new_acceleration, new_isChangingLane, new_lane, new_destinationLane, sim_steps,
                        v_crit, dt):
    """
    Standard time step might be too large for IIDM model. Therefore, the acceleration can be computed in multiple steps
    (withon one time step dt) with this function to smooth the acceleration.
    """
    # calculate inter time steps if sim_steps > 1
    dt_inter = dt / sim_steps
    if sim_steps > 1:
        new_acceleration = calculate_inter_time_step(node, vehicle, new_acceleration, new_isChangingLane, new_lane,
                                                     new_destinationLane, sim_steps, dt_inter, v_crit)
    # check if velocity > 0
    new_velocity = node.velocity[vehicle] + new_acceleration * dt
    if new_velocity < 0:
        new_velocity = 0
        # avoid division by zero
        if abs(new_acceleration) < 0.1:
            new_acceleration = -0.1
        dt = min(abs(node.velocity[vehicle] / new_acceleration), dt)
    new_position = node.position[vehicle] + node.velocity[vehicle] * dt + new_acceleration * 0.5 * (dt ** 2)
    return new_position, new_velocity, new_acceleration


def progress_lane_change(node, dt, vehicle):
    """
    Progress the lane change for actions that are only for longitudinal direction.
    """
    if node.isChangingLane[vehicle] == 1:  # if vehicle is in lane change
        new_lc_timer = max(node.lc_timer[vehicle] - dt, 0)
        if new_lc_timer == 0:  # if lane change is finished in current time step
            new_lane = node.destinationLane[vehicle]
            new_destinationLane = None
            new_isChangingLane = 0
        else:  # vehicle is still in lane change
            new_lane = node.lane[vehicle]
            new_destinationLane = node.destinationLane[vehicle]
            new_isChangingLane = node.isChangingLane[vehicle]
    else:  # currently no lane change
        new_lc_timer = node.lc_timer[vehicle]
        new_lane = node.lane[vehicle]
        new_destinationLane = node.destinationLane[vehicle]
        new_isChangingLane = node.isChangingLane[vehicle]
    return new_destinationLane, new_isChangingLane, new_lane, new_lc_timer


def get_relevant_leader_acceleration(node, vehicle, lane, dt, v_crit, begin_lc=0):
    """
    Get IIDM acceleration of relevant leader. Considered leader targets are the leader car in lane (standard), the
    leader car in the left lane (right overtaking prohibition) and lane ends.
    """
    # lane ending
    delta_x_lane_end = max(node.roadway.get_dist_to_lane_end(node.position[vehicle], lane) - node.length[vehicle], 0.1)
    # consider lane end only if closer than 400 m and vehicle not in lane change
    if delta_x_lane_end < 400 and node.isChangingLane[vehicle] == 0 and begin_lc == 0:
        # desired behavior: accelerate until braking distance before lane ending
        brake_distance = abs(-0.5 / GD.max_break * node.velocity[vehicle] ** 2)
        if brake_distance < delta_x_lane_end and node.roadway.get_off_track(node.position[vehicle], lane) == 0:
            dist_lane_end = 1000
            v_lane_end = node.velocity[vehicle]
        else:
            dist_lane_end = delta_x_lane_end
            v_lane_end = 0
        a_lane_end = calculate_a_IIDM(dist_lane_end, v_lane_end, node.velocity[vehicle], node.IDM_param[vehicle], dt)
    else:
        a_lane_end = 1000

    # none/direct leader
    # find leader cars in lane and leader cars that change to or from lane
    leaders_in_lane = np.where(
        (node.position > node.position[vehicle]) & (lane - 0.5 <= node.lane) & (node.lane <= lane + 0.5))
    if len(leaders_in_lane[0]) == 0:  # no leader car
        delta_x_leader = 1000
        v_leader = 1000
    else:
        # leader_car = np.where(node.position[leaders_in_lane] == min(node.position[leaders_in_lane]))
        leader_car = np.argmin(node.position[leaders_in_lane])
        delta_x_leader = max(node.position[leaders_in_lane][leader_car] - node.position[vehicle] - node.length[vehicle],
                             0.1)
        v_leader = node.velocity[leaders_in_lane][leader_car]
    a_leader = calculate_a_IIDM(delta_x_leader, v_leader, node.velocity[vehicle], node.IDM_param[vehicle], dt)

    # right overtaking
    # find left leader cars in lane all left lanes
    leaders_left_lane = np.where(
        (node.position > (node.position[vehicle] + node.length[vehicle])) & (node.lane <= lane) & (
                node.velocity < node.velocity[vehicle]))
    if (len(leaders_left_lane[0]) == 0) or (node.velocity[vehicle] < v_crit) or delta_x_lane_end < 400:
        # no left leader car or velocity < v_crit or vehicle on ending lane
        a_left_leader = 1000
    else:
        # leader_car_left = np.where(node.position[leaders_left_lane] == min(node.position[leaders_left_lane]))
        leader_car_left = np.argmin(node.position[leaders_left_lane])
        delta_x_leader_left = max(
            node.position[leaders_left_lane][leader_car_left] - node.position[vehicle] - node.length[vehicle], 0.1)
        v_leader_left = node.velocity[leaders_left_lane][leader_car_left]
        # obey right overtaking prohibition only if a_left_leader >= -2 m/s^2
        threshold_rop = -2
        if GD.ext_params is not None:
            if "thresh_rop" in GD.ext_params.keys():
                threshold_rop = GD.ext_params["thresh_rop"]
        a_left_leader = max(
            calculate_a_IIDM(delta_x_leader_left, v_leader_left, node.velocity[vehicle], node.IDM_param[vehicle], dt),
            threshold_rop)
    return min(a_lane_end, a_leader, a_left_leader)


def calculate_a_IIDM(delta_x, v_leader, v_ego, IDM_param, dt):
    """
    Calculate IIDM acceleration for given distance and velocities with parameters IDM_param.
    """
    delta_v = v_ego - v_leader
    delta_x = max(delta_x, 0.1)

    # parameters
    s0 = IDM_param.s0
    T = IDM_param.T
    C1 = IDM_param.C1
    param_a = IDM_param.a
    v0 = IDM_param.V_0
    delta = IDM_param.delta

    s_star = s0 + max(
        T * v_ego + v_ego * delta_v / C1, 0)
    z = s_star / delta_x

    if z >= 1:  # vehicle is closer to leader car than desired
        a = param_a * (1 - (z ** 2))
    else:
        a_f = max(0.000001, param_a * (1 - (v_ego / v0) ** delta))
        a = a_f * (1 - (z ** (2 * param_a / a_f)))
        # accelerate in one step to v0 if leader is more than 1/0.8*z away
        a_v0 = (v0 - v_ego) / dt
        if z <= 0.8 and abs(a_v0) < param_a:
            a = a_v0
        if -0.05 < a < 0.05:
            a = 0
    return max(a, GD.max_break)


def calculate_inter_time_step(node, vehicle, acceleration, is_changing_lane, lane, destination_lane, sim_steps, dt,
                              v_crit):
    """
    Calculate multiple IIDM accelerations within one planning time step to smooth IIDM accleration in large dt.
    """
    # inter_level_node = copy.deepcopy(node)
    inter_level_node = node.copy_node()
    acceleration_list = [acceleration]
    for i in range(0, sim_steps):
        # integration of velocity and position: without driving backwards
        current_velocity = inter_level_node.velocity[vehicle] + acceleration * dt
        dt_0 = dt
        if current_velocity < 0:
            current_velocity = 0
            # avoid division by zero
            if abs(acceleration) < 0.1:
                acceleration = -0.1
            dt_0 = min(abs(inter_level_node.velocity[vehicle] / acceleration), dt)
        current_position = inter_level_node.position[vehicle] + inter_level_node.velocity[
            vehicle] * dt_0 + acceleration * 0.5 * (dt_0 ** 2)
        # calculate acceleration for next inter time step if there is one
        if i < sim_steps - 1:
            # loop though vehicles and predict position in inter time step with constant velocity model
            for prediction_vehicle in range(0, len(node.vehicle_id)):
                inter_level_node.position[prediction_vehicle] += max(inter_level_node.velocity[prediction_vehicle],
                                                                     0) * dt
            inter_level_node.velocity[vehicle] = current_velocity
            inter_level_node.position[vehicle] = current_position
            # calculate new acceleration
            if is_changing_lane == 1:
                # car in lane change
                start_lane = (2 * lane) - destination_lane
                # consider acceleration on start_lane only if start lane is on track when car changes lane
                if node.roadway.get_off_track(inter_level_node.position[vehicle], start_lane) == 1:
                    acc_start_lane = 1000
                else:
                    acc_start_lane = get_relevant_leader_acceleration(inter_level_node, vehicle, start_lane, dt, v_crit)
                    # no full break in inter time step if no full break in major time step
                    if acc_start_lane == GD.max_break and acceleration != GD.max_break:
                        acc_start_lane = acceleration
                acc_dest_lane = get_relevant_leader_acceleration(inter_level_node, vehicle, destination_lane, dt,
                                                                 v_crit)
            else:
                # car not in lane change
                acc_start_lane = 1000
                acc_dest_lane = get_relevant_leader_acceleration(inter_level_node, vehicle, lane, dt, v_crit)
            acceleration = min(acc_start_lane, acc_dest_lane)
            acceleration_list.append(acceleration)
    mean_acceleration = np.mean(acceleration_list)
    return mean_acceleration


if __name__ == '__main__':
    # test case
    from SimulationEnvironment.Car_Class import IDM_Variables

    IDM_param = IDM_Variables()
    IDM_param.set_Variables(
        V_0=31.5,
        T=0.5,
        a=1.4,
        b=2.0,
        delta=4,
        s0=2,
        s1=0,
        p=0.2)
    a = calculate_a_IIDM(16.4, 29.6, 28.8, IDM_param, 2.5)
