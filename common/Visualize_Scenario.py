from matplotlib import pyplot as plt
import pandas
import numpy as np
import os
from scipy.special import expit
from matplotlib import offsetbox as osb
import math
from common.user_settings import scene
from DriverModels.A_star.costs import evaluate_safety

# set parameters for matplotlib latex output
use_latex = False
if use_latex is True:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.latex.preamble": [
            "\\usepackage{siunitx}"  # load additional packages
        ],
        "pgf.preamble": [
            "\\usepackage{siunitx}"  # load additional packages
        ]
    })
    # parameters for width in latex
    textwidth_pt = 443.86  # textwidth in pt in latex document
    textwidth_in = textwidth_pt / 71.86  # textwidth in inches for matplotlib
else:
    textwidth_in = 6


def plot_scenario_from_file(path_results, path_roadway):
    """
    Created by:
    Christian Knies

    Modified by:
    -

    Description:
    Plot scenario from result file.
    :param path_results: Path to result file.
    :param path_roadway: Path to roadway file.
    """
    # read result data
    vehicle_data, num_vehicles, dt = read_result_file(path_results)
    # find lane changes
    lane_changes = find_lane_changes(vehicle_data)
    # plot longitudinal data
    scenario = {}
    plt.figure(figsize=[textwidth_in * 1.0, textwidth_in * 0.9])
    plot_longitudinal(scenario, vehicle_data, num_vehicles, lane_changes, dt)
    if use_latex is True:
        plt.savefig("common/figures/Figure_long" + ".pgf", format="pgf", dpi=1200, bbox_inches='tight')
    # plot lateral data
    roadway = read_roadway_data(path_roadway)
    plt.figure(figsize=[textwidth_in * 1.15, textwidth_in * (len(lane_changes) + 2) * 0.3])
    plot_roadway(roadway, vehicle_data, num_vehicles, lane_changes, dt)
    if use_latex is True:
        plt.savefig("common/figures/Figure_lat" + ".pgf", format="pgf", dpi=1200, bbox_inches='tight')
    plt.show()


def plot_scenario_from_dict(scenario, vehicle_data, modelname=None):
    """
    Created by:
    Christian Knies

    Modified by:
    -

    Description:
    Plot scenario from scenario dictionary
    :param scenario: Dictionary of scenario.
    :param vehicle_data: Result of simulation for each vehicle.
    :param modelname: Name of behavior model (optional).
    """
    # calculate number of vehicles and dt
    num_vehicles = len(scenario["vehicle_id"])
    dt = min(vehicle_data["time"][vehicle_data["time"] > 0])
    # find lane changes
    lane_changes = find_lane_changes(vehicle_data)
    # plot longitudinal data
    plt.figure(figsize=[textwidth_in * 1.0, textwidth_in * 0.9])
    plot_longitudinal(scenario, vehicle_data, num_vehicles, lane_changes, dt)
    if use_latex is True:
        plt.savefig("figures/Figure_long_" + str(modelname) + ".pgf", format="pgf", dpi=1200, bbox_inches='tight')
    # plot lateral data
    roadway = scenario["roadway_model"]
    plt.figure(figsize=[textwidth_in * 1.15, textwidth_in * (len(lane_changes) + 2) * 0.3])
    plot_roadway(roadway, vehicle_data, num_vehicles, lane_changes, dt)
    if use_latex is True:
        plt.savefig("figures/Figure_lat_" + str(modelname) + ".pgf", format="pgf", dpi=1200, bbox_inches='tight')
    plt.show()


def plot_scenario_from_node(node, time_step):
    """
    Created by:
    Christian Knies

    Modified by:
    -

    Description:
    Plot scenario from A-Star node (for debugging).
    :param node: Node of A-Star planning process.
    :param time_step: Time step of simulation.
    """
    # calculate number of vehicles and dt
    num_vehicles = len(node.vehicle_id)
    roadway_model = node.roadway
    if not roadway_model.entrances:
        entrance_list = []
    else:
        entrance_list = [{"start": roadway_model.entrances[0][0], "begin_merge": roadway_model.entrances[0][1],
                          "end_merge": roadway_model.entrances[0][2], "lane": roadway_model.entrances[0][3]}]
    dt = time_step
    roadway = {"section": [{"length": list(roadway_model.sections.keys())[0],
                            "lanes": len(roadway_model.sections[list(roadway_model.sections.keys())[0]]),
                            "start_lane": min(roadway_model.sections[list(roadway_model.sections.keys())[0]])}],
               "entrance": entrance_list}
    scenario = {}
    scenario["desired_velocity"] = []
    for parameter_set in node.IDM_param:
        scenario["desired_velocity"].append(parameter_set.V_0)
    # build up vehicle_data dict
    vehicle_data = {"id_leader": {}, "vel": {}, "a": {}, "lane": {}, "position": {}, "time_rough": np.array([]),
                    "length": {}}
    while node is not None:
        # find leader car ids
        leader_car_ids = find_leader_car_ids(node)
        # loop through node and build up vehicle_data dict
        loop_vehicles(vehicle_data["id_leader"], leader_car_ids, "front")
        loop_vehicles(vehicle_data["vel"], node.velocity, "front")
        loop_vehicles(vehicle_data["a"], node.acceleration, "front")
        loop_vehicles(vehicle_data["lane"], node.lane, "front")
        loop_vehicles(vehicle_data["position"], node.position, "front")
        vehicle_data["time_rough"] = np.append(node.level * time_step, vehicle_data["time_rough"])
        loop_vehicles(vehicle_data["length"], node.length, "front")
        node = node.parent
    # stuff vectors to higher dt
    dt_plot = 0.1
    vehicle_data["time"] = np.arange(min(vehicle_data["time_rough"]), max(vehicle_data["time_rough"]) + dt_plot,
                                     dt_plot)
    for vehicle in range(0, num_vehicles):
        vehicle_data["id_leader"][vehicle] = stuff_vectors(vehicle_data["time"], vehicle_data["time_rough"],
                                                           vehicle_data["id_leader"][vehicle], "block")
        vehicle_data["vel"][vehicle] = stuff_vectors(vehicle_data["time"], vehicle_data["time_rough"],
                                                     vehicle_data["vel"][vehicle], "linear")
        vehicle_data["a"][vehicle] = stuff_vectors(vehicle_data["time"], vehicle_data["time_rough"],
                                                   vehicle_data["a"][vehicle], "block")
        vehicle_data["lane"][vehicle] = stuff_vectors(vehicle_data["time"], vehicle_data["time_rough"],
                                                      vehicle_data["lane"][vehicle], "block")
        vehicle_data["position"][vehicle] = stuff_vectors(vehicle_data["time"], vehicle_data["time_rough"],
                                                          vehicle_data["position"][vehicle], "linear")
        vehicle_data["length"][vehicle] = stuff_vectors(vehicle_data["time"], vehicle_data["time_rough"],
                                                        vehicle_data["length"][vehicle], "linear")
    # find lane changes
    lane_changes = find_lane_changes(vehicle_data)
    # plot longitudinal data
    plt.figure(figsize=[textwidth_in * 1.0, textwidth_in * 0.9])
    plot_longitudinal(scenario, vehicle_data, num_vehicles, lane_changes, dt_plot)
    # plot lateral data
    plt.figure(figsize=[textwidth_in * 1.15, textwidth_in * (len(lane_changes) + 2) * 0.3])
    roadway2 = read_roadway_data(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))) + '/SimulationEnvironment/init_data/Scenes/' + scene + '/roadway.txt')
    plot_roadway(roadway, vehicle_data, num_vehicles, lane_changes, dt_plot)
    plt.show()


def read_result_file(result_path):
    # initialize dict
    vehicle_data = {}
    vehicle_data["id_leader"] = {}
    vehicle_data["vel"] = {}
    vehicle_data["a"] = {}
    vehicle_data["lane"] = {}
    vehicle_data["position"] = {}
    vehicle_data["time"] = {}
    vehicle_data["length"] = {}
    # read result file
    file_path = open(result_path, "r")
    csv_data = pandas.read_csv(file_path)
    dt = min(csv_data.time_step.values[csv_data.time_step.values > 0])
    num_vehicles = max(csv_data.id.values) + 1
    for vehicle in range(0, num_vehicles):
        datalines = np.where(csv_data.id.values == vehicle)
        vehicle_data["id_leader"][vehicle] = csv_data.id_leader.values[datalines]
        vehicle_data["vel"][vehicle] = csv_data.vel.values[datalines]
        vehicle_data["a"][vehicle] = csv_data.acc.values[datalines]
        vehicle_data["lane"][vehicle] = csv_data.lane.values[datalines]
        vehicle_data["position"][vehicle] = csv_data.pos.values[datalines]
        vehicle_data["length"][vehicle] = csv_data.length.values[datalines]
    vehicle_data["time"] = csv_data.time_step.values[datalines]
    return vehicle_data, num_vehicles, dt


def read_roadway_data(path_roadway):
    roadway = {}
    roadway["section"] = []
    roadway["entrance"] = []
    file_load_roadway = open(path_roadway)
    mode = 0
    for line in file_load_roadway:
        line = line.split()
        if line == []:
            mode = 0
        else:
            if mode > 0:
                for i in range(0, len(line)):
                    line[i] = int(line[i])
            if mode == 1:  # read/init roadway data
                roadway["section"].append({"length": line[0], "lanes": line[1], "start_lane": line[2]})
            if mode == 2:  # read/init entrances  data
                roadway["entrance"].append(
                    {"start": line[0], "begin_merge": line[0] + line[1], "end_merge": line[0] + line[1] + line[2],
                     "lane": line[3]})
            if line[0] == 'roadway':
                mode = 1
            if line[0] == 'entrances':
                mode = 2
            if line[0] == 'exits':
                mode = 3
            if line[0] == 'measurement_stations':
                mode = 4
    file_load_roadway.close()
    return roadway


def plot_longitudinal(scenario, vehicle_data, num_vehicles, lane_changes, dt):
    # plot velocity
    plt.subplot(3, 1, 1)
    plt.xlabel("time in s")
    if use_latex is True:
        plt.ylabel('velocity in \\SI{}{\meter\per\second}')
    else:
        plt.ylabel('velocity in m/s')
    plt.xlim(min(vehicle_data["time"]), max(vehicle_data["time"]))
    plt.grid(b=True)
    for vehicle in range(0, num_vehicles):
        p = plt.plot(vehicle_data["time"], vehicle_data["vel"][vehicle], label="vehicle " + str(vehicle))
        if "desired_velocity" in scenario:
            plt.plot(vehicle_data["time"], scenario["desired_velocity"][vehicle] * np.ones(len(vehicle_data["time"])),
                     color=p[-1].get_color(), linestyle="-.")
    for lc in lane_changes:
        plt.axvline(x=lc * dt, linewidth=2, color='k', linestyle="--")
    plt.legend(loc="upper right")
    # plt.legend(loc="lower right")

    # plot acceleration
    plt.subplot(3, 1, 2)
    plt.xlabel("time in s")
    if use_latex is True:
        plt.ylabel("acceleration in \\SI{}{\meter\per\square\second}")
    else:
        plt.ylabel("acceleration in m/s^2")
    plt.xlim(min(vehicle_data["time"]), max(vehicle_data["time"]))
    plt.grid(b=True)
    for vehicle in range(0, num_vehicles):
        plt.plot(vehicle_data["time"], vehicle_data["a"][vehicle], label="vehicle " + str(vehicle))
    for lc in lane_changes:
        plt.axvline(x=lc * dt, linewidth=2, color='k', linestyle="--")
    plt.legend(loc="upper right")

    # plot front distance
    plt.subplot(3, 1, 3)
    plt.xlabel("time in s")
    if use_latex is True:
        plt.ylabel("front distance in \\SI{}{\meter}")
    else:
        plt.ylabel("front distance in m")
    plt.xlim(min(vehicle_data["time"]), max(vehicle_data["time"]))
    plt.grid(b=True)
    for vehicle in range(0, num_vehicles):
        front_distance = np.ones(len(vehicle_data["time"])) * 500
        safety = np.zeros(len(vehicle_data["time"]))
        for time in range(0, len(vehicle_data["time"])):
            if vehicle_data["id_leader"][vehicle][time] in vehicle_data["id_leader"].keys():
                id_leader = vehicle_data["id_leader"][vehicle][time]
                front_distance[time] = vehicle_data["position"][id_leader][time] - vehicle_data["position"][vehicle][
                    time] - vehicle_data["length"][vehicle][time]
                # calculate safety
                if evaluate_safety(front_distance[time], vehicle_data["vel"][id_leader][time],
                                   vehicle_data["vel"][vehicle][time], 0.5) != 0:
                    safety[time] = 1
        plt.plot(vehicle_data["time"], front_distance, label="vehicle " + str(vehicle))
        # plot unsafe states in red
        plt.plot(vehicle_data["time"][safety == 1], front_distance[safety == 1], color="red")
    for lc in lane_changes:
        plt.axvline(x=lc * dt, linewidth=2, color='k', linestyle="--")
    plt.ylim(0, 100)
    plt.legend(loc="upper right")


def plot_roadway(roadway, vehicle_data, num_vehicles, lane_changes, dt):
    # time steps to plot: start, end, lane changes
    plot_time_steps = np.append(lane_changes, np.array([0 / dt, max(vehicle_data["time"]) / dt]))
    plot_time_steps = np.sort(np.unique(plot_time_steps))  # sort all unique time steps
    num_plots = len(plot_time_steps)
    # find out length of roadway plot
    x_spread = 0
    for n_plot in range(0, num_plots):
        ts = int(plot_time_steps[n_plot])
        x_start = 1e10
        x_end = 0
        for car in range(0, num_vehicles):
            x_start = min(vehicle_data["position"][car][ts], x_start)
            x_end = max(vehicle_data["position"][car][ts] + vehicle_data["length"][car][ts], x_end)
        x_spread = max(np.ceil(x_end - x_start), x_spread)
    # loop through all lane changes and create subplot
    for pts in range(0, num_plots):
        # create subplot
        ax = plt.subplot(num_plots, 1, pts + 1)
        plt.gca().invert_yaxis()
        plt.gca().set_facecolor('xkcd:grey')
        # plot sections
        length_before_section = 0
        min_lane = 10
        max_lane = 0
        for section in roadway["section"]:
            # find min and max lane
            min_lane = min(min_lane, section["start_lane"])
            max_lane = max(max_lane, section["start_lane"] + section["lanes"] - 1)
            # create x values
            x_values = [length_before_section, length_before_section + section["length"]]
            length_before_section = length_before_section + section["length"]
            # create lanes
            for lane in range(section["start_lane"], section["start_lane"] + section["lanes"] + 1):
                # decide if outer lane (solid line) or inner lane (dashed line)
                if lane == section["start_lane"] or lane == section["start_lane"] + section["lanes"]:
                    plt.plot(x_values, [lane - 0.5, lane - 0.5], "w-")
                else:
                    plt.plot(x_values, [lane - 0.5, lane - 0.5], "w--")
        for entrance in roadway["entrance"]:
            # find min and max lane
            min_lane = min(min_lane, entrance["lane"])
            max_lane = max(max_lane, entrance["lane"])
            # setting for bow
            x_shift = 10  # sigmoid function sig(10) is close to zero and therefore end point of bow
            x_scale = 0.2
            length_entrance = entrance["end_merge"] - entrance["start"]
            num_points = length_entrance * 1
            # x_value for dashed line
            x_dash = [entrance["begin_merge"], entrance["end_merge"] + (x_shift / x_scale) * (
                        ((x_shift / x_scale) + length_entrance) / length_entrance)]
            plt.plot(x_dash, [entrance["lane"] - 0.5, entrance["lane"] - 0.5], color="xkcd:grey", linestyle="--")
            # x_value for bow
            x_bow = np.linspace(entrance["start"], entrance["end_merge"] + (x_shift / x_scale) * (
                        ((x_shift / x_scale) + length_entrance) / length_entrance), num_points)
            # y_value for bow
            y_bow = entrance["lane"] + 0.5 - expit(np.linspace(-length_entrance, 0, num_points) * x_scale + x_shift)
            plt.plot(x_bow, y_bow, "w-")
        # plot cars on roadway
        min_x = 1e10
        max_x = 0
        for car in range(0, num_vehicles):
            # read position data
            time_step = int(plot_time_steps[pts])
            x_coordinate_start = vehicle_data["position"][car][time_step]
            x_coordinate_end = x_coordinate_start + vehicle_data["length"][car][time_step]
            y_coordinate_start = vehicle_data["lane"][car][time_step] + 0.25
            y_coordinate_end = vehicle_data["lane"][car][time_step] - 0.25
            # read image depending on type (car, truck) and color
            if vehicle_data["length"][car][time_step] < 10:
                vehicle_type = "car"
            else:
                vehicle_type = "truck"
            img_path = os.path.dirname(os.path.realpath(__file__)) + "/images/" + vehicle_type + str(car) + ".png"
            pic = plt.imread(img_path)
            # Plot with annotationbox (image has always same size regardless of window size and zoom)
            # y_coordinate = vehicle_data["lane"][car][time_step]
            # img = osb.OffsetImage(pic, zoom=0.015)
            # img = osb.AnnotationBbox(img, (x_coordinate_start, y_coordinate), xycoords='data', pad=0.0, box_alignment=(0, 0.5), frameon=False)
            # plt.gca().add_artist(img)
            plt.imshow(pic, aspect="auto",
                       extent=(x_coordinate_start, x_coordinate_end, y_coordinate_start, y_coordinate_end),
                       resample=False, interpolation='quadric', zorder=2)
            # find min and max x limitation
            min_x = min(min_x, x_coordinate_start)
            max_x = max(max_x, x_coordinate_end)
        x_core = max_x - min_x
        min_margin_right = 70
        margin_left = (x_spread + min_margin_right - x_core) / 2
        min_x = max(min_x - margin_left, 0)
        max_x = min_x + x_spread + min_margin_right + 0
        # set yticks to integers and xlim/ylim
        plt.yticks(range(min_lane, max_lane + 1))
        plt.xlim(min_x, max_x)
        plt.ylim(max_lane + 0.7, min_lane - 0.7)
        # Textbox for time stamp
        textbox = osb.TextArea("t = " + '{:2.1f}'.format(time_step * dt) + " s", minimumdescent=False)
        ab = osb.AnnotationBbox(textbox, (max_x - 5, min_lane - 0.3), xycoords='data', pad=0.5, box_alignment=(1, 1),
                                frameon=True)
        plt.gca().add_artist(ab)


def find_lane_changes(vehicle_data):
    # find time steps with beginning and ending of lane changes for plot
    lane_changes = np.array([])
    for lane in vehicle_data["lane"]:
        delta_lane = np.array([0])
        delta_lane = np.append(delta_lane, vehicle_data["lane"][lane][0:-1] - vehicle_data["lane"][lane][1:])
        lane_changes = np.append(lane_changes, np.where(delta_lane != 0))
    lane_changes = np.sort(lane_changes)
    return lane_changes


def find_leader_car_ids(node_object):
    leader_list = []
    for vehicle in range(0, len(node_object.vehicle_id)):
        lane = node_object.lane[vehicle]
        leaders_in_lane = np.where(
            (node_object.position > node_object.position[vehicle]) & (math.floor(lane) - 0.5 <= node_object.lane) & (
                    node_object.lane <= math.ceil(lane) + 0.5))
        if len(leaders_in_lane[0]) == 0:  # no leader car
            leader_list.append(-1)
        else:
            leader_car = np.argmin(node_object.position[leaders_in_lane])
            leader_list.append(leaders_in_lane[0][leader_car])
    return leader_list


def loop_vehicles(vd_dict, variable_list, put_position):
    for var_position in range(0, len(variable_list)):
        if var_position not in vd_dict.keys():
            vd_dict[var_position] = np.array([variable_list[var_position]])
        else:
            if put_position == "front":
                vd_dict[var_position] = np.append(variable_list[var_position], vd_dict[var_position])
            elif put_position == "back":
                vd_dict[var_position] = np.append(vd_dict[var_position], variable_list[var_position])
            else:
                print("no valid parameter given")


def stuff_vectors(x_target, x_rough, vector_rough, stuff_type):
    if stuff_type == "linear":
        return np.interp(x_target, x_rough, vector_rough)
    elif stuff_type == "block":
        dt = min(np.diff(x_target))
        x_stuffed = np.append(x_rough, x_rough + dt)
        x_stuffed = np.sort(x_stuffed[x_stuffed <= max(x_rough)])
        vector_stuffed = np.repeat(vector_rough, 2)[1:]
        return np.interp(x_target, x_stuffed, vector_stuffed)


if __name__ == "__main__":
    path_results = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) + '/SimulationEnvironment/memory/03_V4_overtake.csv'
    path_roadway = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) + '/SimulationEnvironment/init_data/Scenes/08_V1_merge/roadway.txt'
    plot_scenario_from_file(path_results, path_roadway)
