from SimulationEnvironment.SceneLoader import *
import random
import numpy as np
import array_funcs


class SimulationCore:
    """
    Created by:
    Matthias Blum

    Modified by:
    Marius Gruber

    Description:
    Core functions of simulation environment.
    """

    def __init__(self, global_data, behaviour_provider, scene_loader):
        self.global_data = global_data
        self.behaviour_provider = behaviour_provider
        self.sceneLoader = scene_loader

    def delete_all_cars(self):
        # Delete all vehicles
        self.global_data.number_of_cars = 0
        self.global_data.all_cars_data_array = []
        self.global_data.all_cars_object_list.clear()

    def apply_lane_changes(self):
        # check if two neighbor vehicles don't change to same lane. Therefore, within range
        # "puffer_between_lane_changes" only one lane change per time step is executed.
        if len(self.global_data.List_Lane_Changes) > 0:
            # set last lane change to negative value at beginning of each time step
            last_lane_change = self.global_data.start_len_without_lane_changes
            for i in range(0, len(self.global_data.List_Lane_Changes)):
                if self.global_data.List_Lane_Changes[
                                i].get_pos() > last_lane_change + self.global_data.puffer_between_lane_changes:
                    self.global_data.List_Lane_Changes[i].start_lane_change()
                    last_lane_change = self.global_data.List_Lane_Changes[i].get_pos()
                    print('applying lane change...')
            self.global_data.List_Lane_Changes = []

    ###################################################################################################################
    # Main process
    ###################################################################################################################

    def calculation_loop(self, queue_reference, loop_idx):
        try:
            if self.global_data.number_of_cars > 0 and len(self.global_data.all_cars_object_list) > 0:
                # Save values
                if self.global_data.save_step_data == True:
                    self.sceneLoader.save_step_data()

                # sort self.globalData.all_cars_data_array after position
                self.global_data.all_cars_data_array = self.global_data.all_cars_data_array[
                    np.argsort(self.global_data.all_cars_data_array[:, self.global_data.col_pos])]

                # Initialize indices of envirnoment models (lower value to 0 upper to maximum)
                self.global_data.all_cars_data_array[:, self.global_data.col_env_start] = 0
                self.global_data.all_cars_data_array[:, self.global_data.col_env_end] = \
                    self.global_data.all_cars_data_array.shape[0]

                # Calculate indices of all vehicles in self.globalData.all_cars_data_array. Indices are one row
                # matrix. The cars in the environment model are lines in self.globalData.all_cars_data_array from
                # line col_env_start to col_env_end. Works because self.globalData.all_cars_data_array was sorted by
                # position
                array_funcs.get_env_index(self.global_data.all_cars_data_array, int(self.global_data.col_pos),
                                          int(self.global_data.col_env_start),
                                          int(self.global_data.col_env_end), self.global_data.len_half_env)

                # Every vehicle in self.globalData.all_cars_object_list is assigned a matrix with information of
                # all vehicles witin environment.
                for i in range(0, self.global_data.number_of_cars):
                    # Enviroment as copy
                    self.global_data.all_cars_object_list[
                        int(self.global_data.all_cars_data_array[i][self.global_data.col_id])].set_car_env(
                        self.global_data.all_cars_data_array[
                        int(self.global_data.all_cars_data_array[i][self.global_data.col_env_start]):
                        int(self.global_data.all_cars_data_array[i][self.global_data.col_env_end])][:].copy())

                    self.global_data.all_cars_object_list[
                        int(self.global_data.all_cars_data_array[i][self.global_data.col_id])].set_data(
                        self.global_data.all_cars_data_array
                        [i:i + 1][:])

                for i in range(0, self.global_data.number_of_cars):
                    # update lane environment
                    array_funcs.get_lane_env(self.global_data.roadway.lane_matrix,
                                             self.global_data.roadway.lane_len_sections,
                                             self.global_data.all_cars_object_list[i].environment.env_lanes,
                                             self.global_data.all_cars_object_list[i].get_pos(),
                                             self.global_data.len_env)

                    # find direct neighbor cars
                    array_funcs.analyse_neighbor_cars(self.global_data.all_cars_object_list[i].environment.env_cars,
                                                      self.global_data.all_cars_object_list[
                                                          i].environment.direct_neighboor_cars,
                                                      self.global_data.col_pos, self.global_data.col_lane,
                                                      self.global_data.col_id,
                                                      self.global_data.all_cars_object_list[i].get_id(),
                                                      self.global_data.len_env)

                    array_funcs.analyse_all_neighbor_cars(self.global_data.all_cars_object_list[i].environment.env_cars,
                                                          self.global_data.all_cars_object_list[
                                                              i].environment.all_leader_and_follower_cars,
                                                          self.global_data.all_cars_object_list[i].get_data(),
                                                          self.global_data.col_pos, self.global_data.col_lane,
                                                          self.global_data.col_id)

                    # calculate IDs of direct neighbors (leader, follower, left leader...)
                    self.global_data.all_cars_object_list[i].calc_IDs_direkt_neighboors()

                vehicles = self.global_data.all_cars_object_list[:]
                # Make sure vehicles are in order
                vehicles.sort(key=lambda x: x.get_id())
                behaviours, queue_reference, loop_idx = self.behaviour_provider.get_behaviours(queue_reference,
                                                                                               loop_idx, vehicles)
                # Check for valid output
                if len(behaviours) is not len(self.global_data.all_cars_object_list):
                    raise ValueError(
                        "Error in calculation_loop: array dimension of return behaviours "
                                "does not match all_cars_object_list")
                elif None in behaviours:
                    raise ValueError("Error in calculation_loop: Missing return value in vehicleBehaviours")

                # Set car behaviour attributes
                for vehicle, behaviour in list(zip(self.global_data.all_cars_object_list, behaviours)):
                    assert vehicle.get_id() == behaviour.vehicle_id
                    vehicle.setBehaviour(behaviour)

                # All vehicles that want to change lanes are collected in self.globalData.List_Lane_Changes
                for i in range(0, self.global_data.number_of_cars):
                    self.global_data.all_cars_object_list[i].collect_all_lane_changes()

                # apply lane changes
                self.apply_lane_changes()

                # move cars
                for i in range(0, self.global_data.number_of_cars):
                    self.global_data.all_cars_object_list[i].move_car()
                return queue_reference, loop_idx

        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            raise e
