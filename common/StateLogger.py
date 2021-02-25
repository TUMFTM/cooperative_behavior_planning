import pandas as pd
import time
from typing import List
from pathlib import Path
from lib.IVehicle import IVehicle
import common.user_settings as cfg


class StateFileRepository:
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    This class provides file persistence for store logged state data
    by using pandas data frames and csv files.
    """

    def __init__(self):
        # Store state data
        self.state_data = []

    def add(self, row):
        self.state_data.append(row)

    def dump_row(self, row, file_path):
        df = pd.DataFrame([row], columns=['time_step', 'id', 'id_leader', 'pos', 'vel', 'acc', 'lane', 'length'])
        df.to_csv(file_path, mode='a', index=False, header=False)

    def dump(self, file_path):
        df = pd.DataFrame(self.state_data,
                          columns=['time_step', 'id', 'id_leader', 'pos', 'vel', 'acc', 'lane', 'length'])
        df.to_csv(file_path, mode='w+', index=False)

    def load(self, file_path):
        df = pd.read_csv(file_path)
        return df

    def reset(self):
        self.state_data = []

    def return_dataframe(self):
        return pd.DataFrame(self.state_data,
                            columns=['time_step', 'id', 'id_leader', 'pos', 'vel', 'acc', 'lane', 'length'])


class StateLogger:
    """
    Created by:
    Marius Gruber

    Modified by:
    Christian Knies

    Description:
    The StateLogger class saves simulation data to file specified in "file_path"
    """
    scene = cfg.scene
    file_path = Path(__file__).parent.parent.joinpath('SimulationEnvironment/memory/{}.csv'.format(scene))
    repo = StateFileRepository()
    total_time = cfg.logging_total_time
    time_step = 0
    time_stats = []

    # Simulation timing log
    at_start = True
    simulation_starting_time = None
    simulation_total_time = None

    @staticmethod
    def log_state(cars: List[IVehicle], sim_time_step):
        # only execute logging within logging time
        if StateLogger.time_step <= StateLogger.total_time:
            # Store simulation starting time
            if StateLogger.at_start is True:
                StateLogger.repo.reset()
                StateLogger.simulation_starting_time = time.time()
                StateLogger.at_start = False

            # create new row in logging file
            for car in cars:
                # Extract states from car object and store in repository
                id = car.get_id()
                id_leader = car.id_leader
                pos = car.get_pos()
                vel = car.get_vel()
                acc = car.get_a()
                lane = car.get_lane()
                length = car.get_len()
                row = [StateLogger.time_step, id, id_leader, pos, vel, acc, lane, length]

                StateLogger.repo.add(row)
            StateLogger.time_step += sim_time_step
            # Round to avoid floating point errors
            StateLogger.time_step = round(StateLogger.time_step, 2)

        if StateLogger.time_step == StateLogger.total_time + sim_time_step:
            StateLogger.save()
            StateLogger.simulation_total_time = time.time() - StateLogger.simulation_starting_time
            StateLogger.time_step = 0
            StateLogger.at_start = True
            StateLogger.simulation_starting_time = None
            StateLogger.simulation_total_time = None

        return StateLogger.repo.return_dataframe()

    @staticmethod
    def log_execution_time(time):
        StateLogger.time_stats.append(time)

    @staticmethod
    def save():
        StateLogger.repo.dump(StateLogger.file_path)
