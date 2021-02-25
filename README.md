# Test scenarios for cooperative behavior planning on highways

## Introduction

This repository contains a simulation tool for cooperative behavior planning on highways. The user can specify a traffic scenario by providing a cars.txt and roadway.txt file (see description below) and solve it with three different behavior models:

- Improved Intelligent Driver Model (IIDM) with MOBIL lane change model (for details see https://doi.org/10.1007/978-3-642-32460-4_11                    and https://doi.org/10.3141/1999-10                   ) 
- Central A*-Search algorithm: Plans the behavior of multiple vehicles with perfect coordination (witin the selcted discretization). For further details see https://doi.org/10.3390/app10228154                   .
- Decentral A*-Search algorithm: Plans the behavior of vehicles without inter vehicle coordination. Uses the IIDM and MOBIL model to estimate the behavior of surrounding traffic.

The simulation outputs the simulation result as a .csv file with all relevant data (positions, velocities, accelerations and lanes over time). In addition, a maneuver plot visualizes the  sequence of events of the scenario.


## Description of In- and Outputs

The input of the simulation are .txt files for specifying the roadway and the start scenario of the vehicles. Example scenario definitions can be found in the folder `\SimulationEnvironment\init_data\Scenes`. The roadway file comprises two elements to model a highway: sections and entrances. A section is specified by its length in meter (first column), number of lanes (second column) and the start lane (third column). The lanes are numbered in ascending order from left (leftmost lane ID 0) to right. The entry lanes are specified by their start position (first column), the length of the solid line that separates the entry lane from the section lanes (second column, for visualization), the length of the dashed line that separates the entry lane from the section lanes (third column) and the lane number the highway entry is located on (fourth column). Example for a roadway file:

```
roadway
4000 	2 	0

entrances
0 	50 	200 	2
```
A cars file contains the information for specifying the state of al vehicles for the start of the scenario and is structured as a table as shown below. Every line represents one vehicle for which the following parameters need to be provided:
- ID: ID of the vehicle.
- pos: Position in x-direction. Reference point is the rear end of a vehicle.
- vel: Velocity in x-direction.
- a: Acceleration in x-direction.
- lane: Lane ID.
- len: Length of the vehicle.
- width: Width of the vehicle.
- driver_model: Behavior model to be used for simulation (1: IIDM, 2: central planner, 3: decentral planner). Different behavior models should not be used in one simulation.
- T: Time headway parameter of the IIDM model.
- V_0: Desired velocity of the vehicle.
- a: Acceleration parameter of the IIDM model.
- b: Deceleration parameter of the IIDM model.
- delta: Acceleration coefficient parameter of the IIDM model.
- p: Politeness parameter of the MOBIL model.
- s0: Minimum distance parameter of the IIDM model.
- lc_time_out: Length of vehicle's lane change in seconds.

```
ID		pos		vel		a		lane            len		width	        driver_model        T		V_0		a		b		delta	        p	        s0	        lc_time_out
0.000000 	150.000000 	40.000000 	0.000000 	1.000000 	7.000000 	2.000000 	3.000000 	    1.000000 	40.000000 	1.000000 	1.500000 	4.000000 	0.200000 	2.000000 	5.000000
1.000000 	250.000000 	32.000000 	0.000000 	1.000000 	7.000000 	2.000000 	3.000000 	    1.000000 	32.000000 	1.000000 	1.500000 	4.000000 	0.200000 	2.000000 	5.000000
2.000000 	150.000000 	41.000000 	0.000000 	0.000000 	7.000000 	2.000000 	3.000000 	    1.000000 	41.000000 	1.000000 	1.500000 	4.000000 	0.200000 	2.000000 	5.000000
3.000000 	 30.000000 	40.000000 	0.000000 	0.000000 	7.000000 	2.000000 	3.000000 	    1.000000 	40.000000 	1.000000 	1.500000 	4.000000 	0.200000 	2.000000 	5.000000
```


In addition to the roadway and cars .txt files, the simulaten can be controlled externally by passing a scene element to the function call of the main script. If not specified the scenario description will be read from Cars.txt and Roadway.txt defined in `user_settings.py`. The scene element is a tuple of two lists (roadway_list, cars_list) that consist of one string for each line in Cars.txt/Roadway.txt.

The simulation output is written in a .csv file in the folder `\SimulationEnvironment\memory` and the file is named according to the name of the performed simulation (specified in `user_settings.py`). It contains the following parameters:
- time_step: Time stamp of the simulaton data in this line.
- ID: ID of the vehicle.
- ID_leader: ID of the leading vehicle.
- pos: Position in x-direction. Reference point is the rear end of a vehicle.
- vel: Velocity in x-direction.
- acc: Acceleration in x-direction.
- lane: Lane ID.
- length: Length of the vehicle.

Furthermore, the simulation result is visualized by means of two plots each. The lateral plot (first figure) shows the scenario from the bird’s eye perspective in each relevant time step. Besides the start and end of the scenario, relevant time steps are the first time steps of and
after a lane change. The longitudinal plot (second figure) shows the velocity of all involved vehicles in solid lines as well as their desired velocities in horizontal dashed lines. The vertical dashed lines indicate the time steps where a lane change starts or ends. These time steps are the relevant time steps depicted in the bird’s eye perspective plot.

![Lateral plot.](/common/images/plot_lateral.png "Lateral plot.")
![Longitudinal plot.](/common/images/plot_longitudinal.png "Longitudinal plot.")

## List of main components
- `CooperativeBehaviorPlanning.py`: Main file of the simulation program.
- `AsymmetricIDM.py`: Implementation of the IIDM driver model with MOBIL lane change behavior. "Asymmetric" refers to the prohibition of overtaking on the right, which is obeyed within all behavior models.
- `AStarSearch.py`: Implementation of the central A*-Search algorithm.
- `DecentralizedAStarSearch.py`: Implementation of the decentral A*-Search algorithm.
- `Node.py`: Implementation of the nodes that build the search tree for both the central and decentral planning algorithm.
- `costs.py`: Cost function for the evaluation of node states of the search tree.
- `ParameterEstimator.py`: Parameter estimation of the behavior of other vehicles (necesarry only for decentral planning).


## Requirements
The code is developed with Python 3.6. All required python modules are listed in the `requirements.txt`file in this repo. They can be installed automatically with `pip3 install -r /path/to/requirements.txt`.

Furthermore you need to install a custom library for running the simulation program (see below in "getting started").

  
## Getting started
- `Step 1`: Clone the repo.
- `Step 2`: Cython requires a C++ compiler. If no compiler is installed on your system you have to install one (e.g. the build tools for Visual Studio (https://visualstudio.microsoft.com/de/downloads/ -> tools for Visual Studio 2019 -> build tools, install them and chose the C++ build tools option to install the required C++ compiler and its dependencies).
- `Step 3`: Install the custom library:
    - Windows
        - start command line (cmd.exe)
        - activate correct virtual environment of your IDE (e.g. PyCharm) by running the "activate.bat" file: execute \path\to\virtual_env\Scripts\activate in command window ("venv" must be shown in the command line)
        - navigate to the folder "SimulationEnvironment" and run "python setup.py install" to start compilation
    - Linux
        - activate correct virtual environment of your IDE (e.g. PyCharm) by running "source /path/to/virtual_env/bin/activate" in the command line
        - install gcc if not already installed: "sudo apt-get install gcc"
        - navigate to the folder "SimulationEnvironment" and run "python3 setup.py install" to start compilation


## Running the code
- `Step 1` (optional): Specify the scenario to be simulated in `user_settings.py` or create an own scenario (Cars.txt and Roadway.txt file).
- `Step 2`: Run the `CooperativeBehaviorPlanning.py` file. The scenario specified in `user_settings.py` will be computed. After the simulation has finished, the result plots are shown and the result file will be written to the `\SimulationEnvironment\memory` folder, named like the scenario in `user_settings.py`.

## License
This project is licensed under the LGPL License - see the LICENSE file for details
 
 
## References
The simulation environment was used and described briefly in:

Data-Driven Test Scenario Generation for Cooperative Maneuver Planning on Highways
by Christian Knies and Frank Diermeyer
DOI: 10.3390/app10228154                   
Free full text available: https://www.mdpi.com/2076-3417/10/22/8154

If you find our work useful in your research, please consider citing:

```
@article{Knies2020,
  doi = {10.3390/app10228154},
  url = {https://doi.org/10.3390/app10228154},                   
  year = {2020},
  month = nov,
  publisher = {{MDPI} {AG}},
  volume = {10},
  number = {22},
  pages = {8154},
  author = {Christian Knies and Frank Diermeyer},
  title = {Data-Driven Test Scenario Generation for Cooperative Maneuver Planning on Highways},
  journal = {Applied Sciences}
}
```
