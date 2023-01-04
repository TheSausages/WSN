import math
import os
import shutil

import generate_graph
from graph import NetworkInformation, Vertex, Graph, VertexType
from timeit import default_timer as timer

from solver import solver_solution, get_path_in_order
from wsn_algorithm import run_algorythm


# INPUT DATA  - VARIABLES

# Probability for successfully sending

p_other = 0.9

# Distance, above which communication is impossible
r_max = 25

# Max and Min energies
e_max = 20
e_min = 5

# Energy lost per Package
energy_per_package = 2

# Mathematical coeficients
beta_coef = 0.03
gamma_coef = 0.15

# Game theory:
# Payment for intermediate node for successful packet transmission
q = 5.0
# Payment for source node for successful packet transmission
m = 30.0

# Check if reliability is ok
if p_other < 0 or p_other > 1:
    raise ValueError(f'Wrong reliability value: {p_other}')

network_info = NetworkInformation(p_other, r_max, e_max, e_min, energy_per_package, beta_coef, gamma_coef, q, m)


# TESTING - VARIABLES

nr_sensors = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
nr_sensors_base = 15

graph_density = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
graph_density_base = 60

nr_packages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nr_packages_base = 15

base_result_folder_name = "results"

base_nr_sensors_folder_name = f"{base_result_folder_name}/nr_sensors"
base_density_folder_name = f"{base_result_folder_name}/density"
base_nr_packages_folder_name = f"{base_result_folder_name}/nr_packages"

single_execution_nr_sensors_folder_name = f"{base_nr_sensors_folder_name}/single"
single_execution_density_folder_name = f"{base_density_folder_name}/single"
single_execution_nr_packages_folder_name = f"{base_nr_packages_folder_name}/single"


# METHODS
graph = generate_graph.generate_real_grah_percentage(10, 40, network_info)
graph.print_graph()
