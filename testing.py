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

def path_to_str(path):
    return " -> ".join(map(lambda vertex: vertex.name if vertex is not None else "(None)", path))

def is_path_viable(graph, path):
    starting_vertex = graph.vertices[0]
    ending_vertex = graph.vertices[len(graph.vertices) - 1]

    # Check if algorithm path is viable:
    # first is start vertex, last is end vertex
    # There is no 'None' in teh path
    # Total length of path is higher than 1 (2 at least, start and end)
    return len(path) > 1 and None not in path and starting_vertex.__eq__(path[0]) \
           and ending_vertex.__eq__(path[len(path) - 1])

def test_algorythm(graph: Graph):
    starting_vertex = graph.vertices[0]
    ending_vertex = graph.vertices[len(graph.vertices) - 1]

    # Test Algorythm Solution
    algorythm_time_start = timer()

    algorithm_output = run_algorythm(graph, starting_vertex, ending_vertex)

    algorythm_time_end = timer()

    # Reset the graph
    graph.reset_edges()
    starting_vertex.reset_vertex_type()
    ending_vertex.reset_vertex_type()

    return algorythm_time_end - algorythm_time_start, algorithm_output


def test_solver(graph: Graph):
    starting_vertex = graph.vertices[0]
    ending_vertex = graph.vertices[len(graph.vertices) - 1]

    solver_time_start = timer()

    vertices_to_change, answer, node_matrix = solver_solution(graph, starting_vertex, ending_vertex)
    solver_output = get_path_in_order(graph, node_matrix)

    solver_time_end = timer()

    # Reset the graph
    graph.reset_edges()
    starting_vertex.reset_vertex_type()
    ending_vertex.reset_vertex_type()

    return solver_time_end - solver_time_start, solver_output

def calculate_total_value_from_path(graph: Graph, path):
    if is_path_viable(graph, path):
        values = []

        for index, vertex_from in enumerate(path[:-1]):
            vertex_to = path[index + 1]

            # Multiply p for each vertex on the path till now
            p_k = graph.network_info.p_other ** (index + 1)

            distance_from_to = graph.get_distance(vertex_from, vertex_to)
            if distance_from_to < graph.network_info.r_max:
                cost_distance = distance_from_to * distance_from_to
            else:
                cost_distance = math.inf

            # Cost depending on the remained energy
            if vertex_to.current_energy >= graph.network_info.e_min:
                cost_energy = graph.network_info.e_max / vertex_to.current_energy
            else:
                cost_energy = math.inf

            # Cost depending on load traffic
            traffic_density_to = vertex_to.load_traffic / graph.total_load_traffic if graph.total_load_traffic >= 1 else 1
            cost_load = 1.0 + graph.network_info.gamma_coef * traffic_density_to

            if vertex_from.type == VertexType.START:
                h = len(path)
                mult = graph.network_info.beta_coef / (graph.network_info.m - h * graph.network_info.q)
            else:
                mult = graph.network_info.beta_coef / graph.network_info.q

            total_cost_ij = mult * cost_distance * cost_energy * cost_load

            values.append(p_k - total_cost_ij)

        return sum(values)
    else:
        return -1

def check_and_recreate_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    # Recreate the directory
    os.makedirs(folder_name, exist_ok=True)

def save_single_execution_data_to_file(graph, algorythm_time, algorithm_output, solver_time,
                                       solver_output, file_name):
    algorythm_cost = calculate_total_value_from_path(graph, algorithm_output)
    solver_cost = calculate_total_value_from_path(graph, solver_output)

    with open(file_name, '+a') as file:
        file.write("Graph Information\n")
        for vertex in graph.vertices:
            file.write(f"{vertex.__str__()}\n")
        for edge in graph.edges:
            file.write(f"{edge.__str__()}\n")

        file.write("\n")

        file.write("Algorythm Solution Information\n")
        file.write(f"Time:          {round(algorythm_time, 6)}\n")
        file.write(f"Is Viable?:    {is_path_viable(graph, algorithm_output)}\n")
        file.write(f"Path:          {path_to_str(algorithm_output)}\n")
        file.write(f"Cost:          {round(algorythm_cost, 6)}\n")

        file.write("\n")

        file.write("Solver Solution Information\n")
        file.write(f"Time:          {round(solver_time, 6)}\n")
        file.write(f"Is Viable?:    {is_path_viable(graph, solver_output)}\n")
        file.write(f"Path:          {path_to_str(solver_output)}\n")
        file.write(f"Cost:          {round(solver_cost, 6)}\n")




# TESTING - CREATE FLDER STRUCTURE
check_and_recreate_folder(single_execution_nr_sensors_folder_name)
check_and_recreate_folder(single_execution_density_folder_name)
check_and_recreate_folder(single_execution_nr_packages_folder_name)

# TESTING - NUMBER OF SENSORS

for sensors in nr_sensors:
    # Do this 10 times
    for test_nr in range(10):
        # Create Graph
        graph = generate_graph.generate_real_grah_percentage(sensors, graph_density_base, network_info)

        test_solver_time = []
        test_algorythm_time = []
        test_solver_output = []
        test_algorythm_output = []

        for package in range(nr_packages_base):
            # Test Algorythm Solution
            algorythm_time, algorithm_output = test_algorythm(graph)

            # Test Solver Solution
            solver_time, solver_output = test_solver(graph)

            # Save single test solutions
            save_single_execution_data_to_file(graph, algorythm_time, algorithm_output, solver_time, solver_output,
                                               f"{single_execution_nr_sensors_folder_name}/{sensors}-{test_nr}-{package + 1}.txt")

            test_algorythm_output.append(calculate_total_value_from_path(graph, algorithm_output))
            test_solver_output.append(calculate_total_value_from_path(graph, solver_output))

            # For every vertex in both results (if there are duplicates, they are done 2 times)
            for vertex in algorithm_output + solver_output:
                if vertex is not None:
                    if vertex.type != VertexType.START and vertex.type != VertexType.END:
                        vertex.current_energy = vertex.current_energy - energy_per_package
                        if vertex.current_energy == vertex.min_energy:
                            vertex.current_energy = 0.000001
                    vertex.load_traffic += 1

            print(f"{sensors} - {test_nr} - {package} finished")



# # TESTING - DENSITY OF CONNECTIONS
#
# for density in graph_density:
#     # Do this 15 times
#     for package in range(nr_packages_base):
#         # Create Graph
#         graph = generate_graph.generate_real_grah_percentage(nr_packages_base, density, network_info)
#
#
#
# # TESTING - NR OF PACKAGES SEND
#
# for n_package in nr_packages:
#     # Do this X times, where x is in table
#     for package in range(n_package):
#         # Create Graph
#         graph = generate_graph.generate_real_grah_percentage(nr_packages_base, graph_density_base, network_info)
