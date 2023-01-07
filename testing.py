import math
import os
import shutil

import pandas as pd
import generate_graph
from graph import NetworkInformation, Graph, VertexType, Vertex
from timeit import default_timer as timer

from solver import solver_solution, get_path_in_order
from wsn_algorithm import run_algorythm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# INPUT DATA  - VARIABLES

# Probability for successfully sending

p_other = 0.9

# Distance, above which communication is impossible
r_max = 25

# Max and Min energies
e_max = 25
e_min = 5

# Energy lost per Package
energy_per_package = 2

# Mathematical coeficients
beta_coef = 0.000003
gamma_coef = 0.00015

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

# nr_sensors = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
nr_sensors = [5, 7, 9]
nr_sensors_base = 15

graph_density = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
graph_density_base = 60

nr_packages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nr_packages_base = 15

nr_tests = 10

base_result_folder_name = "results"

base_nr_sensors_folder_name = f"{base_result_folder_name}/nr_sensors"
base_density_folder_name = f"{base_result_folder_name}/density"
base_nr_packages_folder_name = f"{base_result_folder_name}/nr_packages"

single_execution_nr_sensors_folder_name = f"{base_nr_sensors_folder_name}/single"
single_execution_density_folder_name = f"{base_density_folder_name}/single"
single_execution_nr_packages_folder_name = f"{base_nr_packages_folder_name}/single"

single_test_nr_sensors_folder_name = f"{base_nr_sensors_folder_name}/single_test"
single_test_density_folder_name = f"{base_density_folder_name}/single_test"
single_test_nr_packages_folder_name = f"{base_nr_packages_folder_name}/single_test"


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

    # If no solution is found, return an empty response
    try:
        vertices_to_change, answer, node_matrix = solver_solution(graph, starting_vertex, ending_vertex)
        solver_output = get_path_in_order(graph, node_matrix)
    except Exception:
        solver_output = []

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
            if distance_from_to <= graph.network_info.r_max:
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


def save_single_test_data_to_file(graph, algorythm_time, algorithm_output, solver_time,
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


def save_single_test_results(results: [pd.DataFrame], file_name: str):
    df = pd.DataFrame(columns=['path_found', 'path_cost', 'time', 'package'])
    for res in results:
        df = pd.concat([df, res])

    df.to_csv(file_name)


def create_dataframe_for_single_test_result(data: [pd.DataFrame], sensors, density, packages):
    how_many_found = len(list(filter(lambda res: res['path_found'].bool() == True, data)))
    return pd.DataFrame({
        'nr_path_found': how_many_found,
        'nr_path_all': len(data),
        'found_percent': round(how_many_found  / len(data), 4),
        'path_cost_mean': pd.Series(map(lambda res: res['path_cost'], list(filter(lambda res: res['path_cost'].item() > 0, data)))).mean(),
        'time_mean': pd.Series(map(lambda res: res['time'], data)).mean(),
        'sensors': sensors,
        'density': density,
        'packages': packages
    }, index=[0])

def create_dataframe_for_test_result(data: [pd.DataFrame]):
    # print(list(filter(lambda res: res['path_cost_mean'].item() > 0, data)))
    return pd.DataFrame({
        'found_percent_mean': pd.Series(map(lambda res: res['found_percent'], data)).mean(),
        'path_cost_mean': pd.Series(map(lambda res: res['path_cost_mean'], list(filter(lambda res: res['path_cost_mean'].item() > 0, data)))).mean(),
        'time_mean': pd.Series(map(lambda res: res['time_mean'], data)).mean(),
        'sensors': data[0]['sensors'],
        'density': data[0]['density'],
        'packages': data[0]['packages']
    }, index=[0])


# TESTING - CREATE FOLDER STRUCTURE

# check_and_recreate_folder(single_execution_nr_sensors_folder_name)
# check_and_recreate_folder(single_execution_density_folder_name)
# check_and_recreate_folder(single_execution_nr_packages_folder_name)
#
# check_and_recreate_folder(single_test_nr_sensors_folder_name)
# check_and_recreate_folder(single_test_density_folder_name)
# check_and_recreate_folder(single_test_nr_packages_folder_name)

# TESTING - NUMBER OF SENSORS

# Create dataframes for results
sensors_algorythm_df = pd.DataFrame(
    columns=['found_percent_mean', 'path_cost_mean', 'time_mean', 'sensors', 'density', 'packages'])
sensors_solver_df = pd.DataFrame(
    columns=['found_percent_mean', 'path_cost_mean', 'time_mean', 'sensors', 'density', 'packages'])

for sensors in nr_sensors:
    test_algorythm_result = []
    test_solver_result = []

    # Run each test 10 times
    for test_nr in range(nr_tests):
        # Create Graph
        graph = generate_graph.generate_real_grah_percentage(sensors, graph_density_base, network_info)

        package_algorythm_result = []
        package_solver_result = []

        for package in range(nr_packages_base):
            print(f"{sensors} - {test_nr} - {package} started")

            # Test Algorythm Solution
            algorythm_time, algorithm_output = test_algorythm(graph)

            # Test Solver Solution
            solver_time, solver_output = test_solver(graph)

            # Save single test solutions
            save_single_test_data_to_file(graph, algorythm_time, algorithm_output, solver_time, solver_output,
                                          f"{single_execution_nr_sensors_folder_name}/{sensors}-{test_nr}-{package}.txt")

            # Save the results for both algorythm and solver
            package_algorythm_result.append(
                pd.DataFrame({'path_found': is_path_viable(graph, algorithm_output),
                                        'path_cost': calculate_total_value_from_path(graph, algorithm_output),
                                        'time': algorythm_time, 'package': package}, index=[0]))
            package_solver_result.append(
                pd.DataFrame({'path_found': is_path_viable(graph, solver_output),
                                        'path_cost': calculate_total_value_from_path(graph, solver_output),
                                        'time': solver_time, 'package': package}, index=[0]))

            # For every vertex in both results (if there are duplicates, they are done 2 times)
            for vertex in algorithm_output + solver_output:
                if vertex is not None:
                    if vertex.type != VertexType.START and vertex.type != VertexType.END:
                        vertex.current_energy = vertex.current_energy - energy_per_package
                        if vertex.current_energy == vertex.min_energy:
                            vertex.current_energy = 0.000001
                    vertex.load_traffic += 1

            print(f"{sensors} - {test_nr} - {package} finished")

        # Save the single test result for both solutions
        save_single_test_results(package_algorythm_result,
                                 f'{single_test_nr_sensors_folder_name}/{sensors}_{test_nr}_algorythm.csv')
        save_single_test_results(package_solver_result,
                                 f'{single_test_nr_sensors_folder_name}/{sensors}_{test_nr}_solver.csv')

        # Save the results to the list
        test_algorythm_result.append(create_dataframe_for_single_test_result(package_algorythm_result, sensors, graph_density_base, nr_packages_base))
        test_solver_result.append(create_dataframe_for_single_test_result(package_solver_result, sensors, graph_density_base, nr_packages_base))

    # Save the mean results for all 10 test
    sensors_algorythm_df = pd.concat([sensors_algorythm_df, create_dataframe_for_test_result(test_algorythm_result)])
    sensors_solver_df = pd.concat([sensors_solver_df, create_dataframe_for_test_result(test_solver_result)])

# Save the results
sensors_algorythm_df.to_csv(f"{base_nr_sensors_folder_name}/algorythm.csv", mode='a')
sensors_solver_df.to_csv(f"{base_nr_sensors_folder_name}/solver.csv", mode='a')
#
# # TESTING - DENSITY OF CONNECTIONS
#
# # Create dataframes for results
# density_algorythm_df = pd.DataFrame(
#     columns=['found_percent_mean', 'path_cost_mean', 'time_mean', 'sensors', 'density', 'packages'])
# density_solver_df = pd.DataFrame(
#     columns=['found_percent_mean', 'path_cost_mean', 'time_mean', 'sensors', 'density', 'packages'])
#
# for density in graph_density:
#     test_algorythm_result = []
#     test_solver_result = []
#
#     # Run each test 10 times
#     for test_nr in range(nr_tests):
#         # Create Graph
#         graph = generate_graph.generate_real_grah_percentage(nr_packages_base, density, network_info)
#
#         package_algorythm_result = []
#         package_solver_result = []
#
#         for package in range(nr_packages_base):
#             # Test Algorythm Solution
#             algorythm_time, algorithm_output = test_algorythm(graph)
#
#             # Test Solver Solution
#             solver_time, solver_output = test_solver(graph)
#
#             # Save single test solutions
#             save_single_test_data_to_file(graph, algorythm_time, algorithm_output, solver_time, solver_output,
#                                           f"{single_execution_density_folder_name}/{density}-{test_nr}-{package}.txt")
#
#             # Save the results for both algorythm and solver
#             package_algorythm_result.append(
#                 pd.DataFrame({'path_found': is_path_viable(graph, algorithm_output),
#                               'path_cost': calculate_total_value_from_path(graph, algorithm_output),
#                               'time': algorythm_time, 'package': package}, index=[0]))
#             package_solver_result.append(
#                 pd.DataFrame({'path_found': is_path_viable(graph, solver_output),
#                               'path_cost': calculate_total_value_from_path(graph, solver_output),
#                               'time': solver_time, 'package': package}, index=[0]))
#
#             # For every vertex in both results (if there are duplicates, they are done 2 times)
#             for vertex in algorithm_output + solver_output:
#                 if vertex is not None:
#                     if vertex.type != VertexType.START and vertex.type != VertexType.END:
#                         vertex.current_energy = vertex.current_energy - energy_per_package
#                         if vertex.current_energy == vertex.min_energy:
#                             vertex.current_energy = 0.000001
#                     vertex.load_traffic += 1
#
#             print(f"{density} - {test_nr} - {package} finished")
#
#         # Save the single test result for both solutions
#         save_single_test_results(package_algorythm_result,
#                                  f'{single_test_density_folder_name}/{density}_{test_nr}_algorythm.csv')
#         save_single_test_results(package_solver_result,
#                                  f'{single_test_density_folder_name}/{density}_{test_nr}_solver.csv')
#
#         # Save the results to the list
#         test_algorythm_result.append(create_dataframe_for_single_test_result(package_algorythm_result, nr_sensors_base, density, nr_packages_base))
#         test_solver_result.append(create_dataframe_for_single_test_result(package_solver_result, nr_sensors_base, density, nr_packages_base))
#
#     # Save the mean results for all 10 test
#     density_algorythm_df = pd.concat([density_algorythm_df, create_dataframe_for_test_result(test_algorythm_result)])
#     density_solver_df = pd.concat([density_solver_df, create_dataframe_for_test_result(test_solver_result)])
#
# # Save the results
# density_algorythm_df.to_csv(f"{base_density_folder_name}/algorythm.csv")
# density_solver_df.to_csv(f"{base_density_folder_name}/solver.csv")
#
#
# # TESTING - NR OF PACKAGES SEND
#
# # Create dataframes for results
# package_algorythm_df = pd.DataFrame(
#     columns=['found_percent_mean', 'path_cost_mean', 'time_mean', 'sensors', 'density', 'packages'])
# package_solver_df = pd.DataFrame(
#     columns=['found_percent_mean', 'path_cost_mean', 'time_mean', 'sensors', 'density', 'packages'])
#
# for n_package in nr_packages:
#     test_algorythm_result = []
#     test_solver_result = []
#
#     # Run each test 10 times
#     for test_nr in range(nr_tests):
#         # Create Graph
#         graph = generate_graph.generate_real_grah_percentage(nr_packages_base, graph_density_base, network_info)
#
#         package_algorythm_result = []
#         package_solver_result = []
#
#         for package in range(n_package):
#             # Test Algorythm Solution
#             algorythm_time, algorithm_output = test_algorythm(graph)
#
#             # Test Solver Solution
#             solver_time, solver_output = test_solver(graph)
#
#             # Save single test solutions
#             save_single_test_data_to_file(graph, algorythm_time, algorithm_output, solver_time, solver_output,
#                                           f"{single_execution_nr_packages_folder_name}/{n_package}-{test_nr}-{package}.txt")
#
#             # Save the results for both algorythm and solver
#             package_algorythm_result.append(
#                 pd.DataFrame({'path_found': is_path_viable(graph, algorithm_output),
#                               'path_cost': calculate_total_value_from_path(graph, algorithm_output),
#                               'time': algorythm_time, 'package': package}, index=[0]))
#             package_solver_result.append(
#                 pd.DataFrame({'path_found': is_path_viable(graph, solver_output),
#                               'path_cost': calculate_total_value_from_path(graph, solver_output),
#                               'time': solver_time, 'package': package}, index=[0]))
#
#             # For every vertex in both results (if there are duplicates, they are done 2 times)
#             for vertex in algorithm_output + solver_output:
#                 if vertex is not None:
#                     if vertex.type != VertexType.START and vertex.type != VertexType.END:
#                         vertex.current_energy = vertex.current_energy - energy_per_package
#                         if vertex.current_energy == vertex.min_energy:
#                             vertex.current_energy = 0.000001
#                     vertex.load_traffic += 1
#
#             print(f"{n_package} - {test_nr} - {package} finished")
#
#         # Save the single test result for both solutions
#         save_single_test_results(package_algorythm_result,
#                                  f'{single_test_nr_packages_folder_name}/{n_package}_{test_nr}_algorythm.csv')
#         save_single_test_results(package_solver_result,
#                                  f'{single_test_nr_packages_folder_name}/{n_package}_{test_nr}_solver.csv')
#
#         # Save the results to the list
#         test_algorythm_result.append(create_dataframe_for_single_test_result(package_algorythm_result, nr_sensors_base, graph_density_base, n_package))
#         test_solver_result.append(create_dataframe_for_single_test_result(package_solver_result, nr_sensors_base, graph_density_base, n_package))
#
#     # Save the mean results for all 10 test
#     package_algorythm_df = pd.concat([package_algorythm_df, create_dataframe_for_test_result(test_algorythm_result)])
#     package_solver_df = pd.concat([package_solver_df, create_dataframe_for_test_result(test_solver_result)])
#
# # Save the results
# package_algorythm_df.to_csv(f"{base_nr_packages_folder_name}/algorythm.csv")
# package_solver_df.to_csv(f"{base_nr_packages_folder_name}/solver.csv")
