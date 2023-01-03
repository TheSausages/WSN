import generate_graph
from graph import NetworkInformation, Vertex, Graph
from timeit import default_timer as timer

from solver import solver_solution, get_path_in_order
from wsn_algorithm import run_algorythm


# METHODS

def path_to_str(path: [Vertex]):
    return " -> ".join(map(lambda vertex: vertex.name, path))

def test_algorythm(graph: Graph):
    starting_vertex = graph.vertices[0]
    ending_vertex = graph.vertices[len(graph.vertices) - 1]

    # Test Algorythm Solution
    algorythm_time_start = timer()

    algorithm_output = run_algorythm(graph, starting_vertex, ending_vertex)

    # Check if algorithm path is viable - first is start, last is end
    is_algorythm_pack_variable = len(algorithm_output) > 0 and starting_vertex.__eq__(
        algorithm_output[0]) and ending_vertex.__eq__(algorithm_output[len(algorithm_output) - 1])

    algorythm_time_end = timer()

    # Reset the graph
    graph.reset()

    return algorythm_time_end - algorythm_time_start, algorithm_output, is_algorythm_pack_variable

def test_solver(graph: Graph):
    starting_vertex = graph.vertices[0]
    ending_vertex = graph.vertices[len(graph.vertices) - 1]

    solver_time_start = timer()

    vertices_to_change, answer, node_matrix = solver_solution(graph, starting_vertex, ending_vertex)
    solver_output = get_path_in_order(graph, node_matrix)

    solver_time_end = timer()

    # Reset the graph
    graph.reset()

    return solver_time_end - solver_time_start, solver_output


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



# TESTING - NUMBER OF SENSORS

for sensors in nr_sensors:
    # Do this 15 times
    for package in range(nr_packages_base):
        # Create Graph
        graph = generate_graph.generate_real_grah_percentage(sensors, graph_density_base, network_info)

        # Test Algorythm Solution
        algorythm_time, algorithm_output, is_algorythm_pack_variable = test_algorythm(graph)

        # Test Solver Solution
        solver_time, solver_output = test_solver(graph)

        print(f"{package + 1}.")
        print(f"Algorythm: time: {round(algorythm_time, 5)}, path: {path_to_str(algorithm_output) if is_algorythm_pack_variable else ''}, path found: {is_algorythm_pack_variable}")
        print(f"Solver:    time: {round(solver_time, 5)}, path: {path_to_str(solver_output)}")


# TESTING - DENSITY OF CONNECTIONS

for density in graph_density:
    # Do this 15 times
    for package in range(nr_packages_base):
        # Create Graph
        graph = generate_graph.generate_real_grah_percentage(nr_packages_base, density, network_info)



# TESTING - NR OF PACKAGES SEND

for n_package in nr_packages:
    # Do this X times, where x is in table
    for package in range(n_package):
        # Create Graph
        graph = generate_graph.generate_real_grah_percentage(nr_packages_base, graph_density_base, network_info)
