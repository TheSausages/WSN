import math

# INPUT DATA  - VARIABLES

# Probability for successfully sending
from graph import Vertex, Graph, VertexType

p_other = 0.9

# Distance, above which communication is impossible
r_max = 5

# Max and Min energies
e_max = 20
e_min = 5

# Energy lost per Package
energy_per_package = 2

# Mathematical coeficients
beta_coef = 0.05
gamma_coef = 0.2

# Game theory:
# Payment for intermediate node for successful packet transmission
q = 5.0
# Payment for source node for successful packet transmission
m = 30.0

# 1. Check if reliability is ok
if p_other < 0 or p_other > 1:
    raise ValueError(f'Wrong reliability value: {p_other}')


def get_number_of_vertexes_in_path(previous: dict, previous_candidate: Vertex):
    nr_of_previous = 0
    prev = previous_candidate

    while True:
        prev = previous[prev]

        # We only increase when the previous is not the end vertex
        if prev not in previous:
            break

        nr_of_previous += 1

    return nr_of_previous


# graph - the network graph we use
# vertex_i - the one who sends
# vertex_j - the one who receives
# previous - dict with previous vertexes for a given vertexes
def calculate_cost_function(graph: Graph, vertex_i: Vertex, vertex_j: Vertex, previous: dict):
    # print(f'{vertex_i.name} -> {vertex_j.name}')

    # Cost depending on distance
    distance_ij = graph.get_distance(vertex_i, vertex_j)
    if distance_ij < r_max:
        cost_distance = distance_ij * distance_ij
    else:
        cost_distance = math.inf

    # Cost depending on the remained energy
    if vertex_j.current_energy >= e_min:
        cost_energy = e_max / vertex_j.current_energy
    else:
        cost_energy = math.inf

    # Cost depending on load traffic
    traffic_density_j = vertex_j.load_traffic / graph.total_load_traffic if graph.total_load_traffic >= 1 else 1
    cost_load = 1.0 + gamma_coef * traffic_density_j

    if vertex_i.type == VertexType.START:
        h = get_number_of_vertexes_in_path(previous, vertex_j)
        mult = beta_coef / (m - h * q)
    else:
        mult = beta_coef / q

    total_cost_ij = mult * cost_distance * cost_energy * cost_load
    return total_cost_ij


def run_algorythm(graph: Graph, starting_vertex: Vertex, ending_vertex: Vertex):
    # Begin

    # 1. Set types for vertexes and copy the graph
    starting_vertex.make_start_vertex()
    ending_vertex.make_end_vertex()
    original_vertexes = list(graph.vertices)
    original_edges = list(graph.edges)

    # 2. Initialise necessary elements:
    #       - Create the empty set of labeled nodes: W
    #       - Create dict of costs: L
    #       - Create dict of earnings for each vertex: M
    #       - Create disc for previous vertexes: Previous
    W = []
    L = {
        ending_vertex: 0,
    }
    M = {
        ending_vertex: 1
    }
    Previous = {}

    # 3. Initialization Loop
    for vertex in graph.get_vertixes_besides(ending_vertex):
        L[vertex] = math.inf
        Previous[vertex] = None
        M[vertex] = 1

    # 4. Main Loop
    # Because we need it to enter at least once, we use a do while
    while True:
        # Select next vertex to analyse
        vertex_j = None
        for vertex in graph.vertices:
            if all(L[vertex] <= L[vertex_k] for vertex_k in graph.get_vertixes_besides(vertex)):
                vertex_j = vertex

        # Remove it from the graph and add it to the set of empty nodes
        graph.vertices.remove(vertex_j)
        W.append(vertex_j)

        # Go thought it's neighbors
        for vertex_i in graph.get_neighbors_of_vertex(vertex_j):
            # Calculate the costs
            # x = L(v_j) + C_ij
            c_ij = calculate_cost_function(graph, vertex_i, vertex_j, Previous)
            x = L[vertex_j] + c_ij

            # Check if the current costs are higher than existing costs
            if L[vertex_i] > x:
                # L(vi) = x
                L[vertex_i] = x

                # M(vi) = p_i * M(vj)
                M[vertex_i] = p_other * M[vertex_j]

                if (M[vertex_i] - c_ij) < 0:
                    # delete edge (vi, vj) from E
                    graph.delete_edge(vertex_i, vertex_j)
                else:
                    # previous(vi) = vj
                    Previous[vertex_i] = vertex_j

        # Ending condition
        if not ((not W.__contains__(starting_vertex)) and len(graph.get_neighbors_of_vertexes(W)) != 0):
            break

    # End

    # Get the optimal path from the Previous dict
    path = []
    try:
        path = [starting_vertex]
        next_hop = Previous[starting_vertex]
        path.append(next_hop)
        while True:
            next_hop = Previous[next_hop]
            path.append(next_hop)

            if next_hop not in Previous.keys():
                break
    except:
        print(f'Could not find any path to send data')

    # After - Return to the original values
    graph.edges = original_edges
    graph.vertices = original_vertexes

    return path


# PRZYKŁADOWE OBLICZENIA

graph = Graph()

A = graph.add_vertex('A', e_max, e_min, p_other)
B = graph.add_vertex('B', e_max, e_min, p_other)
C = graph.add_vertex('C', e_max, e_min, p_other)
D = graph.add_vertex('D', e_max, e_min, p_other)
E = graph.add_vertex('E', e_max, e_min, p_other)
F = graph.add_vertex('F', e_max, e_min, p_other)
G = graph.add_vertex('G', e_max, e_min, p_other)

graph.add_edge(A, B, 3)
graph.add_edge(A, C, 3)
graph.add_edge(B, D, 2)
graph.add_edge(B, E, 5.5)
graph.add_edge(C, E, 3)
graph.add_edge(C, F, 3)
graph.add_edge(D, G, 4)
graph.add_edge(E, G, 3)
graph.add_edge(F, G, 6)

print('Created Graph')
graph.print_graph()

for package in range(0, 10):
    starting_vertex = A
    ending_vertex = G

    out = run_algorythm(graph, starting_vertex, ending_vertex)

    print(f'Round {package + 1} finished with path:')

    last_element = out[-1]
    for path_element in out:
        if path_element == last_element:
            print(f'{path_element.name}')
        else:
            print(f'{path_element.name} -> ', end='')

    graph.total_load_traffic += 1
    for vertex in out:
        if vertex.type == VertexType.TYPICAL or vertex.type == VertexType.START:
            # For Start Type it will always return max energy anyway
            vertex.current_energy = vertex.current_energy - energy_per_package
            vertex.load_traffic += 1

    starting_vertex.reset_vertex_type()
    ending_vertex.reset_vertex_type()

    print('')
