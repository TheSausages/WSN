import math
from graph import Vertex, Graph, VertexType
import networkx as nx

# INPUT DATA  - VARIABLES

# Probability for successfully sending
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
        mult = beta_coef / q
    else:
        mult = beta_coef / q

    total_cost_ij = mult * cost_distance * cost_energy * cost_load
    return total_cost_ij

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

    # we create it in the loop, to reset the weigths for the edges (so the cost)
    Gr = nx.Graph()

    for vertex in graph.vertices:
        Gr.add_node(vertex.name)

    for edge in graph.edges:
        Gr.add_edge(edge.point_one.name, edge.point_two.name,
                    weight=calculate_cost_function(graph, edge.point_one, edge.point_two, {}))

    starting_vertex.make_start_vertex()
    ending_vertex.make_end_vertex()

    out = nx.shortest_path(Gr, source=A.name, target=G.name, weight="weight")
    print(out)

    graph.total_load_traffic += 1
    for vertex_name in out:
        vertex = [vertex for vertex  in graph.vertices if vertex.name == vertex_name][0]

        if vertex.type == VertexType.TYPICAL or vertex.type == VertexType.START:
            # For Start Type it will always return max energy anyway
            vertex.current_energy = vertex.current_energy - energy_per_package
            vertex.load_traffic += 1

    starting_vertex.reset_vertex_type()
    ending_vertex.reset_vertex_type()

    print('')