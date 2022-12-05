import math

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import ortools.graph
from graph import Vertex, Graph, NetworkInformation, VertexType
from ortools.graph.python import min_cost_flow

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

# Check if reliability is ok
if p_other < 0 or p_other > 1:
    raise ValueError(f'Wrong reliability value: {p_other}')

network_info = NetworkInformation(p_other, r_max, e_max, e_min, energy_per_package, beta_coef, gamma_coef, q, m)

graph = Graph(network_info)

A = graph.add_vertex('A', p_other)
B = graph.add_vertex('B', p_other)
C = graph.add_vertex('C', p_other)
D = graph.add_vertex('D', p_other)
E = graph.add_vertex('E', p_other)
F = graph.add_vertex('F', p_other)
G = graph.add_vertex('G', p_other)

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

starting_vertex = A
ending_vertex = G

starting_vertex.make_start_vertex()
ending_vertex.make_end_vertex()

"""MinCostFlow simple interface example."""
# Instantiate a SimpleMinCostFlow solver.
smcf = min_cost_flow.SimpleMinCostFlow()

# # Define four parallel arrays: sources, destinations, capacities,
# # and unit costs between each pair. For instance, the arc from node 0
# # to node 1 has a capacity of 15.
# start_nodes = np.array([0])
# end_nodes = np.array([4])
# capacities = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
# unit_costs = np.array([4, 4, 2, 2, 6, 1, 3, 2, 3])
#
# # Define an array of supplies at each node.
# supplies = [0, 0, 0, 0, 1]
#
# # Add arcs, capacities and costs in bulk using numpy.
# all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
#     start_nodes, end_nodes, capacities, unit_costs)

def calculate_cost_function(graph: Graph, vertex_i: Vertex, vertex_j: Vertex, previous: dict):
    # print(f'{vertex_i.name} -> {vertex_j.name}')

    # Cost depending on distance
    distance_ij = graph.get_distance(vertex_i, vertex_j)
    if distance_ij < graph.network_info.r_max:
        cost_distance = distance_ij * distance_ij
    else:
        cost_distance = math.inf

    # Cost depending on the remained energy
    if vertex_j.current_energy >= graph.network_info.e_min:
        cost_energy = graph.network_info.e_max / vertex_j.current_energy
    else:
        cost_energy = math.inf

    # Cost depending on load traffic
    traffic_density_j = vertex_j.load_traffic / graph.total_load_traffic if graph.total_load_traffic >= 1 else 1
    cost_load = 1.0 + graph.network_info.gamma_coef * traffic_density_j

    if vertex_i.type == VertexType.START:
        mult = graph.network_info.beta_coef / graph.network_info.q
    else:
        mult = graph.network_info.beta_coef / graph.network_info.q

    total_cost_ij = mult * cost_distance * cost_energy * cost_load
    return total_cost_ij

arcs = []
unit_costs = []
costs = []
for edge in graph.edges:
    # we need int, so we multiply by 100 to get better differences
    print( calculate_cost_function(graph, edge.point_one, edge.point_two, {}))
    print(calculate_cost_function(graph, edge.point_one, edge.point_two, {}) * 1000)
    cost_float = calculate_cost_function(graph, edge.point_one, edge.point_two, {}) * 1000
    cost = 1000000 if cost_float == math.inf else int(cost_float)
    unit_costs.append(cost)
    arcs.append(smcf.add_arc_with_capacity_and_unit_cost(graph.vertices.index(edge.point_one), graph.vertices.index(edge.point_two), 1, cost))

smcf.set_node_supply(graph.vertices.index(starting_vertex), 1)
smcf.set_node_supply(graph.vertices.index(ending_vertex), -1)

# Find the min cost flow.
status = smcf.solve()

if status != smcf.OPTIMAL:
    print('There was an issue with the min cost flow input.')
    print(f'Status: {status}')
    exit(1)
print(f'Minimum cost: {smcf.optimal_cost()}')
print('')
print(' Arc         Cost')
solution_flows = smcf.flows(arcs)
costs = solution_flows * unit_costs
for arc, flow, cost in zip(arcs, solution_flows, costs):
    if flow == 1:
        print(
            f'{graph.vertices[smcf.tail(arc)].name} -> {graph.vertices[smcf.head(arc)].name}       {cost / 1000}'
        )
