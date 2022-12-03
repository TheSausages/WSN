import math
from graph import Vertex, Graph, VertexType
import networkx as nx

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

def solver_solution(graph: Graph, starting_vertex: Vertex, ending_vertex: Vertex):
    # we create it in the loop, to reset the weigths for the edges (so the cost)
    graph_nx = nx.Graph()

    for vertex in graph.vertices:
        graph_nx.add_node(vertex.name)

    for edge in graph.edges:
        graph_nx.add_edge(edge.point_one.name, edge.point_two.name,
                    weight=calculate_cost_function(graph, edge.point_one, edge.point_two, {}))

    starting_vertex.make_start_vertex()
    ending_vertex.make_end_vertex()

    return nx.shortest_path(graph_nx, source=starting_vertex.name, target=ending_vertex.name, weight="weight")