import math
import random
import networkx as nx
from networkx import NetworkXNoPath

from graph import NetworkInformation, Graph


def path_still_exists(graph: Graph, max_distance: float):
    # Create a networkx graph
    nx_graph = nx.Graph()
    for vertex in graph.vertices:
        nx_graph.add_node(vertex.name)

    for edge in graph.edges:
        # Only add the edge if the edge can be used
        if edge.distance <= max_distance:
            nx_graph.add_edge(edge.point_one.name, edge.point_two.name)

    # Check if a path exists
    try:
        nx.dijkstra_path(nx_graph, graph.vertices[0].name, graph.vertices[len(graph.vertices) - 1].name)

        return True
    except NetworkXNoPath:
        return False
    except Exception:
        raise Exception


def generate_random_complete_graph(nr_of_vertices: int, network_info: NetworkInformation, plane_dim: int) -> Graph:
    # Create the graph
    graph = Graph(network_info)

    # Create an empty map to remember point locations for now
    point_location = {}

    # Create vertices
    for vertex_number in range(nr_of_vertices):
        # We have a 50 x 50 plane, and generate points on it, this makes it easier to calculate distance
        point_location[str(vertex_number)] = [random.randint(0, plane_dim), random.randint(0, plane_dim)]
        graph.add_vertex(str(vertex_number), network_info.p_other)

    # Create edges to get a complete graph
    for vertex in graph.vertices:
        # Add all necessary edges for a complete graph
        for vert in graph.get_vertixes_besides(vertex):
            distance = math.dist(point_location[vertex.name], point_location[vert.name])
            graph.add_edge(vertex, vert, distance)

    return graph

def generate_graph_with_percenteges_edges(nr_of_vertices: int, percents_edges: int, network_info: NetworkInformation, plane_dim: int):
    # Generate a complete graph, that has at least 1 variable path
    while True:
        graph = generate_random_complete_graph(nr_of_vertices, network_info, plane_dim)

        path_exists = path_still_exists(graph, network_info.r_max)
        if path_exists:
            break

    max_nr_edges = len(graph.edges) * (percents_edges / 100)

    nr_of_repeats = 0
    # Get rid of edges until percents_edges% remain
    # This must be done in a way to always have at least 1 way from start to end
    while True:
        # If deleting any edge breaks the path, and the percents are not reached, return None
        if nr_of_repeats > 100:
            print("Exception Raised")
            nr_of_repeats = 0

            raise Exception("Could not create a graph")

        # Pop an edge
        poped_edge = graph.edges.pop(random.randint(0, len(graph.edges) - 1))

        # If a solution from first to last doesn't exist anymore, add it back again and try again
        if not path_still_exists(graph, network_info.r_max):
            graph.edges.append(poped_edge)
            nr_of_repeats += 1

            continue

        if len(graph.edges) < max_nr_edges:
            break

    return graph


def generate_graph(nr_of_vertices: int, percents_edges: int, network_info: NetworkInformation, plane_dim: int) -> Graph:
    # Run the method until no error occurs and a graph is returned
    while True:
        try:
            generated = generate_graph_with_percenteges_edges(nr_of_vertices, percents_edges, network_info, plane_dim)

            generated.print_graph()

            return generated
        except Exception:
            continue
