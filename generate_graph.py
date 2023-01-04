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


def generate_random_complete_graph(nr_of_vertices: int, network_info: NetworkInformation, plane_dim: float) -> Graph:
    # Create the graph
    graph = Graph(network_info)

    # Create an empty map to remember point locations for now
    point_location = {}

    # Create vertices
    for vertex_number in range(nr_of_vertices):
        # We have a 50 x 50 plane, and generate points on it, this makes it easier to calculate distance
        point_location[str(vertex_number)] = [random.randint(0, int(plane_dim*1000))/1000.0, random.randint(0, int(plane_dim*1000))/1000.0]
        graph.add_vertex(str(vertex_number), network_info.p_other)

    # Create edges to get a complete graph
    for vertex in graph.vertices:
        # Add all necessary edges for a complete graph
        for vert in graph.get_vertixes_besides(vertex):
            distance = math.dist(point_location[vertex.name], point_location[vert.name])
            graph.add_edge(vertex, vert, distance)

    return graph, point_location

def generate_graph_with_percenteges_edges(nr_of_vertices: int, percents_edges: int, network_info: NetworkInformation, plane_dim: float):
    # Generate a complete graph, that has at least 1 variable path
    while True:
        graph, point_location = generate_random_complete_graph(nr_of_vertices, network_info, plane_dim)

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

def calculate_real_graph_percentage(graph: Graph):
    
    # Calculate number of edges of full graph with given size n
    full_graph_edges = 0.0
    n = len(graph.vertices)

    # n*(n-1) for directional graph, otherwise n*(n-1)/2
    full_graph_edges = n*(n-1)/2

    # Calculate how many edges are in given graph (excluding non-existing edges)
    real_edges = 0.0
    r_max = graph.network_info.r_max

    for edge in graph.edges:
        # Only add the edge if the edge can be used
        if edge.distance <= r_max:
            real_edges+=1

    # Calculate real graph edges density percentage based of those two values

    real_percentage = real_edges/full_graph_edges
    return real_percentage

def generate_graph_from_points(nr_of_vertices: int, network_info: NetworkInformation, point_location) -> Graph:
    # Create the graph
    graph = Graph(network_info)

    # Create vertices
    for vertex_number in range(nr_of_vertices):
        graph.add_vertex(str(vertex_number), network_info.p_other)

    # Create edges to get a complete graph
    for vertex in graph.vertices:
        # Add all necessary edges for a complete graph
        for vert in graph.get_vertixes_besides(vertex):
            distance = math.dist(point_location[vertex.name], point_location[vert.name])
            graph.add_edge(vertex, vert, distance)

    return graph

def move_points_from_center(point_location, plane_dim, new_plane_dim):
    new_point_location = {}
    middle_x = plane_dim/2.0
    middle_y = plane_dim/2.0
    new_middle_x = new_plane_dim/2.0
    new_middle_y = new_plane_dim/2.0

    for vertex_number in range(len(point_location)):
        # We have a 50 x 50 plane, and generate points on it, this makes it easier to calculate distance
        new_x = (point_location[str(vertex_number)][0] - middle_x)*(new_plane_dim/plane_dim) + new_middle_x
        new_y = (point_location[str(vertex_number)][1] - middle_y)*(new_plane_dim/plane_dim) + new_middle_y
        new_point_location[str(vertex_number)] = [new_x, new_y]
    
    return new_point_location



def generate_real_grah_percentage(nr_of_vertices: int, percents_edges: int, network_info: NetworkInformation) -> Graph:
    start_plane_size = network_info.r_max/3.0

    plane_change_var = start_plane_size
    current_plane_size = 0

    graph, point_location = generate_random_complete_graph(nr_of_vertices, network_info, start_plane_size)
    current_plane_size = start_plane_size

    mult = 1

    new_graph = None

    for i in range(100):
        current_plane_size = current_plane_size+plane_change_var*mult
        new_point_location = move_points_from_center(point_location, start_plane_size, current_plane_size)
        new_graph = generate_graph_from_points(nr_of_vertices, network_info, new_point_location)

        graph_real_percentage = calculate_real_graph_percentage(new_graph)

        if graph_real_percentage < percents_edges/100.0+0.001 and graph_real_percentage > percents_edges/100.0-0.001:
            break
        if graph_real_percentage*mult < (percents_edges/100.0)*mult:
            mult = mult*-1
            plane_change_var = plane_change_var/2.0

    # remove fake edges from graph
    r_max = network_info.r_max

    edges_to_remove = []

    for edge in new_graph.edges:
        # Only add the edge if the edge can be used
        if edge.distance > r_max:
            edges_to_remove.append(edge)

    for edge in edges_to_remove:
        new_graph.delete_edge(edge)

    print(edges_to_remove)
            
    return new_graph

    

def generate_graph(nr_of_vertices: int, percents_edges: int, network_info: NetworkInformation, plane_dim: float) -> Graph:
    # Run the method until no error occurs and a graph is returned
    while True:
        try:
            generated = generate_graph_with_percenteges_edges(nr_of_vertices, percents_edges, network_info, plane_dim)

            # generated.print_graph()

            return generated
        except Exception:
            continue
