import math
from enum import Enum

# class EdgeTo:
#     def __init__(self, points_to, distance):
#         self.points_to = points_to
#         self.distance = distance
#
#
# # https://www.educative.io/answers/how-to-implement-a-graph-in-python
#
# # Add a vertex to the dictionary
# def add_vertex(v):
#     global graph
#     if v in graph:
#         print("Vertex ", v, " already exists.")
#     else:
#         graph[v] = []
#
#
# # Add an edge between vertex v1 and v2 with edge weight e
# def add_edge(v1, v2, e):
#     global graph
#     # Check if vertex v1 is a valid vertex
#     if v1 not in graph:
#         print("Vertex ", v1, " does not exist.")
#     # Check if vertex v2 is a valid vertex
#     elif v2 not in graph:
#         print("Vertex ", v2, " does not exist.")
#     else:
#         temp = EdgeTo(v2, e)
#         graph[v1].append(temp)
#
#
# # Print the graph
# def print_graph():
#     global graph
#
#     print('')
#     for vertex in graph:
#         for edges in graph[vertex]:
#             print(vertex, "<-> ", edges.points_to, " distance: ", edges.distance)
#
# def get_all_neighbors_of(v):
#   global graph
#
#   neighboars = []
#
#   neighboars.append(graph[v])
#
#   return neighboars
#
#
#
# # driver code
# graph = {}
#
# add_vertex(start_node_name)
# add_vertex('A')
# add_vertex('B')
# add_vertex('C')
# add_vertex('D')
# add_vertex('E')
# add_vertex('F')
# add_vertex(end_node_name)
# # Add the edges between the vertices by specifying
# # the from and to vertex along with the edge weights.
# add_edge(start_node_name, 'A', 1)
# add_edge('A', 'B', 3)
# add_edge('A', 'C', 3)
# add_edge('B', 'D', 2)
# add_edge('B', 'E', 5.5)
# add_edge('C', 'E', 3)
# add_edge('C', 'F', 3)
# add_edge('D', end_node_name, 4)
# add_edge('E', end_node_name, 3)
# add_edge('F', end_node_name, 6)
# print_graph()
#
# # 1. Sprawdzenie warunku z odległościami - d_i < r.max
# for vertex in graph:
#     for edge in graph[vertex]:
#         if edge.distance > r_max: graph[vertex].remove(edge)
#
# print_graph()
#
#
# print(get_all_neighbors_of('A'))


# DANE WEJŚCIOWE - CONSTANT
p_start_end = 1
t_start_end = math.inf

# DANE WEJŚCIOWE - ZMIENNE

# prawdopodobieństwo, że uda się przesłać
p_other = 0.9

# Odległości, poza którą nie ma komunikacji między elementami (m)
r_max = 5

# Energia
e_max = 20
e_min = 5

# Obciążenie
t_i = 5

# 1. Check if reliability is ok
if p_other < 0 or p_other > 1:
    raise ValueError(f'Wrong reliability value: {p_other}')


class VertexType(Enum):
    TYPICAL = 'sensor'
    START = 'start'
    END = 'end'


class Vertex:
    def __init__(self, name: str):
        self.name = name
        self.current_energy = e_max
        self.min_energy = e_min
        self.reliability = p_other
        self.load_traffic = 0
        self.type = VertexType.TYPICAL

    def make_start(self):
        self.type = VertexType.START

    def make_end(self):
        self.type = VertexType.END

    def reset_to_normal_type(self):
        self.type = VertexType.TYPICAL

    @property
    def reliability(self):
        if self.type == VertexType.TYPICAL:
            return self.reliability
        else:
            return p_start_end

    @reliability.setter
    def reliability(self, reliability):
        self._reliability = reliability

    # Load Traffic
    @property
    def load_traffic(self):
        if self.type == VertexType.TYPICAL:
            return self.load_traffic
        else:
            return t_start_end

    @load_traffic.setter
    def load_traffic(self, load_traffic):
        self._load_traffic = load_traffic

    # Current Energy
    @property
    def current_energy(self):
        if self.type == VertexType.TYPICAL:
            return self.current_energy
        else:
            return e_max

    @current_energy.setter
    def current_energy(self, current_energy):
        self._current_energy = current_energy


class Edge:
    def __init__(self, point_one: Vertex, point_two: Vertex, distance: float):
        self.point_one = point_one
        self.point_two = point_two
        self.distance = distance


class Graph:
    def __init__(self):
        self.vertices: list[Vertex] = []
        self.edges: list[Edge] = []

    def add_vertex(self, name: str):
        if any(vertex.name == name for vertex in self.vertices):
            print(f'{name} already added')
            return

        vertex = Vertex(name)
        self.vertices.append(vertex)

        return vertex

    def add_edge(self, point_one: Vertex, point_two: Vertex, distance: float):
        if not self.vertices.__contains__(point_one):
            print(f'{point_one} does not exist')
            return

        if not self.vertices.__contains__(point_two):
            print(f'{point_two} does not exist')
            return

        # Zabezpieczenie żeby nie było kilka takich samych?
        self.edges.append(Edge(point_one, point_two, distance))

    def print_graph(self):
        print('')
        for edge_element in self.edges:
            print(f'{edge_element.point_one.name} <-> {edge_element.point_two.name}, distance: {edge_element.distance}')

    def get_neighbors_of_vertex(self, vertex: Vertex):
        if not self.vertices.__contains__(vertex):
            print(f'Does not contain {vertex.name}')
            return

        neighbors = []
        for edge_element in self.edges:
            if edge_element.point_one == vertex:
                neighbors.append(edge_element.point_two)

            if edge_element.point_two == vertex:
                neighbors.append(edge_element.point_one)

        return neighbors


graph = Graph()

A = graph.add_vertex('A')
B = graph.add_vertex('B')
C = graph.add_vertex('C')
D = graph.add_vertex('D')
E = graph.add_vertex('E')
F = graph.add_vertex('F')
G = graph.add_vertex('G')

graph.add_edge(A, B, 3)
graph.add_edge(A, C, 3)
graph.add_edge(B, D, 2)
graph.add_edge(B, E, 5.5)
graph.add_edge(C, E, 3)
graph.add_edge(C, F, 3)
graph.add_edge(D, G, 4)
graph.add_edge(E, G, 3)
graph.add_edge(F, G, 6)

graph.print_graph()

# OBLICZENIA

# 2. Sprawdzenie warunku z odległościami - d_i < r.max
for edge in graph.edges:
    if edge.distance > r_max:
        graph.edges.remove(edge)

graph.print_graph()

neighbor_test = list(map(lambda vertex: vertex.name, graph.get_neighbors_of_vertex(A)))
print(neighbor_test)
