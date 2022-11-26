import math
from collections import namedtuple

# DANE WEJŚCIOWE
start_node_name = 'start'
end_node_name = 'end'


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

# prawdopodobieństwo, że uda się przesłać
p_start_end = 1
p_other = 0.9

# Odległości, poza którą nie ma komunikacji między elementami (m)
r_max = 5

# Energia
e_max = 20
e_start_min = e_max
e_min = 5

# Obciążenie
t_i = 5
t_start_end = math.inf



class Vertex:
    def __init__(self, name: str):
        self.name = name
        self.current_energy = e_max


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
        else:
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

Start = graph.add_vertex(start_node_name)
A = graph.add_vertex('A')
B = graph.add_vertex('B')
C = graph.add_vertex('C')
D = graph.add_vertex('D')
E = graph.add_vertex('E')
F = graph.add_vertex('F')
End = graph.add_vertex(end_node_name)

graph.add_edge(Start, A, 1)
graph.add_edge(A, B, 3)
graph.add_edge(A, C, 3)
graph.add_edge(B, D, 2)
graph.add_edge(B, E, 5.5)
graph.add_edge(C, E, 3)
graph.add_edge(C, F, 3)
graph.add_edge(D, End, 4)
graph.add_edge(E, End, 3)
graph.add_edge(F, End, 6)

graph.print_graph()

# OBLICZENIA

# 1. Sprawdzenie warunku z odległościami - d_i < r.max
for edge in graph.edges:
    if edge.distance > r_max:
        graph.edges.remove(edge)

graph.print_graph()

neighbor_test = list(map(lambda vertex: vertex.name, graph.get_neighbors_of_vertex(A)))
print(neighbor_test)
