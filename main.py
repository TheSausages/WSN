import math
from enum import Enum


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

    def make_start_vertex(self):
        self.type = VertexType.START

    def make_end_vertex(self):
        self.type = VertexType.END

    def reset_vertex_type(self):
        self.type = VertexType.TYPICAL

    @property
    def reliability(self):
        if self.type == VertexType.TYPICAL:
            return self._reliability
        else:
            return p_start_end

    @reliability.setter
    def reliability(self, reliability):
        self._reliability = reliability

    # Load Traffic
    @property
    def load_traffic(self):
        if self.type == VertexType.TYPICAL:
            return self._load_traffic
        else:
            return t_start_end

    @load_traffic.setter
    def load_traffic(self, load_traffic):
        self._load_traffic = load_traffic

    # Current Energy
    @property
    def current_energy(self):
        if self.type == VertexType.TYPICAL:
            return self._current_energy
        else:
            return e_max

    @current_energy.setter
    def current_energy(self, current_energy):
        self._current_energy = current_energy

    def __str__(self):
        return f'{self.name}: {self.type}, {self.reliability}, {self.load_traffic}, {self.current_energy}'


class Edge:
    def __init__(self, point_one: Vertex, point_two: Vertex, distance: float):
        self.point_one = point_one
        self.point_two = point_two
        self.distance = distance

    def __str__(self):
        return f'{self.point_one.name} <-> {self.point_two.name}, distance: {self.distance}'


class Graph:
    def __init__(self, vertexes=[], edges=[]):
        self.vertices: list[Vertex] = vertexes
        self.edges: list[Edge] = edges

    def add_vertex(self, name: str) -> Vertex:
        if any(vertex.name == name for vertex in self.vertices):
            raise ValueError(f'{name} already added')

        vertex = Vertex(name)
        self.vertices.append(vertex)

        return vertex

    def add_edge(self, point_one: Vertex, point_two: Vertex, distance: float):
        if not self.vertices.__contains__(point_one):
            raise ValueError(f'{point_one} does not exist')

        if not self.vertices.__contains__(point_two):
            raise ValueError(f'{point_two} does not exist')

        # Zabezpieczenie żeby nie było kilka takich samych?
        self.edges.append(Edge(point_one, point_two, distance))

    def print_graph(self):
        for edge_element in self.edges:
            print(edge_element.__str__())
        print('\n')

    def get_vertixes_besides(self, besides: Vertex) -> list:
        copy = list(self.vertices)
        copy.remove(besides)

        return copy

    def get_neighbors_of_vertex(self, vertex: Vertex) -> list:
        # We do not chekc it, bcs we delete is before using this method
        # if not self.vertices.__contains__(vertex):
        #     raise ValueError(f'Does not contain {vertex.name}')

        neighbors = []
        for edge_element in self.edges:
            if edge_element.point_one == vertex:
                neighbors.append(edge_element.point_two)

            if edge_element.point_two == vertex:
                neighbors.append(edge_element.point_one)

        return neighbors

    def get_neighbors_of_vertexes(self, vertexes: [Vertex]) -> list:
        neighbors = []
        for vertex in vertexes:
            neighbors.extend(self.get_neighbors_of_vertex(vertex))

        return neighbors


def run_algorythm(graph: Graph, starting_vertex: Vertex, ending_vertex: Vertex):
    # Begin

    # 1. Set types for vertexes and copy the graph
    starting_vertex.make_start_vertex()
    ending_vertex.make_end_vertex()
    original_vertexes = list(graph.vertices)
    original_edges = list(graph.edges)

    # 2. Initialise necessary elements:
    #       - Create the empty set of labeled values: W
    #       - Create dict of (): L (Co to wgl jest? Nigdzie tego nie było XD)
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
        vertex_j = None
        for vertex in graph.vertices:
            if all(L[vertex] <= L[vertex_k] for vertex_k in graph.get_vertixes_besides(vertex)):
                vertex_j = vertex

        graph.vertices.remove(vertex_j)
        W.append(vertex_j)

        for vertex_i in graph.get_neighbors_of_vertex(vertex_j):
            print(vertex_i)
        print('')

        # Ending condition
        if not ((not W.__contains__(starting_vertex)) and len(graph.get_neighbors_of_vertexes(W)) != 0):
            break


    # After - Return the old values
    graph.edges = original_edges
    graph.vertices = original_vertexes


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

# neighbor_test = list(map(lambda vertex: vertex.name, graph.get_neighbors_of_vertex(A)))
# print(neighbor_test)

run_algorythm(graph, A, G)