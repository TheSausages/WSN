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

# Mathematical coeficients
beta_coef = 0.05
gamma_coef = 0.2

# Zmienne
energy_per_package = 2

# Game theory:
# Payment for intermediate node for successfull packet transmission
q = 5.0
# Payment for source node for successfull packet transmission
m = 30.0

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
        if self.type == VertexType.END:
            return t_start_end
        else:
            return self._load_traffic

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

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return (
                hasattr(other, 'name') and other.name == self.name
        )


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
        self.total_load_traffic = 0

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

        self.edges.append(Edge(point_one, point_two, distance))

    def print_graph(self):
        for vertex in self.vertices:
            print(vertex.__str__())

        for edge_element in self.edges:
            print(edge_element.__str__())
        print('\n')

    def get_vertixes_besides(self, besides: Vertex) -> list:
        copy = list(self.vertices)
        copy.remove(besides)

        return copy

    def get_neighbors_of_vertex(self, vertex: Vertex) -> list:
        neighbors = []
        for edge_element in self.edges:
            # If it needs to be one-directional, delete a
            # selected one of the following 2 (depending on direction)
            if edge_element.point_one == vertex:
                neighbors.append(edge_element.point_two)

            if edge_element.point_two == vertex:
                neighbors.append(edge_element.point_one)

        return neighbors

    def get_distance(self, vertex_i: Vertex, vertex_j: Vertex) -> float:
        for edge in self.edges:
            if ((edge.point_one == vertex_i and edge.point_two == vertex_j) or
                    (edge.point_two == vertex_i and edge.point_one == vertex_j)):
                return edge.distance
        return math.inf

    def get_neighbors_of_vertexes(self, vertexes) -> list:
        neighbors = []
        for vertex in vertexes:
            neighbors.extend(self.get_neighbors_of_vertex(vertex))

        return neighbors

    def delete_edge(self, vertex_i: Vertex, vertex_j: Vertex):
        for edge in self.edges:
            if ((edge.point_one == vertex_i and edge.point_two == vertex_j) or
                    (edge.point_two == vertex_i and edge.point_one == vertex_j)):
                self.edges.remove(edge)


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
# vertex_i - Starting point
# vertex_j - next hop from vertex_i
# previous - dict with previous vertexes for a given vertexes
def calculate_cost_function(graph: Graph, vertex_i: Vertex, vertex_j: Vertex, previous: dict):
    # print(f'{vertex_i.name} -> {vertex_j.name}')

    # Cost function for intermediate node

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
    if vertex_j.type == VertexType.START:
        traffic_density_j = 1.0
    else:
        traffic_density_j = vertex_j.load_traffic / graph.total_load_traffic if graph.total_load_traffic >= 1 else 0
    cost_load = 1.0 + gamma_coef * traffic_density_j

    if vertex_j.type == VertexType.START:
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
            c_ij = calculate_cost_function(graph, vertex_j, vertex_i, Previous)
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
