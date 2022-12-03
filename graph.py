from enum import Enum
import math

# INPUT DATA - CONSTANT
p_start_end = 1
t_start_end = 1


class NetworkInformation:
    def __init__(self, p_other: float, r_max: float, e_max: float, e_min: float, energy_per_package: float, beta_coef: float,
                 gamma_coef: float, q: float, m: float):
        self.p_other = p_other
        self.r_max = r_max
        self.e_max = e_max
        self.e_min = e_min
        self.energy_per_package = energy_per_package
        self.beta_coef = beta_coef
        self.gamma_coef = gamma_coef
        self.q = q
        self.m = m


class VertexType(Enum):
    TYPICAL = 'sensor'
    START = 'start'
    END = 'end'


class Vertex:
    def __init__(self, name: str, max_energy: float, min_energy: float, reliability: float):
        self.name = name
        self.current_energy = max_energy
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.reliability = reliability
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
            return self.max_energy

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
    def __init__(self, network_info: NetworkInformation, vertexes=[], edges=[]):
        self.vertices: list[Vertex] = vertexes
        self.edges: list[Edge] = edges
        self.total_load_traffic = 0
        self.network_info = network_info

    def add_vertex(self, name: str, reliability: float) -> Vertex:
        if any(vertex.name == name for vertex in self.vertices):
            raise ValueError(f'{name} already added')

        vertex = Vertex(name, self.network_info.e_max, self.network_info.e_min, reliability)
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
            # If it needs to be one-directional, have one selected
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
