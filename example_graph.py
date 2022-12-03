from graph import Graph, NetworkInformation, VertexType
from solver import solver_solution
from wsn_algorithm import run_algorythm

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

# Run wsn_algorythm
print('WSN Algorythm')
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

# Run solver solution
print('Solver Algorythm')
for package in range(0, 10):
    starting_vertex = A
    ending_vertex = G

    out = solver_solution(graph, starting_vertex, ending_vertex)

    print(f'Round {package + 1} finished with path:')
    last_element = out[-1]
    for path_element in out:
        if path_element == last_element:
            print(f'{path_element}')
        else:
            print(f'{path_element} -> ', end='')

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