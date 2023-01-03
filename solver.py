from gekko import GEKKO
from graph import Vertex, Graph, VertexType

def solver_solution(graph: Graph, starting_vertex: Vertex, ending_vertex: Vertex):
    starting_vertex.make_start_vertex()
    ending_vertex.make_end_vertex()

    # Data from graph
    epsilon = 0.000000000001

    vertices = graph.vertices
    edges = graph.edges

    # Nr. of vertices
    n = len(graph.vertices)

    D = [[-1 for i in range(n)] for j in range(n)]

    for edge in graph.edges:
        D[vertices.index(edge.point_one)][vertices.index(edge.point_two)] = edge.distance

    gne = []

    # graph negative edges (-1 if there is no edge, 0 if there is an edge)
    gne = [[-1 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if D[i][j] != -1:
                gne[i][j] = 0

    ###
    load_traffic = []
    load_traffic = [0 for i in range(n)]
    for i in range(n):
        load_traffic[i] = vertices[i].load_traffic

    max_load_traffic = graph.total_load_traffic

    # Energy
    e_max = vertices[0].max_energy
    e_current = [vertices[i].current_energy for i in range(n)]

    # Mathematical coeficients
    beta_coef = graph.network_info.beta_coef
    gamma_coef = graph.network_info.gamma_coef

    # Game theory:
    # Payment for intermediate node for successful packet transmission
    q = graph.network_info.q
    # Payment for source node for successful packet transmission
    m_param = graph.network_info.m

    # Starting and ending vertex indexes
    vertex_start = graph.vertices.index(starting_vertex)
    vertex_end = graph.vertices.index(ending_vertex)

    for vertex in graph.vertices:
        if vertex.type == VertexType.START:
            vertex_start = graph.vertices.index(vertex)
        if vertex.type == VertexType.END:
            vertex_end = graph.vertices.index(vertex)

    load_traffic[vertex_start] = max_load_traffic
    load_traffic[vertex_end] = max_load_traffic
    e_current[vertex_start] = e_max
    e_current[vertex_end] = e_max

    if max_load_traffic == 0:
        max_load_traffic = 1

    reliability = graph.network_info.p_other
    reliability_table = [1 for i in range(n)]

    for i in range(n):
        if i == 0:
            reliability_table[i] = 1
        else:
            reliability_table[i] = reliability_table[i - 1] * reliability

    # Added remote=False, because running tests was too much for remote
    m = GEKKO(remote=False)

    A = [[m.Var(lb=0, ub=1, integer=True) for i in range(n)] for j in range(n)]

    path = [[[m.Var(lb=0, ub=1, integer=True) for i in range(n)] for j in range(n)] for z in range(n)]

    EXIST_IN_A = [m.Var(lb=0, ub=1, integer=True) for i in range(n)]

    EXIST_IN_PATH = [[m.Var(lb=0, ub=1, integer=True) for i in range(n)] for z in range(n)]

    NUM_OF_HOPS = [m.Var(lb=0, integer=True) for i in range(n)]

    C_ij = [[m.Var(integer=False) for i in range(n)] for j in range(n)]
    C_i = [m.Var(integer=False) for i in range(n)]
    B_i = [m.Var(integer=False) for i in range(n)]

    U_total = m.Var(integer=False)

    # Equations

    # 1. Conditions for creating correct path

    # 1.1 Each vertex can be used only one time
    for i in range(n):
        m.Equation(sum(A[i][j] for j in range(n)) <= 1)

    # 1.2 For each vertex, there can only be one input (receiving) and one output (sending to)

    # 1.2.1 For each vertex
    for x in range(n):
        if x == vertex_start:
            m.Equation(sum(A[vertex_start][j] for j in range(n)) == 1)
            m.Equation(sum(A[i][vertex_start] for i in range(n)) == 0)
        elif x == vertex_end:
            m.Equation(sum(A[i][vertex_end] for i in range(n)) == 1)
            m.Equation(sum(A[vertex_end][j] for j in range(n)) == 0)
        else:
            m.Equation(sum(A[x][j] for j in range(n)) - sum(A[i][x] for i in range(n)) == 0)

    # 1.2.2 Only for given, correct edges
    m.Equation(sum(A[i][j] * gne[i][j] for i in range(n) for j in range(n)) == 0)

    ###
    # For all Paths in the graph
    ###

    for z in range(n):
        for i in range(n):
            m.Equation(sum(path[z][i][j] for j in range(n)) <= 1)

    # 1.2 For each vertex, there can only be one input (receiving) and one output (sending to)

    # 1.2.1 For each vertex
    for z in range(n):
        if z == vertex_start:
            continue

        for x in range(n):
            if x == vertex_start:
                m.Equation(sum(path[z][vertex_start][j] for j in range(n)) == 1)
                m.Equation(sum(path[z][i][vertex_start] for i in range(n)) == 0)
            elif x == z:
                m.Equation(sum(path[z][i][z] for i in range(n)) == 1)
                m.Equation(sum(path[z][z][j] for j in range(n)) == 0)
            else:
                m.Equation(sum(path[z][x][j] for j in range(n)) - sum(path[z][i][x] for i in range(n)) == 0)

    # 1.2.2 Only for given, correct edges
    for z in range(n):
        m.Equation(sum(path[z][i][j] * gne[i][j] for i in range(n) for j in range(n)) == 0)

    # Check which edges exist
    for i in range(n):
        if i == vertex_end:
            m.Equation(EXIST_IN_A[i] == 1)
        else:
            m.Equation(sum(A[i][j] for j in range(n)) - EXIST_IN_A[i] == 0)

    # Check which edges exist
    for z in range(n):
        for i in range(n):
            m.Equation(sum(path[z][i][j] for j in range(n)) - EXIST_IN_PATH[z][i] == 0)

    # Check if original is the same
    for z in range(n):
        for i in range(n):
            for j in range(n):
                m.Equation((A[i][j] - path[z][i][j]) * EXIST_IN_A[z] > -0.5)

    ###
    # The above code guarantees correct paths
    # and that the path correspond to the path in the graph

    # Calculate nr. of hops to each vertex in the table
    for z in range(n):
        m.Equation(sum(path[z][i][j] for i in range(n) for j in range(n)) - NUM_OF_HOPS[z] == 0)

    # 2. Utility function conditions

    # Calculate Cij (cost for transporting to each)
    for i in range(n):
        for j in range(n):
            if i == vertex_end:
                m.Equation(C_ij[i][j] == 0)
            else:
                m.Equation(((beta_coef * D[i][j] * D[i][j]) * (e_max / e_current[j]) * (
                            1 + gamma_coef * (load_traffic[j] / max_load_traffic))) * EXIST_IN_A[i] * EXIST_IN_A[j] -
                           C_ij[i][j] - epsilon < 0)

                m.Equation(((beta_coef * D[i][j] * D[i][j]) * (e_max / e_current[j]) * (
                            1 + gamma_coef * (load_traffic[j] / max_load_traffic))) * EXIST_IN_A[i] * EXIST_IN_A[j] -
                           C_ij[i][j] + epsilon > 0)

    # Calculate the sum for each Vertex
    # Because EXIST_IN_A is non-zero for vertexes in path, the sum return the path cost
    for i in range(n):
        m.Equation(sum(C_ij[i][j] for j in range(n)) * EXIST_IN_A[i] - C_i[i] - epsilon < 0)
        m.Equation(sum(C_ij[i][j] for j in range(n)) * EXIST_IN_A[i] - C_i[i] + epsilon > 0)

    # Calculate Bi for each vertex in the path
    for i in range(n):
        if i == vertex_start:
            m.Equation((m_param - NUM_OF_HOPS[vertex_end] * q) - B_i[i] - epsilon < 0)
            m.Equation((m_param - NUM_OF_HOPS[vertex_end] * q) - B_i[i] + epsilon > 0)
        else:
            m.Equation((reliability ** NUM_OF_HOPS[i]) * q - B_i[i] - epsilon < 0)
            m.Equation((reliability ** NUM_OF_HOPS[i]) * q - B_i[i] + epsilon > 0)

    # And calculate the utility function
    m.Equation(sum((B_i[i] - C_i[i]) * EXIST_IN_A[i] for i in range(n)) - U_total - epsilon < 0)
    m.Equation(sum((B_i[i] - C_i[i]) * EXIST_IN_A[i] for i in range(n)) - U_total + epsilon > 0)

    # Select the utility function for maximization
    m.Maximize(U_total)

    # Make the solver use intiger numbers
    m.options.SOLVER = 1

    m.solve(disp=False)

    # Results
    nodes_that_transmited = [EXIST_IN_A[i].value[0] for i in range(n)]

    nodes_matrix = [[A[i][j].value[0] for i in range(n)] for j in range(n)]

    return nodes_that_transmited, U_total.value[0], nodes_matrix

def get_path_in_order(graph: Graph, node_matrix):
    vertex_start = 0
    vertex_end = len(graph.vertices)-1

    for vertex in graph.vertices:
        if vertex.type == VertexType.START:
            vertex_start = graph.vertices.index(vertex)
        if vertex.type == VertexType.END:
            vertex_end = graph.vertices.index(vertex)

    cur_vertex = vertex_end
    path_in_order = []
    path_in_order.append(cur_vertex)

    n = len(graph.vertices)

    while cur_vertex != vertex_start:
        for i in range(n):
            if node_matrix[cur_vertex][i] > 0.1:
                cur_vertex = i
                path_in_order.append(cur_vertex)
                break

    path_in_order.reverse()

    path_in_order = list(map(lambda vertex_index: graph.vertices[vertex_index], path_in_order))

    return path_in_order