from gekko import GEKKO
from graph import Vertex, Graph, VertexType

def solver_solution(graph: Graph, starting_vertex: Vertex, ending_vertex: Vertex):
    # TURBO WAŻNE INFO: WIERZCHOŁEK STARTOWY TO 0 a wierzchołek KOŃCOWY TO N-1!!!!!!1
    # dane z grafu:
    epsilon = 0.000000000001

    vertices = graph.vertices
    edges = graph.edges

    # Liczba wierzchołków
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

    # Wypłata
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

    vertex_start = 0
    vertex_end = n - 1

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

    # reliability = vertices[1].reliability
    reliability = 0.9
    reliability_table = [1 for i in range(n)]

    for i in range(n):
        if i == 0:
            reliability_table[i] = 1
        else:
            reliability_table[i] = reliability_table[i - 1] * reliability

    # Znajdowanie ścieżki w grafie
    m = GEKKO()

    A = [[m.Var(lb=0, ub=1, integer=True) for i in range(n)] for j in range(n)]

    path = [[[m.Var(lb=0, ub=1, integer=True) for i in range(n)] for j in range(n)] for z in range(n)]

    EXIST_IN_A = [m.Var(lb=0, ub=1, integer=True) for i in range(n)]

    EXIST_IN_PATH = [[m.Var(lb=0, ub=1, integer=True) for i in range(n)] for z in range(n)]

    NUM_OF_HOPS = [m.Var(lb=0, integer=True) for i in range(n)]

    C_ij = [[m.Var(integer=False) for i in range(n)] for j in range(n)]
    C_i = [m.Var(integer=False) for i in range(n)]
    B_i = [m.Var(integer=False) for i in range(n)]

    U_total = m.Var(integer=False)

    # initial values?

    # Equations

    # 1. Warunki utworzenia prawidłowej ścieżki

    # 1.1 Kazdy z wierzcholków użyty jest tylko raz

    # sprowadza się do zagwarantowania że z każdego wychodzi maks 1 krawędź
    for i in range(n):
        m.Equation(sum(A[i][j] for j in range(n)) <= 1)

    # 1.2 Do każdego wierzchołka wchodzi i wychodzi tylko jedna krawędź. Suma wchodzącyh == sumie wychodząyxh

    # 1.2.1 Dla dowolnego wierzchołka

    for x in range(n):
        if x == vertex_start:
            m.Equation(sum(A[vertex_start][j] for j in range(n)) == 1)
            m.Equation(sum(A[i][vertex_start] for i in range(n)) == 0)
        elif x == vertex_end:
            m.Equation(sum(A[i][vertex_end] for i in range(n)) == 1)
            m.Equation(sum(A[vertex_end][j] for j in range(n)) == 0)
        else:
            m.Equation(sum(A[x][j] for j in range(n)) - sum(A[i][x] for i in range(n)) == 0)

    # 1.3 Tylko w obrębie dostępnych wag
    m.Equation(sum(A[i][j] * gne[i][j] for i in range(n) for j in range(n)) == 0)

    ###
    # Dla wszystkich sciezek w grafie
    ###

    for z in range(n):
        for i in range(n):
            m.Equation(sum(path[z][i][j] for j in range(n)) <= 1)

    # 1.2 Do każdego wierzchołka wchodzi i wychodzi tylko jedna krawędź. Suma wchodzącyh == sumie wychodząyxh

    # 1.2.1 Dla dowolnego wierzchołka

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

    # 1.3 Tylko w obrębie dostępnych wag
    for z in range(n):
        m.Equation(sum(path[z][i][j] * gne[i][j] for i in range(n) for j in range(n)) == 0)

    # Sprawdź które istnieją:
    for i in range(n):
        if i == vertex_end:
            m.Equation(EXIST_IN_A[i] == 1)
        else:
            m.Equation(sum(A[i][j] for j in range(n)) - EXIST_IN_A[i] == 0)

    for z in range(n):
        for i in range(n):
            m.Equation(sum(path[z][i][j] for j in range(n)) - EXIST_IN_PATH[z][i] == 0)

    # Porównanie zbieżności ścieżki z oryginałem:
    for z in range(n):
        for i in range(n):
            for j in range(n):
                m.Equation((A[i][j] - path[z][i][j]) * EXIST_IN_A[
                    z] > -0.5)  # tutaj musi być > -1 a nie >= 0 bo inaczej się buguje. Pewnie jakieś floating point rounding numery

    ###
    # Kod powyżej powinien zapewnić że algorytm potrafi budować poprawne ścieżki
    # Oraz że path czyli ścieżka do każdego wierzchołka pokrywa się ze ścieżką w grafie

    # Oblicz długość dojścia do każdego wierzchołka:
    for z in range(n):
        m.Equation(sum(path[z][i][j] for i in range(n) for j in range(n)) - NUM_OF_HOPS[z] == 0)

    # Oblicz

    # 2. Warunki funkcji celu

    # Obliczanie kosztu Cij
    # Ten wzór trzeba dopracować!
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

    # Tutaj trzeba zrobić sumę po tych wszystkich Cij żeby otrzymać wartość pojedyncza
    for i in range(n):
        m.Equation(sum(C_ij[i][j] for j in range(n)) * EXIST_IN_A[i] - C_i[i] - epsilon < 0)
        m.Equation(sum(C_ij[i][j] for j in range(n)) * EXIST_IN_A[i] - C_i[i] + epsilon > 0)

    # Łatwo jest wyznaczyć b dla wierzchołka bo wiemy ile było wierzchołków na trasie
    for i in range(n):
        if i == vertex_start:
            m.Equation((m_param - NUM_OF_HOPS[vertex_end] * q) - B_i[i] - epsilon < 0)
            m.Equation((m_param - NUM_OF_HOPS[vertex_end] * q) - B_i[i] + epsilon > 0)
        else:
            m.Equation((reliability ** NUM_OF_HOPS[i]) * q - B_i[i] - epsilon < 0)
            m.Equation((reliability ** NUM_OF_HOPS[i]) * q - B_i[i] + epsilon > 0)

    # i wyliczamy prawidłowo funkcje celu z tych dwóch rzeczy
    m.Equation(sum((B_i[i] - C_i[i]) * EXIST_IN_A[i] for i in range(n)) - U_total - epsilon < 0)
    m.Equation(sum((B_i[i] - C_i[i]) * EXIST_IN_A[i] for i in range(n)) - U_total + epsilon > 0)

    # Maksymalizujemy funkcję użytkową
    m.Maximize(U_total)

    # Ten parametr musi być równy 1 żeby liczyło dla całkowitych
    m.options.SOLVER = 1
    #
    m.solve(disp=False)

    # Results
    nodes_that_transmited = [EXIST_IN_A[i].value[0] for i in range(n)]

    return nodes_that_transmited, U_total.value[0]
