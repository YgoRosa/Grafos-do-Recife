import heapq

def dijkstra(graph, start):

    # Inicialização
    dist = {node: float("inf") for node in graph.adj}
    prev = {node: None for node in graph.adj}

    dist[start] = 0
    pq = [(0, start)]  # heap de prioridades

    while pq:
        d_u, u = heapq.heappop(pq)

        if d_u > dist[u]:
            continue

        # Aqui o formaro correto é: graph.adj[u][v] = (peso, meta)
        for v, (peso, meta) in graph.adj[u].items():
            weight = float(peso)

            alt = d_u + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    return dist, prev



def dijkstra_path(graph, source, target):
    """
    Implementa um equivalente ao networkx.dijkstra_path()

    Retorna:
        caminho (lista)
    """

    dist, prev = dijkstra(graph, source)

    if dist[target] == float("inf"):
        raise ValueError(f"Não existe caminho entre {source} e {target}")

    # Reconstruir caminho
    caminho = []
    atual = target
    while atual is not None:
        caminho.append(atual)
        atual = prev[atual]

    caminho.reverse()
    return caminho

# src/graphs/algorithms.py - Continuação
from typing import Dict, List, Tuple, Any
from collections import deque 

# Variáveis globais/auxiliares para rastrear o tempo/status na DFS
time = 0
status: Dict[str, str] = {} # 'white', 'gray', 'black'
predecessor: Dict[str, str] = {}
discovery_time: Dict[str, int] = {}
finish_time: Dict[str, int] = {}
has_cycle = False # Flag para detecção de ciclo


def dfs_visit(graph, u: str):
    global time, status, predecessor, discovery_time, finish_time, has_cycle
    
    status[u] = 'gray' # Nó descoberto, mas não finalizado
    time += 1
    discovery_time[u] = time
    
    # Itera sobre os vizinhos
    for v in graph.neighbors(u): 
        if status.get(v) == 'white': # Aresta de árvore (tree edge)
            predecessor[v] = u
            dfs_visit(graph, v)
        elif status.get(v) == 'gray':
            # Aresta de retorno (back edge): indica um ciclo em grafos dirigidos
            has_cycle = True
            # Adicionar lógica para classificação de arestas (opcional)

    status[u] = 'black' # Nó finalizado
    time += 1
    finish_time[u] = time


def dfs(graph, start_node: str) -> Tuple[Dict[str, int], Dict[str, int], bool]:
    """
    Busca em Profundidade. 
    Retorna tempos de descoberta, tempos de finalização e flag de ciclo.
    """
    global time, status, predecessor, discovery_time, finish_time, has_cycle
    
    # Resetar variáveis globais para nova execução
    time = 0
    status = {node: 'white' for node in graph.get_nodes()}
    predecessor = {}
    discovery_time = {}
    finish_time = {}
    has_cycle = False
    
    # Tenta visitar o nó inicial
    if start_node in graph.adj:
        dfs_visit(graph, start_node)
        
    # Se o grafo não estiver totalmente conectado, visita o resto dos nós brancos
    for node in graph.get_nodes():
        if status[node] == 'white':
             dfs_visit(graph, node)

    # Note: Para simular o modelo NetworkX, você pode retornar apenas o predecessor ou as arestas,
    # mas para os requisitos do projeto (ciclo, camadas/tempos), a saída acima é mais útil.
    return discovery_time, finish_time, has_cycle

def dijkstra(graph, start_node: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Encontra o caminho mais curto em grafos com pesos não-negativos.
    Retorna distâncias e predecessores.
    """
    if start_node not in graph.adj:
        return {}, {}
    
    # Inicialização
    infinity = float('inf')
    distance: Dict[str, float] = {node: infinity for node in graph.get_nodes()}
    predecessor: Dict[str, str] = {}
    
    distance[start_node] = 0.0
    
    # Fila de prioridade: (distância, nó)
    pq = [(0.0, start_node)]
    
    # Dicionário para rastrear a melhor distância encontrada para um nó
    # (heapq pode ter entradas obsoletas)
    
    while pq:
        d_u, u = heapq.heappop(pq)
        
        # Ignora se encontrarmos um caminho mais longo para 'u'
        if d_u > distance[u]:
            continue
            
        # Relaxamento
        for v in graph.neighbors(u):
            # Obtém o peso da aresta (u -> v)
            data = graph.get_edge_data(u, v)
            if data is None:
                # Se for dirigido e não houver aresta, continua
                continue 
            
            weight_uv, _ = data
            
            # ATENÇÃO: Dijkstra recusa pesos negativos. O projeto exige essa checagem!
            if weight_uv < 0:
                raise ValueError("Dijkstra não suporta pesos negativos.")

            if distance[u] + weight_uv < distance[v]:
                distance[v] = distance[u] + weight_uv
                predecessor[v] = u
                heapq.heappush(pq, (distance[v], v))
                
    return distance, predecessor

# src/graphs/algorithms.py - Continuação

def bellman_ford(graph, start_node: str) -> Tuple[Dict[str, float], Dict[str, str], bool]:
    """
    Encontra o caminho mais curto, suportando pesos negativos. 
    Detecta e reporta ciclos negativos.
    Retorna distâncias, predecessores e flag is_negative_cycle.
    """
    if start_node not in graph.adj:
        return {}, {}, False
    
    infinity = float('inf')
    nodes = graph.get_nodes()
    distance: Dict[str, float] = {node: infinity for node in nodes}
    predecessor: Dict[str, str] = {}
    distance[start_node] = 0.0
    
    # Obter todas as arestas do grafo
    # Para Bellman-Ford, precisamos de uma lista que inclua todas as arestas dirigidas (u, v, w)
    # A função graph.get_edges() deve retornar uma lista de arestas dirigidas (u, v, w, meta)
    all_edges = graph.get_edges()
    
    # 1. Relaxamento V - 1 vezes
    for _ in range(len(nodes) - 1):
        relaxed = False
        for u, v, weight_uv, _ in all_edges:
            # Relaxamento
            if distance[u] != infinity and distance[u] + weight_uv < distance[v]:
                distance[v] = distance[u] + weight_uv
                predecessor[v] = u
                relaxed = True
        # Otimização: Se nenhuma aresta foi relaxada na iteração, podemos parar
        if not relaxed:
            break

    # 2. V-ésima iteração para detectar ciclo negativo
    is_negative_cycle = False
    for u, v, weight_uv, _ in all_edges:
        if distance[u] != infinity and distance[u] + weight_uv < distance[v]:
            # Ciclo negativo detectado!
            is_negative_cycle = True
            break
            
    return distance, predecessor, is_negative_cycle


def bfs(graph, start_node: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Busca em Largura. 
    Retorna distâncias (níveis topológicos) e predecessores a partir de start_node.
    """
    if start_node not in graph.adj:
        return {}, {}

    # Distância (nível topológico). Inicialmente, -1 (não visitado)
    distance: Dict[str, int] = {node: -1 for node in graph.get_nodes()}
    predecessor: Dict[str, str] = {}
    
    queue = deque([start_node])
    distance[start_node] = 0

    while queue:
        u = queue.popleft()

        # Itera sobre os vizinhos (válido para dirigido e não-dirigido)
        for v in graph.neighbors(u): 
            if distance[v] == -1: # Se o nó v não foi visitado
                distance[v] = distance[u] + 1
                predecessor[v] = u
                queue.append(v)
                
    return distance, predecessor