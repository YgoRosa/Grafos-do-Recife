import heapq
from .graph import Graph

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

# Em src/graphs/algorithms.py

def calculate_global_metrics(g: Graph) -> dict:
    """
    Calcula as métricas globais do grafo: Ordem, Tamanho e Densidade.

    Args:
        g (Graph): O objeto Grafo dos Bairros do Recife.

    Returns:
        dict: Dicionário com as métricas calculadas.
    """
    
    # 1. Ordem (Número de Nós)
    ordem = g.get_num_nodes() 
    
    # 2. Tamanho (Número de Arestas)
    tamanho = g.get_num_edges() 
    
    # 3. Densidade
    # Fórmula da Densidade para grafos NÃO-DIRIGIDOS: 
    # Densidade = 2 * E / (V * (V - 1)), onde V é a Ordem e E é o Tamanho.
    
    if ordem <= 1:
        densidade = 0.0
    else:
        # Garante que as operações são de ponto flutuante
        densidade = (2.0 * tamanho) / (ordem * (ordem - 1))
        
    return {
        "ordem": ordem,
        "tamanho": tamanho,
        "densidade": densidade
    }
# Em src/graphs/algorithms.py

import pandas as pd
import json

# Reutiliza a função de cálculo de métricas globais
from .algorithms import calculate_global_metrics 
# Se a função calculate_global_metrics estiver no mesmo arquivo, não precisa importar.
# Em src/graphs/algorithms.py

# Em src/graphs/algorithms.py

# Em src/graphs/algorithms.py

def extract_and_calculate_microrregiao_metrics(g: Graph, output_path: str) -> pd.DataFrame:
    
    # 1. Agrupar nós por microrregião
    microrregioes_map = {}
    nodes_ignored_count = 0 
    
    for node_id in g.get_nodes():
        metrics = g.get_node_metrics(node_id)
        microrregiao_raw = metrics.get('microrregiao')
        
        # Tenta obter a string limpa
        microrregiao_str = str(microrregiao_raw).strip()
        
        # ⚠️ CORREÇÃO: Define o ID da Microrregião
        microrregiao_id = None
        
        # Teste de validação: deve ser uma string não vazia e não 'nan'/'none'
        if microrregiao_str and microrregiao_str.lower() not in ('nan', 'none', ''):
            microrregiao_id = microrregiao_str
        else:
            # ⚠️ Agrupa os nós sem dados válidos sob uma chave única
            microrregiao_id = "SEM REGIAO"
            
        # Agrupa
        if microrregiao_id not in microrregioes_map:
            microrregioes_map[microrregiao_id] = []
        microrregioes_map[microrregiao_id].append(node_id)
        
    # O count de ignorados não é mais necessário, mas vamos mantê-lo a 0.
    nodes_ignored_count = 0
            
    print(f"✅ Agrupamento concluído. {len(microrregioes_map)} microrregiões encontradas.")
    print(f"AVISO: {nodes_ignored_count} nós foram ignorados por falta de Microrregião válida.")
    
    # 2. Inicialização dos Resultados e Iteração
    results = {}
    
    # Certifique-se de que 'json' está importado no topo do arquivo.
    import json
    
    for microrregiao, nodes_list in microrregioes_map.items():
        
        # 3. Construir o Subgrafo Induzido
        subgraph = Graph(is_directed=g.is_directed) 
        for node in nodes_list:
            # Reutiliza as métricas existentes do nó
            subgraph.add_node(node, metrics=g.get_node_metrics(node)) 

        # Adiciona APENAS as arestas internas
        for u, v, w, meta in g.get_edges():
            if u in nodes_list and v in nodes_list:
                subgraph.add_edge(u, v, weight=w, **meta)

        # 4. Calcular Métricas Globais para o Subgrafo
        # Assume-se que calculate_global_metrics está disponível
        metrics = calculate_global_metrics(subgraph) 
        metrics["microrregiao"] = microrregiao 
        
        results[microrregiao] = metrics

    # 5. Salvar o JSON (Requisito 3.2)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Métricas por microrregião salvas em: {output_path}")

    # 6. Retornar um DataFrame para exibição no Streamlit
    df_results = pd.DataFrame.from_dict(results, orient='index')
    # Adiciona SEM REGIAO se existir, ou apenas os IDs numéricos
    df_results = df_results[["microrregiao", "ordem", "tamanho", "densidade"]]
    
    return df_results