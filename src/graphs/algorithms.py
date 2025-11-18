import heapq

def dijkstra(graph, start):
    import heapq

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
