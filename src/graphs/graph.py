from typing import Dict, List, Tuple, Optional, Iterable

class Graph:
    """
    Grafo implementado por lista de adjacência.
    Suporta grafos não-dirigidos (padrão) e dirigidos (com directed=True).
    Cada entrada self.adj[u] é um dict: destino -> (peso, metadados).
    """

    def __init__(self):
        # adj: node -> dict(destino -> (peso: float, meta: dict))
        self.adj: Dict[str, Dict[str, Tuple[float, Dict]]] = {}

    # ----------------------------------------------------
    # --------------- Métodos de Nós ---------------------
    # ----------------------------------------------------
    
    def add_node(self, u: str):
        u = str(u).strip()
        if u not in self.adj:
            self.adj[u] = {}

    def has_node(self, u: str) -> bool:
        return u in self.adj

    def get_nodes(self) -> List[str]:
        return list(self.adj.keys())

    def __len__(self) -> int:
        return len(self.adj)

    # ----------------------------------------------------
    # --------------- Métodos de Arestas -----------------
    # ----------------------------------------------------
    
    def add_edge(self, u: str, v: str, weight: float = 1.0, meta: Optional[Dict] = None, directed: bool = False):
        """
        Adiciona aresta de u para v. 
        - Se directed=False (Parte 1), adiciona v para u também (não-dirigido).
        - Se directed=True (Parte 2), adiciona apenas u -> v.
        """
        if meta is None:
            meta = {}
        u = str(u).strip()
        v = str(v).strip()
        self.add_node(u)
        self.add_node(v)
        
        # 1. Adiciona a aresta dirigida u -> v:
        self.adj[u][v] = (float(weight), dict(meta))
        
        # 2. Se não for dirigido (padrão para Parte 1), adiciona o espelhamento
        if not directed:
            self.adj[v][u] = (float(weight), dict(meta))

    def neighbors(self, u: str) -> List[str]:
        """ Retorna todos os vizinhos (adjacentes) de u. """
        return list(self.adj.get(u, {}).keys())

    def degree(self, u: str) -> int:
        """ Retorna o grau (saída para dirigidos, total para não-dirigidos) de u. """
        return len(self.adj.get(u, {}))

    def get_edge_data(self, u: str, v: str) -> Optional[Tuple[float, Dict]]:
        """ Retorna os dados da aresta u -> v (peso, meta). """
        return self.adj.get(u, {}).get(v)

    def get_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Retorna lista de arestas. 
        - Para grafos NÃO-DIRIGIDOS (Parte 1), evita duplicatas (u,v).
        - Para grafos DIRIGIDOS (Parte 2), retorna todas as arestas (u,v).
        """
        edges = []
        for u, nbrs in self.adj.items():
            for v, (w, meta) in nbrs.items():
                
                # Heurística para evitar duplicatas: 
                # Se a aresta tem metadados de logradouro (Parte 1) e u > v, ignoramos a duplicata
                # Isso funciona porque o add_edge sempre adiciona u->v primeiro.
                if 'logradouro' in meta and u > v:
                    continue
                
                edges.append((u, v, w, meta))
        return edges

    def __repr__(self):
        n = len(self)
        m = len(self.get_edges())
        return f"Graph({n} nós, ~{m} arestas)"
    
    def copy(self) -> 'Graph':
        """Cria uma cópia profunda (deep copy) do grafo."""
        new_graph = Graph()
        new_graph.adj = {u: {v: (w, meta.copy()) for v, (w, meta) in nbrs.items()}
                        for u, nbrs in self.adj.items()}
        return new_graph