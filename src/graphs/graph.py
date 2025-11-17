# src/graphs/graph.py
from typing import Dict, List, Tuple, Optional, Iterable

class Graph:
    """
    Grafo não-direcionado simples implementado por lista de adjacência.
    Cada entrada self.adj[u] é um dict: destino -> (peso, metadados)
    onde metadados é um dict livre (ex.: {'logradouro': 'Av. X'}).
    """

    def __init__(self):
        # adj: node -> dict(destino -> (peso: float, meta: dict))
        self.adj: Dict[str, Dict[str, Tuple[float, Dict]]] = {}

    # --------- nós ----------
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

    # --------- arestas ----------
    def add_edge(self, u: str, v: str, weight: float = 1.0, meta: Optional[Dict] = None):
        """
        Adiciona aresta não-direcionada entre u e v.
        Se os nós não existirem, cria-os.
        Se a aresta já existir, sobrescreve o peso/meta.
        """
        if meta is None:
            meta = {}
        u = str(u).strip()
        v = str(v).strip()
        self.add_node(u)
        self.add_node(v)
        self.adj[u][v] = (float(weight), dict(meta))
        self.adj[v][u] = (float(weight), dict(meta))

    def neighbors(self, u: str) -> List[str]:
        return list(self.adj.get(u, {}).keys())

    def degree(self, u: str) -> int:
        return len(self.adj.get(u, {}))

    def get_edge_data(self, u: str, v: str) -> Optional[Tuple[float, Dict]]:
        return self.adj.get(u, {}).get(v)

    def get_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Retorna lista de arestas sem duplicatas: (u, v, weight, meta) com u <= v lexicograficamente
        """
        seen = set()
        edges = []
        for u, nbrs in self.adj.items():
            for v, (w, meta) in nbrs.items():
                key = tuple(sorted((u, v)))
                if key not in seen:
                    seen.add(key)
                    edges.append((key[0], key[1], w, meta))
        return edges

    def __repr__(self):
        n = len(self)
        m = len(self.get_edges())
        return f"Graph({n} nós, {m} arestas)"
