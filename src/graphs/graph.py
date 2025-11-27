from typing import Dict, List, Tuple, Optional, Set

class Graph:
    def __init__(self, directed: bool = False):
        self.adj: Dict[str, Dict[str, Tuple[float, Dict]]] = {}
        self.directed = directed
    
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

    def add_edge(self, u: str, v: str, weight: float = 1.0, meta: Optional[Dict] = None):
        if meta is None:
            meta = {}
            
        u = str(u).strip()
        v = str(v).strip()
        
        self.add_node(u)
        self.add_node(v)
        
        self.adj[u][v] = (float(weight), dict(meta))

        if not self.directed:
            self.adj[v][u] = (float(weight), dict(meta))

    def neighbors(self, u: str) -> List[str]:
        return list(self.adj.get(u, {}).keys())

    def degree(self, u: str) -> int:
        return len(self.adj.get(u, {}))

    def get_edge_data(self, u: str, v: str) -> Optional[Tuple[float, Dict]]:
        return self.adj.get(u, {}).get(v)

    def get_edges(self) -> List[Tuple[str, str, float, Dict]]:
        edges = []
        seen: Set[Tuple[str, str]] = set()

        for u, nbrs in self.adj.items():
            for v, (weight, meta) in nbrs.items():
                
                if self.directed:
                    edges.append((u, v, weight, meta))
                else:
                    pair = tuple(sorted((u, v)))
                    if pair not in seen:
                        edges.append((u, v, weight, meta))
                        seen.add(pair)
                        
        return edges

    def __repr__(self):
        type_str = "Directed" if self.directed else "Undirected"
        n = len(self)
        m = len(self.get_edges())
        return f"Graph({type_str}, {n} nodes, {m} edges)"
    
    def copy(self) -> 'Graph':
        new_graph = Graph(directed=self.directed)
        new_graph.adj = {
            u: {v: (w, meta.copy()) for v, (w, meta) in nbrs.items()}
            for u, nbrs in self.adj.items()
        }
        return new_graph