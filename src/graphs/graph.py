from typing import Dict, List, Tuple, Optional, Set

class Graph:
    """
    Adjacency List Graph Implementation.
    Supports both Undirected (default) and Directed graphs.
    
    Structure:
    self.adj[u][v] = (weight, metadata)
    """

    def __init__(self, directed: bool = False):
        """
        Args:
            directed (bool): If True, edges are one-way (u -> v). 
                             If False, edges are bidirectional (u <-> v).
        """
        self.adj: Dict[str, Dict[str, Tuple[float, Dict]]] = {}
        self.directed = directed
    
    def add_node(self, u: str):
        """Adds a node to the graph if it doesn't exist."""
        u = str(u).strip()
        if u not in self.adj:
            self.adj[u] = {}

    def has_node(self, u: str) -> bool:
        """Checks if a node exists."""
        return u in self.adj

    def get_nodes(self) -> List[str]:
        """Returns a list of all nodes."""
        return list(self.adj.keys())

    def __len__(self) -> int:
        """Returns number of nodes (Order)."""
        return len(self.adj)

    def add_edge(self, u: str, v: str, weight: float = 1.0, meta: Optional[Dict] = None):
        """
        Adds an edge from u to v.
        If self.directed is False, also adds v to u.
        """
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
        """Returns a list of neighbors for node u."""
        return list(self.adj.get(u, {}).keys())

    def degree(self, u: str) -> int:
        """
        Returns the degree of u.
        - Undirected: Total connections.
        - Directed: Out-degree.
        """
        return len(self.adj.get(u, {}))

    def get_edge_data(self, u: str, v: str) -> Optional[Tuple[float, Dict]]:
        """Returns (weight, metadata) for edge u -> v, or None."""
        return self.adj.get(u, {}).get(v)

    def get_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Returns a list of all edges (u, v, weight, meta).
        
        - If Directed: Returns all directed edges.
        - If Undirected: Returns unique pairs {u, v} to avoid duplicates in iteration.
        """
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
        """Creates a deep copy of the graph."""
        new_graph = Graph(directed=self.directed)
        new_graph.adj = {
            u: {v: (w, meta.copy()) for v, (w, meta) in nbrs.items()}
            for u, nbrs in self.adj.items()
        }
        return new_graph