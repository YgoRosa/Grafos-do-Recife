from typing import Dict, List, Tuple, Optional, Iterable

class Graph:
    """
    Grafo implementado por lista de adjacência.
    Suporta grafos não-dirigidos (padrão) e dirigidos (com directed=True).
    Cada entrada self.adj[u] é um dict: destino -> (peso, metadados).
    """

    def __init__(self, is_directed: bool = False): # <--- ADICIONE is_directed AQUI
    # adj: node -> dict(destino -> (peso: float, meta: dict))
        self.adj: Dict[str, Dict[str, Tuple[float, Dict]]] = {}
        self.node_metrics: Dict[str, Dict] = {}
        # ADICIONE ESTA LINHA:
        self.is_directed = is_directed
        
    # OBS: O copy() também deve ser atualizado para copiar este atributo.
    # Já corrijo abaixo.
    def add_node(self, u: str, metrics: Optional[Dict] = None): # <--- ATUALIZADO
        u = str(u).strip()
        
        if u not in self.adj:
            self.adj[u] = {}
        
        # Armazenar as métricas do nó (como a microrregião)
        if metrics is None:
            metrics = {}
            
        self.node_metrics[u] = metrics

    def has_node(self, u: str) -> bool:
        return u in self.adj

    def get_nodes(self) -> List[str]:
        return list(self.adj.keys())

    def __len__(self) -> int:
        return len(self.adj)

    # ----------------------------------------------------
    # --------------- Métodos de Arestas -----------------
    # ----------------------------------------------------
    
    # Em src/graphs/graph.py
    
# SUBSTITUA seu método add_edge existente por este:

    def add_edge(self, u: str, v: str, weight: float = 1.0, **kwargs): # <--- CORREÇÃO AQUI (remova meta, use **kwargs)
        """
        Adiciona aresta de u para v. 
        Todos os argumentos nomeados adicionais (ex: logradouro) são armazenados em metadados.
        """
        # Define o estado dirigido baseado no atributo da classe, se não for passado em kwargs
        is_directed = kwargs.pop('directed', self.is_directed) 
        
        u = str(u).strip()
        v = str(v).strip()
        self.add_node(u)
        self.add_node(v)
        
        # Os metadados serão todos os argumentos extras passados em **kwargs (incluindo 'logradouro', se houver)
        meta = kwargs
        
        # 1. Adiciona a aresta dirigida u -> v:
        self.adj[u][v] = (float(weight), meta.copy()) # usa .copy() para garantir deep copy
        
        # 2. Se não for dirigido, adiciona o espelhamento
        # Agora usa 'is_directed' da classe ou do argumento opcional
        if not is_directed:
            self.adj[v][u] = (float(weight), meta.copy())

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
    
    # Em src/graphs/graph.py

    def copy(self) -> 'Graph':
        """Cria uma cópia profunda (deep copy) do grafo."""
        new_graph = Graph(is_directed=self.is_directed)
        
        # Cópia da lista de adjacência (OK)
        new_graph.adj = {u: {v: (w, meta.copy()) for v, (w, meta) in nbrs.items()}
                         for u, nbrs in self.adj.items()}
        
        # [CORREÇÃO 3] Cópia das métricas dos nós
        new_graph.node_metrics = self.node_metrics.copy() # <--- ADICIONE ESTA LINHA!
        
        return new_graph
    
    def get_num_nodes(self) -> int:
        """ Retorna o número de nós (bairros) no grafo. """
        return len(self) # Chama o seu método __len__ já implementado

    def get_num_edges(self) -> int:
        """ Retorna o número de arestas no grafo. """
        # Seu método get_edges já lida com duplicatas em grafos não-dirigidos.
        return len(self.get_edges())
    
    # Em src/graphs/graph.py, dentro da classe Graph:

    def get_node_metrics(self, u: str) -> Optional[Dict]:
        """ 
        Retorna o dicionário de métricas/propriedades do nó u. 
        Exemplo: {'microrregiao': '1.1'}
        """
        # Acessa o dicionário que criamos para armazenar as propriedades do nó
        return self.node_metrics.get(u)

    # Note: O método get_nodes() já deve estar implementado logo acima:
    # def get_nodes(self) -> List[str]:
    #     return list(self.adj.keys())