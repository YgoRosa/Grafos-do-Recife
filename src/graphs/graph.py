class Graph:
    def __init__(self):
        """Inicializa um grafo vazio."""
        self.adj_list = {}  # dicionário: {bairro: [bairros_vizinhos]}

    def add_node(self, bairro: str):
        """Adiciona um novo nó (bairro) ao grafo, se ainda não existir."""
        if bairro not in self.adj_list:
            self.adj_list[bairro] = []

    def get_nodes(self):
        """Retorna a lista de nós (bairros) do grafo."""
        return list(self.adj_list.keys())

    def __len__(self):
        """Número de nós do grafo."""
        return len(self.adj_list)

    def __repr__(self):
        return f"Graph({len(self)} nós)"
