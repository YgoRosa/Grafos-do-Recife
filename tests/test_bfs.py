# test_bfs.py
from src.graphs.algorithms import Algorithms # Ajuste o caminho de importação
from src.graphs.graph import Graph
 # Ajuste o caminho de importação
import pytest

def test_bfs_niveis_corretos():
    """Testa se o BFS calcula os níveis (distâncias de salto) corretos."""
    g = Graph(directed=True)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "E")
    g.add_edge("D", "F")
    
    resultado = Algorithms.bfs(g, "A")
    
    niveis_esperados = {
        "A": 0, 
        "B": 1, 
        "C": 1, 
        "D": 2, 
        "E": 2, 
        "F": 3
    }
    
    # O BFS na sua implementação retorna "levels"
    assert resultado.get("levels") == niveis_esperados
    assert "back" not in resultado.get("order")