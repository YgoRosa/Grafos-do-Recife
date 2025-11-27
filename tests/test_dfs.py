from src.graphs.algorithms import Algorithms
from src.graphs.graph import Graph

import pytest
def test_dfs_classificacao_arestas_sem_ciclo():
    """Testa a classificação de arestas (tree, cross, forward) e ausência de ciclo."""
    g = Graph(directed=True)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D") 
    g.add_edge("A", "D") 
    
    resultado = Algorithms.dfs(g, "A")
    
    classificacoes = list(resultado.get("classification", {}).values())
    
    assert "back" not in classificacoes
    assert resultado["has_cycle"] is False
    assert classificacoes.count('tree') >= 3 
    assert classificacoes.count('cross') >= 1 
    

def test_dfs_deteccao_ciclo():
    """Testa a detecção de ciclo e a classificação 'back'."""
    g = Graph(directed=True)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")
    resultado = Algorithms.dfs(g, "A")
    
    classificacoes = resultado.get("classification", {}).values()
    
    assert "back" in classificacoes
    assert resultado["has_cycle"] is True