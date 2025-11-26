# test_dfs.py
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
    g.add_edge("A", "D") # Aresta forward ou cross
    
    resultado = Algorithms.dfs(g, "A")
    
    classificacoes = list(resultado.get("classification", {}).values())
    
    assert "back" not in classificacoes
    assert resultado["has_cycle"] is False
    assert classificacoes.count('tree') >= 3 # Deve ter pelo menos 3 tree edges
    assert classificacoes.count('cross') >= 1 # Deve ter pelo menos 1 cross edge
    

def test_dfs_deteccao_ciclo():
    """Testa a detecção de ciclo e a classificação 'back'."""
    g = Graph(directed=True)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A") # Ciclo: C -> A é uma aresta de retorno
    
    resultado = Algorithms.dfs(g, "A")
    
    classificacoes = resultado.get("classification", {}).values()
    
    assert "back" in classificacoes
    assert resultado["has_cycle"] is True