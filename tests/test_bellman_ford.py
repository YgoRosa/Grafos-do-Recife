from src.graphs.algorithms import Algorithms
from src.graphs.graph import Graph

import pytest

def test_bellman_ford_pesos_negativos_sem_ciclo():
    """(i) Testa Bellman-Ford com pesos negativos sem ciclo negativo."""
    g = Graph(directed=True) 
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", -2) 
    g.add_edge("A", "C", 4)
    
    resultado = Algorithms.bellman_ford(g, "A", "C")
    
    assert resultado["cost"] == -1.0
    assert resultado["path"] == ["A", "B", "C"]
    assert resultado["negative_cycle"] is False
    

def test_bellman_ford_detecta_ciclo_negativo():
    """(ii) Testa Bellman-Ford com ciclo negativo para verificar a flag."""
    g = Graph(directed=True) 
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", -2)
    g.add_edge("C", "A", 0.5) 
    
    resultado = Algorithms.bellman_ford(g, "A", "C")
    
    assert resultado["negative_cycle"] is True
    assert resultado["path"] == []