# test_bellman_ford.py
from src.graphs.algorithms import Algorithms
from src.graphs.graph import Graph

import pytest

def test_bellman_ford_pesos_negativos_sem_ciclo():
    """(i) Testa Bellman-Ford com pesos negativos sem ciclo negativo."""
    # O grafo deve ser DIRECIONADO para evitar a criação acidental de ciclo B <-> C.
    g = Graph(directed=True) 
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", -2) # Peso negativo!
    g.add_edge("A", "C", 4)
    
    # A -> C: 4
    # A -> B -> C: 1 + (-2) = -1 (Mais curto)
    resultado = Algorithms.bellman_ford(g, "A", "C")
    
    # Requisito (i): distâncias corretas
    assert resultado["cost"] == -1.0
    assert resultado["path"] == ["A", "B", "C"]
    
    # Requisito (i): sem ciclo negativo
    assert resultado["negative_cycle"] is False
    

def test_bellman_ford_detecta_ciclo_negativo():
    """(ii) Testa Bellman-Ford com ciclo negativo para verificar a flag."""
    g = Graph(directed=True) 
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", -2)
    g.add_edge("C", "A", 0.5) # Ciclo A->B->C->A, peso total: 1 + (-2) + 0.5 = -0.5 (Negativo)
    
    resultado = Algorithms.bellman_ford(g, "A", "C")
    
    # Requisito (ii): flag deve ser True
    assert resultado["negative_cycle"] is True
    assert resultado["path"] == []