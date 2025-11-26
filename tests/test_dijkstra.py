# test_dijkstra.py
from src.graphs.algorithms import Algorithms
from src.graphs.graph import Graph

import pytest

def test_dijkstra_caminho_correto_pesos_positivos():
    """Testa se Dijkstra encontra o caminho mais curto com pesos positivos."""
    g = Graph(directed=True)
    g.add_edge("A", "B", 10)
    g.add_edge("A", "C", 3)
    g.add_edge("C", "B", 4)
    g.add_edge("B", "D", 2)
    g.add_edge("C", "D", 8) 
    
    # Caminho mais curto A -> D é: A -> C (3) -> B (4) -> D (2) = 9
    resultado = Algorithms.dijkstra(g, "A", "D")
    
    assert resultado["cost"] == 9.0
    assert resultado["path"] == ["A", "C", "B", "D"]
    

def test_dijkstra_ignora_peso_negativo_e_calcula_caminho():
    """
    Testa se Dijkstra IGNORA (não usa) arestas negativas, conforme a sua implementação atual.
    O requisito é 'recusar dado com peso negativo', e a sua função atual ignora-o via 'continue'.
    """
    g = Graph(directed=True)
    g.add_edge("A", "B", 10) # Caminho direto (custo 10)
    g.add_edge("A", "C", 5)  # Rota 2, Parte 1
    g.add_edge("C", "B", -3) # Rota 2, Parte 2 (com peso NEGATIVO)

    # Rota 2 deveria ser 5 + (-3) = 2. Como a sua função ignora -3,
    # A única rota válida considerada é A -> B com custo 10.
    resultado = Algorithms.dijkstra(g, "A", "B") 

    assert resultado["cost"] == 10.0 
    assert resultado["path"] == ["A", "B"]