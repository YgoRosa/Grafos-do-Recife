import os
import json
from pyvis.network import Network
from typing import List, Tuple, Dict
from graphs.graph import Graph 
import webbrowser


def construir_arestas_arvore_percurso(graph, path_nodes: List[str]) -> List[Tuple[str, str, float, Dict]]:
    """
    Constrói a lista de arestas que formam o percurso a partir de uma lista de nós.
    Usa o método get_edge_data da sua classe Graph.
    """
    edges = []
    # Itera sobre os nós consecutivos no caminho
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        # O método get_edge_data retorna (peso, meta) ou None.
        data = graph.get_edge_data(u, v)

        if data is not None:
            weight, meta = data
            # Como o percurso é (u, v), a aresta deve ser salva nessa ordem (u, v)
            edges.append((u, v, weight, meta))
        else:
            # Caso não encontre a aresta (nunca deve acontecer se o Dijkstra for correto)
            print(f"[ERRO - VIZ] Aresta esperada ({u} -> {v}) não encontrada no grafo.")
            
    return edges

def visualize_path_tree(path_nodes: List[str], path_edges: List[Tuple[str, str, float, Dict]], output_file: str):
    """
    Gera a visualização interativa do subgrafo do percurso.
    """
    
    net = Network(height="750px", width="100%", 
                  directed=False, 
                  heading="Percurso: Nova Descoberta → Setúbal") 

    # 1. Adicionar Nós
    for node in path_nodes:
        # Destacar origem e destino
        color = '#38761d' if node == path_nodes[0] else \
                '#cc0000' if node == path_nodes[-1] else \
                '#3c78d8' # Cor para nós intermediários
                
        title_text = f"Bairro: **{node}**"
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=15)

    # 2. Adicionar Arestas
    
    # Parâmetros de escala:
    # Ajuste o fator e os limites conforme a faixa de pesos que você definiu na Seção 5.
    FATOR_ESCALA = 4.0 # Multiplica o peso para amplificar a diferença visual.
    MIN_WIDTH = 2.0    # Espessura mínima para qualquer aresta do caminho.
    MAX_WIDTH = 15.0   # Espessura máxima para evitar linhas muito grandes.

    for u, v, weight, meta in path_edges:
        
        # 1. Calcular a espessura proporcional ao peso
        proportional_width = weight * FATOR_ESCALA
        
        # 2. Aplicar limites
        edge_width = max(MIN_WIDTH, min(MAX_WIDTH, proportional_width))
        
        # Rótulo para o Tooltip
        label = f"Custo: {weight:.2f}"
        if "logradouro" in meta:
            label += f"<br>Via: {meta['logradouro']}"

        # Adicionar aresta com a espessura calculada
        net.add_edge(source=u, to=v, weight=weight, title=label, 
                     color='#ff9900', 
                     value=weight, 
                     width=edge_width # <-- A espessura agora é dinâmica!
                    )

    # 3. Salvar o HTML, forçando a codificação UTF-8 para evitar o UnicodeEncodeError
    
    # 3a. Gera o HTML do gráfico como uma string (Notebook=False para HTML autônomo)
    html_content = net.generate_html(notebook=False)
    
    # 3b. Escreve a string em um arquivo, especificando explicitamente a codificação UTF-8
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        
        # Opcional: tentar abrir o navegador, mas a escrita é o que importa para o projeto
        # import webbrowser
        # webbrowser.open(output_file)

        print(f"[VIZ] Árvore de percurso interativa gerada → {output_file}")

        webbrowser.open(output_file)
    except Exception as e:
        print(f"[ERRO DE ESCRITA] Não foi possível salvar ou escrever o arquivo HTML: {e}")