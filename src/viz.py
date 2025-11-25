import os
import json
import pandas as pd
from pyvis.network import Network
from typing import List, Tuple, Dict
from graphs.graph import Graph 
import webbrowser
import matplotlib.pyplot as plt
import numpy as np 


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

def visualize_degree_map(graph: Graph, df_graus: pd.DataFrame, output_file: str):
    """
    1. Mapa de Cores por Grau do Bairro (Visualização Analítica 1)
    Gera uma visualização do grafo onde o tamanho/cor do nó é proporcional ao seu grau.
    """
    print(f"[VIZ] Gerando Mapa de Cores por Grau → {output_file}")
    
    # Prepara a rede Pyvis
    net = Network(height="750px", width="100%", 
                  directed=False, 
                  heading="Grafo de Bairros do Recife: Visualização de Grau") 

    # Normaliza o Grau para Escala de Cores/Tamanhos
    max_degree = df_graus['grau'].max()
    min_degree = df_graus['grau'].min()
    degree_range = max_degree - min_degree
    
    # Dicionário de Grau para acesso rápido
    degree_map = df_graus.set_index('bairro')['grau'].to_dict()

    # Define a escala de cor (verde escuro para alto grau, amarelo claro para baixo)
    def degree_to_color(degree):
        if degree_range == 0:
            norm_degree = 0.5
        else:
            norm_degree = (degree - min_degree) / degree_range
        # Mapeia 0 (min_degree) para claro (amarelo) e 1 (max_degree) para escuro (verde/azul)
        # Escolha um esquema de cor BGR (R, G, B) em hexadecimal, por exemplo:
        r = int(255 * (1 - norm_degree)) # Diminui Vermelho
        g = int(255 * norm_degree)       # Aumenta Verde
        b = int(100 * (1 - norm_degree)) # Diminui Azul
        return f'#{r:02x}{g:02x}{b:02x}' # Ex: #FF0064 para baixo, #00FF00 para alto

    # 1. Adicionar Nós
    for node in graph.adj.keys():
        degree = degree_map.get(node, 0)
        color = degree_to_color(degree)
        size = 10 + (degree * 2) # Tamanho proporcional ao grau (Ajuste o fator)
        title_text = f"Bairro: **{node}**<br>Grau: {degree}"
        
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=size)

    # 2. Adicionar Arestas
    for u, v, weight, meta in graph.get_edges():
        net.add_edge(source=u, to=v, weight=weight, title=f"Custo: {weight:.2f}", color='#999999', value=weight, width=1)

    # 3. Salvar o HTML
    html_content = net.generate_html(notebook=False)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Visualização de Grau gerada em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")


def visualize_degree_histogram(df_graus: pd.DataFrame, output_file: str):
    """
    2. Distribuição dos Graus (Visualização Analítica 2)
    Gera um histograma da distribuição de graus dos bairros.
    """
    print(f"[VIZ] Gerando Histograma de Graus → {output_file}")
    
    degrees = df_graus['grau'].dropna().tolist()
    
    if not degrees:
        print("[AVISO VIZ] Não há dados de grau para gerar o histograma.")
        return

    plt.figure(figsize=(10, 6))
    
    # Calcula os bins (caixas/intervalos)
    bins = np.arange(min(degrees), max(degrees) + 1.5) - 0.5
    
    plt.hist(degrees, bins=bins, color='#4682B4', edgecolor='black', rwidth=0.9)
    
    plt.title('Distribuição de Graus dos Bairros do Recife', fontsize=16)
    plt.xlabel('Grau (Número de Interconexões)', fontsize=12)
    plt.ylabel('Frequência (Número de Bairros)', fontsize=12)
    plt.xticks(np.arange(min(degrees), max(degrees) + 1)) # Garante ticks em números inteiros
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    try:
        plt.savefig(output_file)
        plt.close() # Fecha a figura para liberar memória
        print(f"[OK] Histograma de Graus gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")


def visualize_top_10_degree_subgraph(graph: Graph, df_graus: pd.DataFrame, output_file: str):
    """
    3. Subgrafo dos 10 Bairros com Maior Grau (Visualização Analítica 3)
    Gera uma visualização contendo apenas os 10 nós de maior grau e as arestas entre eles.
    """
    print(f"[VIZ] Gerando Subgrafo Top 10 por Grau → {output_file}")
    
    # 1. Seleciona os Top 10 Bairros por Grau
    top_10_df = df_graus.sort_values(by='grau', ascending=False).head(10)
    top_nodes = set(top_10_df['bairro'].tolist())
    top_degree_map = top_10_df.set_index('bairro')['grau'].to_dict()

    if not top_nodes:
        print("[AVISO VIZ] Não há nós suficientes para o Top 10.")
        return

    # 2. Prepara a rede Pyvis para o subgrafo
    net = Network(height="750px", width="100%", 
                  directed=False, 
                  heading="Subgrafo dos 10 Bairros Mais Conectados (Maior Grau)") 
    
    # 3. Adiciona Nós (apenas o Top 10)
    max_degree = top_10_df['grau'].max() if not top_10_df.empty else 1
    
    for node in top_nodes:
        degree = top_degree_map.get(node, 0)
        size = 15 + (degree / max_degree) * 20 # Escala de 15 a 35
        title_text = f"Bairro: **{node}**<br>Grau: {degree}"
        
        # Cor de destaque para o núcleo
        net.add_node(n_id=node, label=node, title=title_text, color='#FFC107', size=size) 

    # 4. Adiciona Arestas (apenas as que conectam nós do Top 10)
    for u, v, weight, meta in graph.get_edges():
        if u in top_nodes and v in top_nodes:
             net.add_edge(source=u, to=v, weight=weight, title=f"Custo: {weight:.2f}", color='#FF9800', value=weight, width=2)

    # 5. Salvar o HTML
    html_content = net.generate_html(notebook=False)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Subgrafo Top 10 gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")

def compute_parte2_degrees(g: Graph) -> pd.DataFrame:
    """
    Calcula o Grau de Saída (Out-Degree) para o grafo dirigido da Parte 2.
    """
    rows = []
    for node in g.get_nodes():
        # g.degree(node) retorna o grau de saída em um grafo dirigido
        grau_saida = g.degree(node) 
        rows.append({"aeroporto": node, "grau_saida": grau_saida})

    df = pd.DataFrame(rows, columns=["aeroporto", "grau_saida"])
    return df

def visualize_parte2_degree_histogram(df_graus_p2: pd.DataFrame, output_file: str):
    """
    Gera um histograma da Distribuição de Graus de Saída (Out-Degree) do grafo aéreo.
    """
    print(f"[VIZ] Gerando Histograma de Graus (Parte 2) → {output_file}")
    
    degrees = df_graus_p2['grau_saida'].dropna().tolist()
    
    if not degrees:
        print("[AVISO VIZ P2] Não há dados de grau para gerar o histograma da Parte 2.")
        return

    plt.figure(figsize=(10, 6))
    
    # Calcula os bins (caixas/intervalos)
    bins = np.arange(min(degrees), max(degrees) + 1.5) - 0.5
    
    plt.hist(degrees, bins=bins, color='#CC5500', edgecolor='black', rwidth=0.9) # Cor Laranja
    
    plt.title('Distribuição de Graus de Saída (Rede Aérea)', fontsize=16)
    plt.xlabel('Grau de Saída (Número de Rotas Internacionais)', fontsize=12)
    plt.ylabel('Frequência (Número de Aeroportos)', fontsize=12)
    plt.xticks(np.arange(min(degrees), max(degrees) + 1, step=5))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    try:
        plt.savefig(output_file)
        plt.close()
        print(f"[OK] Histograma de Graus P2 gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ P2] Falha ao salvar {output_file}: {e}")