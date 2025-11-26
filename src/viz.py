import os
import json
import pandas as pd
from pyvis.network import Network
from typing import List, Tuple, Dict
from graphs.graph import Graph 
import webbrowser
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.colors as mcolors


def construir_arestas_arvore_percurso(graph, path_nodes: List[str]) -> List[Tuple[str, str, float, Dict]]:
    """
    Constrói a lista de arestas que formam o percurso a partir de uma lista de nós.
    """
    edges = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        data = graph.get_edge_data(u, v)

        if data is not None:
            weight, meta = data
            edges.append((u, v, weight, meta))
        else:
            print(f"[ERRO - VIZ] Aresta esperada ({u} -> {v}) não encontrada no grafo.")
            
    return edges

def visualize_path_tree(path_nodes: List[str], path_edges: List[Tuple[str, str, float, Dict]], output_file: str):
    """
    Gera a visualização interativa do subgrafo do percurso.
    """
    net = Network(height="750px", width="100%", directed=False, heading="Percurso: Nova Descoberta → Setúbal") 

    # 1. Adicionar Nós
    for node in path_nodes:
        color = '#38761d' if node == path_nodes[0] else \
                '#cc0000' if node == path_nodes[-1] else \
                '#3c78d8'
        title_text = f"Bairro: **{node}**"
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=15)

    # 2. Adicionar Arestas
    FATOR_ESCALA = 4.0 
    MIN_WIDTH = 2.0
    MAX_WIDTH = 15.0

    for u, v, weight, meta in path_edges:
        proportional_width = weight * FATOR_ESCALA
        edge_width = max(MIN_WIDTH, min(MAX_WIDTH, proportional_width))
        
        label = f"Custo: {weight:.2f}"
        if "logradouro" in meta:
            label += f"<br>Via: {meta['logradouro']}"

        net.add_edge(source=u, to=v, weight=weight, title=label, 
                     color='#ff9900', value=weight, width=edge_width)

    # 3. Salvar HTML
    html_content = net.generate_html(notebook=False)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[VIZ] Árvore de percurso interativa gerada → {output_file}")
        webbrowser.open(output_file)
    except Exception as e:
        print(f"[ERRO DE ESCRITA] Não foi possível salvar arquivo HTML: {e}")

def visualize_degree_map(graph: Graph, df_graus: pd.DataFrame, output_file: str):
    """
    1. Mapa de Cores por Grau do Bairro
    """
    print(f"[VIZ] Gerando Mapa de Cores por Grau → {output_file}")
    net = Network(height="750px", width="100%", directed=False, heading="Grafo de Bairros do Recife: Visualização de Grau") 

    max_degree = df_graus['grau'].max()
    min_degree = df_graus['grau'].min()
    degree_range = max_degree - min_degree
    degree_map = df_graus.set_index('bairro')['grau'].to_dict()

    def degree_to_color(degree):
        if degree_range == 0: norm_degree = 0.5
        else: norm_degree = (degree - min_degree) / degree_range
        r = int(255 * (1 - norm_degree))
        g = int(255 * norm_degree)
        b = int(100 * (1 - norm_degree))
        return f'#{r:02x}{g:02x}{b:02x}'

    for node in graph.get_nodes():
        degree = degree_map.get(node, 0)
        color = degree_to_color(degree)
        size = 10 + (degree * 2)
        title_text = f"Bairro: **{node}**<br>Grau: {degree}"
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=size)

    for u, v, weight, meta in graph.get_edges():
        net.add_edge(source=u, to=v, weight=weight, title=f"Custo: {weight:.2f}", color='#999999', width=1)

    html_content = net.generate_html(notebook=False)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Visualização de Grau gerada em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")


def visualize_degree_histogram(df_graus: pd.DataFrame, output_file: str):
    """
    2. Distribuição dos Graus
    """
    print(f"[VIZ] Gerando Histograma de Graus → {output_file}")
    degrees = df_graus['grau'].dropna().tolist()
    if not degrees: return

    plt.figure(figsize=(10, 6))
    bins = np.arange(min(degrees), max(degrees) + 1.5) - 0.5
    plt.hist(degrees, bins=bins, color='#4682B4', edgecolor='black', rwidth=0.9)
    plt.title('Distribuição de Graus dos Bairros do Recife', fontsize=16)
    plt.xlabel('Grau (Número de Interconexões)', fontsize=12)
    plt.ylabel('Frequência (Número de Bairros)', fontsize=12)
    plt.xticks(np.arange(min(degrees), max(degrees) + 1))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    try:
        plt.savefig(output_file)
        plt.close()
        print(f"[OK] Histograma de Graus gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")


def visualize_top_10_degree_subgraph(graph: Graph, df_graus: pd.DataFrame, output_file: str):
    """
    3. Subgrafo dos 10 Bairros com Maior Grau
    """
    print(f"[VIZ] Gerando Subgrafo Top 10 por Grau → {output_file}")
    top_10_df = df_graus.sort_values(by='grau', ascending=False).head(10)
    top_nodes = set(top_10_df['bairro'].tolist())
    top_degree_map = top_10_df.set_index('bairro')['grau'].to_dict()

    if not top_nodes: return

    net = Network(height="750px", width="100%", directed=False, heading="Subgrafo dos 10 Bairros Mais Conectados") 
    max_degree = top_10_df['grau'].max() if not top_10_df.empty else 1
    
    for node in top_nodes:
        degree = top_degree_map.get(node, 0)
        size = 15 + (degree / max_degree) * 20
        title_text = f"Bairro: **{node}**<br>Grau: {degree}"
        net.add_node(n_id=node, label=node, title=title_text, color='#FFC107', size=size) 

    for u, v, weight, meta in graph.get_edges():
        if u in top_nodes and v in top_nodes:
             net.add_edge(source=u, to=v, weight=weight, title=f"Custo: {weight:.2f}", color='#FF9800', value=weight, width=2)

    html_content = net.generate_html(notebook=False)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Subgrafo Top 10 gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")

def compute_parte2_degrees(g: Graph) -> pd.DataFrame:
    rows = []
    for node in g.get_nodes():
        grau_saida = g.degree(node) 
        rows.append({"aeroporto": node, "grau_saida": grau_saida})
    return pd.DataFrame(rows, columns=["aeroporto", "grau_saida"])

def visualize_parte2_degree_histogram(df_graus_p2: pd.DataFrame, output_file: str):
    print(f"[VIZ] Gerando Histograma de Graus (Parte 2) → {output_file}")
    degrees = df_graus_p2['grau_saida'].dropna().tolist()
    if not degrees: return

    plt.figure(figsize=(10, 6))
    bins = np.arange(min(degrees), max(degrees) + 1.5) - 0.5
    plt.hist(degrees, bins=bins, color='#CC5500', edgecolor='black', rwidth=0.9)
    plt.title('Distribuição de Graus de Saída (Rede Aérea)', fontsize=16)
    plt.xlabel('Grau de Saída', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(output_file)
        plt.close()
        print(f"[OK] Histograma de Graus P2 gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ P2] Falha ao salvar {output_file}: {e}")

def visualize_interactive_graph(
    graph: Graph,
    df_ego: pd.DataFrame,
    map_micro: Dict[str, str], # ALTERADO: Agora aceita o dicionário vindo do solve.py
    caminho_obrig: list,
    output_file: str
):
    """
    Versão robusta do grafo interativo corrigida para aceitar map_micro.
    """
    print("[VIZ] Gerando grafo interativo completo...")

    # --- Preparar dicionários ---
    # Pegamos o GRAU diretamente do df_ego (que tem a coluna 'grau')
    grau_map = {}
    if "grau" in df_ego.columns:
        grau_map = df_ego.set_index("bairro")["grau"].to_dict()
    
    # ego_map e dens_map
    ego_map = {}
    dens_map = {}
    
    # Busca coluna de ordem_ego
    ego_col = next((c for c in ["ordem_ego", "tamanho_ego"] if c in df_ego.columns), None)
    if ego_col: ego_map = df_ego.set_index("bairro")[ego_col].to_dict()
    
    # Busca coluna de densidade
    dens_col = next((c for c in ["densidade_ego", "densidade"] if c in df_ego.columns), None)
    if dens_col: dens_map = df_ego.set_index("bairro")[dens_col].to_dict()

    # Usa o map_micro passado como argumento (não precisa carregar CSV)
    micro_map = map_micro if map_micro else {}

    # --- Construir pyvis network ---
    net = Network(height="900px", width="100%", directed=False, notebook=False,
                  heading="Grafo dos Bairros do Recife — Interativo")
    try:
        net.barnes_hut()
    except Exception: pass

    import math
    nodes_list = list(graph.get_nodes())
    N = len(nodes_list)
    pos_map = {}
    radius = 1000 

    for i, node in enumerate(nodes_list):
        angle = 2 * math.pi * i / N
        pos_map[node] = (radius * math.cos(angle), radius * math.sin(angle))

    # Adicionar nós
    for node in graph.get_nodes():
        grau = int(grau_map.get(node, 0))
        ego_val = ego_map.get(node, None)
        dens_val = dens_map.get(node, None)
        mic = micro_map.get(node, "")

        size = 12 + min(grau, 20)
        color = "#3c78d8"
        if caminho_obrig and node in caminho_obrig:
            color = "#ffcc00"
            size = max(size, 18)

        title_lines = [f"<b>{node}</b>", f"Grau: {grau}"]
        if mic: title_lines.append(f"Microrregião: {mic}")
        if ego_val is not None: title_lines.append(f"Ordem/Ego: {ego_val}")
        if dens_val is not None: title_lines.append(f"Densidade ego: {float(dens_val):.4f}")

        tooltip = "<br>".join(title_lines)
        x, y = pos_map.get(node, (0,0))
        
        net.add_node(n_id=node, label=node, title=tooltip, size=size, color=color, x=x, y=y, physics=False)

    # Adicionar arestas
    for u, v, weight, meta in graph.get_edges():
        meta = meta or {}
        title = f"Peso: {weight}"
        if "logradouro" in meta: title += f"<br>Via: {meta['logradouro']}"
        net.add_edge(source=u, to=v, value=weight if weight else 1.0, title=title, width=1, color="#97C2FC")

    html_str = net.generate_html()

    # Injetar JS de busca e destaque
    import json
    caminho_json = json.dumps(caminho_obrig or [], ensure_ascii=False)

    js_extra = f"""
    <style>
      #searchBox {{ position: absolute; top: 10px; left: 10px; z-index: 9999; }}
      #highlightBtn {{ position: absolute; top: 10px; left: 300px; z-index: 9999; }}
      #infoBox {{ position: absolute; top: 50px; left: 10px; z-index: 9999; background: rgba(255,255,255,0.9); padding:6px; border-radius:4px; }}
    </style>
    <div id="searchBox">
      <input id="nodeSearch" placeholder="Buscar bairro..." style="padding:6px;width:220px;" />
      <button onclick="doSearch()">Ir</button>
    </div>
    <div id="highlightBtn">
      <button onclick="highlightPath()">Destacar: Nova Descoberta → Boa Viagem (Setúbal)</button>
    </div>
    <div id="infoBox">Nós no caminho obrigatório: {len(caminho_obrig or [])}</div>
    <script>
    const requiredPath = {caminho_json};
    function doSearch() {{
        const q = document.getElementById('nodeSearch').value.trim().toLowerCase();
        if (!q) return;
        const nodes = network.body.data.nodes.get();
        const found = nodes.find(n => n.label.toLowerCase() === q) || nodes.find(n => n.label.toLowerCase().includes(q));
        if (!found) {{ alert("Bairro não encontrado: " + q); return; }}
        network.focus(found.id, {{ scale: 1.4, animation: {{ duration: 300 }} }});
        network.selectNodes([found.id]);
    }}
    function highlightPath() {{
        if (!requiredPath || requiredPath.length < 2) {{ alert("Caminho não disponível."); return; }}
        const edges = network.body.data.edges.get();
        const nodes = network.body.data.nodes.get();
        
        // Reset
        nodes.forEach(n => network.body.data.nodes.update({{id: n.id, color: "#3c78d8"}}));
        edges.forEach(e => network.body.data.edges.update({{id: e.id, color: "#97C2FC", width: 1}}));

        for (let i = 0; i < requiredPath.length; i++) {{
            network.body.data.nodes.update({{id: requiredPath[i], color: "#ffcc00", size: 25}});
            if (i < requiredPath.length - 1) {{
                const a = requiredPath[i], b = requiredPath[i+1];
                edges.forEach(e => {{
                    if ((e.from === a && e.to === b) || (e.from === b && e.to === a)) {{
                        network.body.data.edges.update({{id: e.id, color: "#ff0000", width: 5}});
                    }}
                }});
            }}
        }}
        network.focus(requiredPath[0], {{ scale: 1.3, animation: {{ duration: 300 }} }});
    }}
    </script>
    """

    if "</body>" in html_str:
        html_str = html_str.replace("</body>", js_extra + "\n</body>")
    else:
        html_str += js_extra

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"[OK] Grafo interativo salvo em {output_file}")
        webbrowser.open(output_file)
    except Exception as e:
        print(f"[ERRO] falha ao salvar {output_file}: {e}")