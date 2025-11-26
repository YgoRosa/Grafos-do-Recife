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
        size = 14 + (degree * 2)
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
    map_micro: Dict[str, str],
    caminho_obrig: list,
    output_file: str
):
    import math, json, webbrowser, os
    print("[VIZ] Gerando grafo interativo (layout circular, busca exata)...")

    # ----- MAPAS AUXILIARES -----
    grau_map = df_ego.set_index("bairro")["grau"].to_dict() if "grau" in df_ego.columns else {}
    dens_map = df_ego.set_index("bairro")["densidade_ego"].to_dict() if "densidade_ego" in df_ego.columns else {}

    nodes_list = sorted(graph.get_nodes())
    N = len(nodes_list)

    # ----- LAYOUT CIRCULAR -----
    RADIUS = 2200
    pos_map = {}
    for i, node in enumerate(nodes_list):
        ang = 2 * math.pi * i / N
        pos_map[node] = (RADIUS * math.cos(ang), RADIUS * math.sin(ang))

    # ----- CORES VIVAS -----
    MICRO_COLORS = {
        "1": "rgba(255, 140, 140, 0.85)",
        "2": "rgba(255, 190, 120, 0.85)",
        "3": "rgba(255, 240, 130, 0.85)",
        "4": "rgba(160, 230, 140, 0.85)",
        "5": "rgba(140, 190, 255, 0.85)",
        "6": "rgba(200, 140, 255, 0.85)",
    }

    from pyvis.network import Network
    net = Network(height="900px", width="100%", directed=False, notebook=False,
                  heading="Grafo dos Bairros do Recife — Interativo")

    # ----- ADICIONAR NÓS -----
    for node in nodes_list:
        grau = grau_map.get(node, 0)
        dens = dens_map.get(node)
        micro_raw = str(map_micro.get(node, ""))
        macro = micro_raw.split(".")[0] if "." in micro_raw else micro_raw

        color = MICRO_COLORS.get(macro, "rgba(200,200,200,0.5)")
        if node in caminho_obrig:
            color = "#ffcc00"

        size = 14 + min(grau, 20)

        tooltip = f"<b>{node}</b><br>Microrregião: {micro_raw}<br>Grau: {grau}"
        if dens is not None:
            tooltip += f"<br>Densidade ego: {float(dens):.4f}"

        x, y = pos_map[node]

        net.add_node(
            n_id=node,
            label=node,
            title=tooltip,
            color=color,
            originalColor=color,   # <-- usado no reset
            size=size,
            x=x, y=y,
            physics=False
        )

    # ----- ADICIONAR ARESTAS -----
    edge_color = "#cccccc"
    adj_map = {}

    for u, v, weight, meta in graph.get_edges():
        adj_map.setdefault(u, []).append(v)
        adj_map.setdefault(v, []).append(u)

        title = f"Peso: {weight}"
        if meta and meta.get("logradouro"):
            title += f"<br>Via: {meta['logradouro']}"

        net.add_edge(
            source=u, to=v,
            color=edge_color,
            width=1,
            value=float(weight) if weight else 1,
            smooth={"enabled": False},
            physics=False,
            title=title
        )

    html_str = net.generate_html()

    # ----- JS EXTRA -----
    adj_json = json.dumps(adj_map, ensure_ascii=False)
    path_json = json.dumps(caminho_obrig or [], ensure_ascii=False)

    js = f"""
        <style>
            #searchBox {{ position:absolute; top:10px; left:10px; z-index:9999; }}
            #highlightBtn {{ position:absolute; top:10px; left:300px; z-index:9999; }}
            #infoBox {{ position:absolute; top:50px; left:10px; z-index:9999; background:#fff; padding:6px; border-radius:4px; }}
        </style>

        <div id="searchBox">
            <input id="nodeSearch" placeholder="Buscar bairro..." style="padding:6px; width:220px;">
            <button onclick="doSearch()">Ir</button>
            <button onclick="resetHighlight()">Reset</button>
        </div>

        <div id="highlightBtn">
            <button onclick="highlightPath()">Destacar: Nova Descoberta → Boa Viagem (Setúbal)</button>
        </div>

        <div id="infoBox">Nós no caminho obrigatório: {len(caminho_obrig)}</div>

        <script>
            const ADJ = {adj_json};
            const PATH = {path_json};

            // RESET --------
            function resetHighlight() {{
                const nodes = network.body.data.nodes.get();
                const edges = network.body.data.edges.get();

                nodes.forEach(n => {{
                    network.body.data.nodes.update({{
                        id: n.id,
                        color: n.originalColor,
                        size: undefined
                    }});
                }});

                edges.forEach(e => {{
                    network.body.data.edges.update({{
                        id: e.id,
                        color: '{edge_color}', width: 1
                    }});
                }});
            }}

            // DESTACAR BAIRRO + ARESTAS / VIZINHOS -----
            function highlightNodeAndNeighbors(id) {{
                resetHighlight();

                const neigh = ADJ[id] || [];

                network.body.data.nodes.update({{id:id, color:'#ffd24d', size:26}});

                neigh.forEach(nid => {{
                    network.body.data.nodes.update({{id:nid, color:'#7fb3ff', size:18}});
                }});

                network.body.data.edges.get().forEach(e => {{
                    if ((e.from===id && neigh.includes(e.to)) || 
                        (e.to===id && neigh.includes(e.from))) {{

                        network.body.data.edges.update({{
                            id:e.id, color:'#1f77b4', width:4
                        }});
                    }}
                }});

                network.focus(id, {{scale:1.4, animation:{{duration:300}}}});
            }}

            // BUSCA EXATA --------
            function doSearch() {{
                const q = document.getElementById("nodeSearch").value.toLowerCase().trim();
                if (!q) return;

                const nodes = network.body.data.nodes.get();
                const exact = nodes.find(n => n.label.toLowerCase() === q);

                if (!exact) {{
                    alert("Bairro não encontrado: " + q);
                    return;
                }}

                highlightNodeAndNeighbors(exact.id);
            }}

            // DESTACAR CAMINHO OBRIGATÓRIO --------
            function highlightPath() {{
                resetHighlight();
                if (PATH.length < 2) return;

                for (let i=0; i<PATH.length; i++) {{
                    network.body.data.nodes.update({{
                        id: PATH[i], color:'#ffd24d', size:26
                    }});
                    if (i < PATH.length-1) {{
                        const a = PATH[i], b = PATH[i+1];
                        network.body.data.edges.get().forEach(e => {{
                            if ((e.from===a && e.to===b) || (e.from===b && e.to===a)) {{
                                network.body.data.edges.update({{
                                    id:e.id, color:'#ff3333', width:6
                                }});
                            }}
                        }});
                    }}
                }}
                network.focus(PATH[0], {{scale:1.2, animation:{{duration:300}}}});
            }}

            // CLICK --------
            network.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    highlightNodeAndNeighbors(params.nodes[0]);
                }}
            }});
        </script>
    """


    html_str = html_str.replace("</body>", js + "</body>")

    # salvar
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[OK] Grafo interativo salvo em {output_file}")
    try:
        webbrowser.open(output_file)
    except:
        pass
