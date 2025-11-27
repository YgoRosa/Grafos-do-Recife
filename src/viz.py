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

def visualize_path_tree(path_nodes: list, path_edges: list, output_file: str):
    """
    Gera a visualização interativa do subgrafo do percurso.
    """
    import os, webbrowser
    from pyvis.network import Network

    print(f"[VIZ] Gerando Árvore de Percurso → {output_file}")

    net = Network(height="750px", width="100%", directed=False, heading="") 

    # ----- Adicionar nós -----
    for node in path_nodes:
        color = '#38761d' if node == path_nodes[0] else \
                '#cc0000' if node == path_nodes[-1] else \
                '#3c78d8'
        title_text = f"<b>{node}</b>"
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=15, originalColor=color, physics=False)

    # ----- Adicionar arestas -----
    FATOR_ESCALA = 4.0
    MIN_WIDTH = 2.0
    MAX_WIDTH = 15.0

    for u, v, weight, meta in path_edges:
        proportional_width = weight * FATOR_ESCALA
        edge_width = max(MIN_WIDTH, min(MAX_WIDTH, proportional_width))
        label = f"Custo: {weight:.2f}"
        if meta and "logradouro" in meta:
            label += f"<br>Via: {meta['logradouro']}"
        net.add_edge(source=u, to=v, weight=weight, title=label,
                     color='#ff9900', value=weight, width=edge_width, smooth={"enabled": False}, physics=False)

    # ----- Gerar HTML -----
    html_content = net.generate_html(notebook=False)

    # ----- Inserir título estilo HTML -----
    header_html = """
    <style>
    #pageTitle {
        text-align: center;
        margin-top: 25px;
        margin-bottom: 20px;
        font-family: "Segoe UI", Arial, sans-serif;
    }
    #pageTitle .title-main {
        display: block;
        font-size: 30px;
        font-weight: 800;
        color: #1f3b5c;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.12);
        letter-spacing: 0.5px;
    }
    #pageTitle .title-sub {
        display: block;
        margin-top: 6px;
        font-size: 17px;
        color: #4a90e2;
        font-weight: 500;
        text-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    #mynetwork {
        width: 100%;
        height: 750px;
        background-color: #ffffff;
        border: 1px solid lightgray;
        position: relative;
        float: left;
    }
    </style>

    <div id="pageTitle">
        <span class="title-main">Percurso Entre Bairros</span>
        <span class="title-sub">Nova Descoberta → Setúbal</span>
    </div>
    """

    html_content = html_content.replace('<div id="mynetwork"', f'{header_html}\n<div id="mynetwork"')

    # ----- Salvar e abrir -----
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Árvore de percurso interativa gerada → {output_file}")
        try:
            webbrowser.open(output_file)
        except:
            pass
    except Exception as e:
        print(f"[ERRO DE ESCRITA] Não foi possível salvar arquivo HTML: {e}")


def visualize_degree_map(graph: Graph, df_graus: pd.DataFrame, output_file: str):
    """
    1. Mapa de Cores por Grau do Bairro
    """
    import os, webbrowser, math
    from pyvis.network import Network

    print(f"[VIZ] Gerando Mapa de Cores por Grau → {output_file}")

    max_degree = df_graus['grau'].max()
    min_degree = df_graus['grau'].min()
    degree_range = max_degree - min_degree
    degree_map = df_graus.set_index('bairro')['grau'].to_dict()

    def degree_to_color(degree):
        if degree_range == 0:
            norm_degree = 0.5
        else:
            norm_degree = (degree - min_degree) / degree_range
        r = int(255 * (1 - norm_degree))
        g = int(255 * norm_degree)
        b = int(100 * (1 - norm_degree))
        return f'#{r:02x}{g:02x}{b:02x}'

    # ----- NETWORK -----
    net = Network(height="750px", width="100%", directed=False, heading="")

    # Layout circular
    nodes_list = sorted(graph.get_nodes())
    N = len(nodes_list)
    RADIUS = 1500
    pos_map = {node: (RADIUS * math.cos(2 * math.pi * i / N),
                      RADIUS * math.sin(2 * math.pi * i / N))
               for i, node in enumerate(nodes_list)}

    # Adicionar nós
    for node in nodes_list:
        degree = degree_map.get(node, 0)
        color = degree_to_color(degree)
        size = 14 + (degree * 2)
        x, y = pos_map[node]
        title_text = f"<b>{node}</b><br>Grau: {degree}"
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=size,
                     x=x, y=y, physics=False, originalColor=color)

    # Adicionar arestas
    for u, v, weight, meta in graph.get_edges():
        net.add_edge(source=u, to=v, color='#999999', width=1, value=weight,
                     title=f"Custo: {weight:.2f}", smooth={"enabled": False}, physics=False)

    html_content = net.generate_html(notebook=False)

    # Adicionar título estilo HTML
    header_html = """
    <style>
    #pageTitle {
        text-align: center;
        margin-top: 25px;
        margin-bottom: 20px;
        font-family: "Segoe UI", Arial, sans-serif;
    }
    #pageTitle .title-main {
        display: block;
        font-size: 30px;
        font-weight: 800;
        color: #1f3b5c;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.12);
        letter-spacing: 0.5px;
    }
    #mynetwork {
        width: 100%;
        height: 750px;
        background-color: #ffffff;
        border: 1px solid lightgray;
        position: relative;
        float: left;
    }
    </style>

    <div id="pageTitle">
        <span class="title-main">Mapa de Graus</span>
    </div>
    """

    html_content = html_content.replace('<div id="mynetwork"', f'{header_html}\n<div id="mynetwork"')

    # Salvar e abrir
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Visualização de Grau gerada em {output_file}")
        try:
            webbrowser.open(output_file)
        except:
            pass
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
    import os, webbrowser, math
    from pyvis.network import Network

    print(f"[VIZ] Gerando Subgrafo Top 10 por Grau → {output_file}")

    # Seleciona os top 10 bairros por grau
    top_10_df = df_graus.sort_values(by='grau', ascending=False).head(10)
    top_nodes = set(top_10_df['bairro'].tolist())
    top_degree_map = top_10_df.set_index('bairro')['grau'].to_dict()

    if not top_nodes:
        print("[VIZ] Nenhum nó encontrado para o subgrafo.")
        return

    # ----- NETWORK -----
    net = Network(height="750px", width="100%", directed=False, heading="")
    
    # Layout circular
    nodes_list = sorted(top_nodes)
    N = len(nodes_list)
    RADIUS = 1200
    pos_map = {node: (RADIUS * math.cos(2 * math.pi * i / N),
                      RADIUS * math.sin(2 * math.pi * i / N))
               for i, node in enumerate(nodes_list)}

    max_degree = top_10_df['grau'].max() if not top_10_df.empty else 1

    # ----- ADICIONAR NÓS -----
    for node in nodes_list:
        degree = top_degree_map.get(node, 0)
        size = 15 + (degree / max_degree) * 20
        x, y = pos_map[node]
        tooltip = f"<b>{node}</b><br>Grau: {degree}"
        net.add_node(n_id=node, label=node, title=tooltip, color='#FFC107', size=size, x=x, y=y, physics=False, originalColor='#FFC107')

    # ----- ADICIONAR ARESTAS -----
    for u, v, weight, meta in graph.get_edges():
        if u in top_nodes and v in top_nodes:
            net.add_edge(source=u, to=v, color='#FF9800', width=2, value=weight, title=f"Custo: {weight:.2f}", smooth={"enabled": False}, physics=False)

    html_content = net.generate_html(notebook=False)

    # ----- ADICIONAR TÍTULO ESTILO HTML -----
    header_html = """
    <style>
    #pageTitle {
        text-align: center;
        margin-top: 25px;
        margin-bottom: 20px;
        font-family: "Segoe UI", Arial, sans-serif;
    }
    #pageTitle .title-main {
        display: block;
        font-size: 30px;
        font-weight: 800;
        color: #1f3b5c;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.12);
        letter-spacing: 0.5px;
    }
    #mynetwork {
        width: 100%;
        height: 750px;
        background-color: #ffffff;
        border: 1px solid lightgray;
        position: relative;
        float: left;
    }
    </style>

    <div id="pageTitle">
        <span class="title-main">Subgrafo dos 10 Bairros mais conectados</span>
    </div>
    """

    html_content = html_content.replace('<div id="mynetwork"', f'{header_html}\n<div id="mynetwork"')

    # ----- SALVAR E ABRIR -----
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Subgrafo Top 10 gerado em {output_file}")
        try:
            webbrowser.open(output_file)
        except:
            pass
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

    degrees = df_graus_p2['grau_saida'].dropna().astype(int).tolist()
    if not degrees:
        print("[ERRO] Sem graus para plotar.")
        return

    plt.figure(figsize=(10, 6))

    # histograma com bins automáticos e escala log
    plt.hist(
        degrees,
        bins='auto',          # deixa o numpy escolher (melhor p/ distribuição grande)
        color='#CC5500',
        edgecolor='black',
    )

    plt.yscale('log')         # <<< transforma o gráfico, fica perfeito
    plt.title('Distribuição de Graus de Saída (Rede Aérea)', fontsize=16)
    plt.xlabel('Grau de Saída', fontsize=12)
    plt.ylabel('Frequência (escala log)', fontsize=12)

    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"[OK] Histograma de Graus P2 gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ P2] Falha ao salvar {output_file}: {e}")


def visualize_interactive_graph(
    graph,
    df_ego,
    map_micro,
    caminho_obrig,
    output_file
):
    import math, json, webbrowser, os
    from pyvis.network import Network

    print("[VIZ] Gerando grafo interativo (layout circular, busca exata)...")

    grau_map = df_ego.set_index("bairro")["grau"].to_dict() if "grau" in df_ego.columns else {}
    dens_map = df_ego.set_index("bairro")["densidade_ego"].to_dict() if "densidade_ego" in df_ego.columns else {}

    nodes_list = sorted(graph.get_nodes())
    N = len(nodes_list)

    RADIUS = 2200
    pos_map = {node: (RADIUS * math.cos(2*math.pi*i/N), RADIUS * math.sin(2*math.pi*i/N))
               for i, node in enumerate(nodes_list)}

    MICRO_COLORS = {
        "1": "rgba(255, 140, 140, 0.85)",
        "2": "rgba(255, 190, 120, 0.85)",
        "3": "rgba(255, 240, 130, 0.85)",
        "4": "rgba(160, 230, 140, 0.85)",
        "5": "rgba(140, 190, 255, 0.85)",
        "6": "rgba(200, 140, 255, 0.85)",
    }

    net = Network(height="900px", width="100%", directed=False, notebook=False, heading="")

    orig_nodes = []
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
            originalColor=color,
            size=size,
            x=x, y=y,
            physics=False
        )

        orig_nodes.append({
            "id": node, "label": node, "title": tooltip,
            "color": color, "originalColor": color, "size": size,
            "x": x, "y": y
        })

    edge_color = "#cccccc"
    adj_map = {}
    for u, v, weight, meta in graph.get_edges():
        adj_map.setdefault(u, []).append(v)
        adj_map.setdefault(v, []).append(u)
        title = f"Peso: {weight}" + (f"<br>Via: {meta['logradouro']}" if meta and meta.get("logradouro") else "")
        net.add_edge(source=u, to=v, color=edge_color, width=1,
                     value=float(weight) if weight else 1, smooth={"enabled": False},
                     physics=False, title=title)

    html_str = net.generate_html()

    header_html = """
    <style>
    #pageTitle { text-align: center; margin-top: 20px; margin-bottom: 15px; font-family: "Segoe UI", Arial, sans-serif; }
    #pageTitle .title-main { display: block; font-size: 30px; font-weight: 800; color: #1f3b5c;
                             text-shadow: 0px 2px 4px rgba(0,0,0,0.12); letter-spacing: 0.5px; }
    #pageTitle .title-sub { display: block; margin-top: 4px; font-size: 16px; color: #4a90e2;
                            font-weight: 500; text-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .navbar-custom { background-color: #4a90e2; text-align: center; font-family: "Segoe UI", Arial, sans-serif;
                      border-radius: 8px; margin: 20px auto 10px auto; }
    .navbar-custom a { display: inline-block; background-color: #4a90e2; color: white; text-decoration: none;
                       padding: 8px 16px; margin: 0 8px; border-radius: 6px; font-weight: 600;
                       transition: background 0.2s, transform 0.15s; }
    .navbar-custom a:hover { color: #1f3b5c; transform: translateY(-1px); }
    </style>

    <div id="pageTitle">
        <span class="title-main">Grafo dos Bairros do Recife</span>
        <span class="title-sub">Mapa Interativo e Navegável</span>
    </div>

    <nav class="navbar-custom">
        <a href="arvore_percurso.html">Árvore Nova Descoberta - Setúbal</a>
        <a href="mapa_grau.html">Mapa de Graus</a>
        <a href="subgrafo_top10.html">Bairros mais conectados</a>
    </nav>
    """
    html_str = html_str.replace('<div id="mynetwork"', f'{header_html}\n<div id="mynetwork"')

    adj_json = json.dumps(adj_map, ensure_ascii=False)
    path_json = json.dumps(caminho_obrig or [], ensure_ascii=False)
    nodes_json = json.dumps(orig_nodes, ensure_ascii=False)  # NÓS ORIGINAIS

    js_panel = f"""
    <style>
    #uiPanel {{ position:absolute; top:20px; left:20px; z-index:9999; background:#ffffffd9;
                backdrop-filter: blur(6px); padding:16px 18px; width:260px; border-radius:14px;
                box-shadow:0 6px 18px rgba(0,0,0,0.20); font-family: Arial,sans-serif; }}
    #uiTitle {{ font-size:17px; font-weight:bold; margin-bottom:10px; color:#1f3b5c; }}
    #uiPanel input {{ width:100%; padding:8px 10px; border-radius:8px; border:1px solid #d0d0d0;
                     margin-bottom:8px; outline:none; transition:0.2s; }}
    #uiPanel input:focus {{ border-color:#4a90e2; box-shadow:0 0 5px #4a90e255; }}
    #uiPanel button {{ width:100%; padding:8px; margin-top:6px; background:#4a90e2; color:white;
                       border:none; border-radius:8px; cursor:pointer; font-size:14px; font-weight:bold;
                       transition: background .25s, transform .15s; }}
    #uiPanel button:hover {{ background:#357acb; transform:translateY(-1px); }}
    #uiPanel button:active {{ transform:translateY(0px); }}
    .btn-secondary {{ background:#666; }}
    .btn-secondary:hover {{ background:#555; }}
    </style>

    <div id="uiPanel">
        <div id="uiTitle">🔍 Controles do Mapa</div>
        <input id="nodeSearch" placeholder="Buscar bairro...">
        <button onclick="doSearch()">Buscar</button>
        <button onclick="resetHighlight()" class="btn-secondary">Resetar Destaques</button>
        <button onclick="highlightPath()">⭐ Destacar Caminho Obrigatório</button>
    </div>

    <script>
    const ADJ = {adj_json};
    const PATH = {path_json};
    const ORIG_NODES = {nodes_json};
    const edgeColor = '{edge_color}';

    function resetHighlight(){{
        network.body.data.nodes.clear();
        ORIG_NODES.forEach(n => network.body.data.nodes.add(n));
        network.body.data.edges.get().forEach(e => network.body.data.edges.update({{id:e.id, color:edgeColor, width:1}}));
    }}

    function highlightNodeAndNeighbors(id){{
        resetHighlight();
        const neigh = ADJ[id] || [];
        network.body.data.nodes.update({{id:id, color:'#ffd24d', size:26}});
        neigh.forEach(nid => network.body.data.nodes.update({{id:nid, color:'#7fb3ff', size:18}}));
        network.body.data.edges.get().forEach(e => {{
            if((e.from===id && neigh.includes(e.to))||(e.to===id && neigh.includes(e.from))) {{
                network.body.data.edges.update({{id:e.id, color:'#1f77b4', width:4}});
            }}
        }});
        network.focus(id, {{scale:1.4, animation:{{duration:300}}}});
    }}

    function doSearch(){{
        const q=document.getElementById("nodeSearch").value.toLowerCase().trim();
        if(!q) return;
        const nodes=network.body.data.nodes.get();
        const exact=nodes.find(n=>n.label.toLowerCase()===q);
        if(!exact) return alert("Bairro não encontrado: "+q);
        highlightNodeAndNeighbors(exact.id);
    }}

    function highlightPath(){{
        resetHighlight();
        if(PATH.length<2) return;
        for(let i=0;i<PATH.length;i++){{
            network.body.data.nodes.update({{id:PATH[i], color:'#ffd24d', size:26}});
            if(i<PATH.length-1){{
                const a=PATH[i], b=PATH[i+1];
                network.body.data.edges.get().forEach(e=>{{
                    if((e.from===a && e.to===b)||(e.from===b && e.to===a)){{
                        network.body.data.edges.update({{id:e.id, color:'#ff3333', width:6}});
                    }}
                }});
            }}
        }}
        network.focus(PATH[0], {{scale:1.2, animation:{{duration:300}}}});
    }}

    network.on("click", function(params){{
        if(params.nodes.length>0) highlightNodeAndNeighbors(params.nodes[0]);
    }});
    </script>
    """

    html_str = html_str.replace("</body>", js_panel + "</body>")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[OK] Grafo interativo salvo em {output_file}")
    try: webbrowser.open(output_file)
    except: pass

