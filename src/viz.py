import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------
# Função 1: carregar grafo
# ----------------------------
def build_graph_from_adjlist(adj_dict):
    """
    Recebe um dicionário:
        { "Boa Viagem": {"Ibura": 2.0, "Pina": 1.0}, ... }
    e retorna um graph NetworkX.
    """
    G = nx.Graph()
    for origem, vizinhos in adj_dict.items():
        for destino, peso in vizinhos.items():
            G.add_edge(origem, destino, weight=float(peso))
    return G


# ----------------------------
# Função 2: desenhar grafo simples
# ----------------------------
def draw_graph(G, output_path):
    """
    Desenha o grafo inteiro e salva em PNG.
    """
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 10))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=800,
        font_size=8
    )

    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Função 3: desenhar ÁRVORE do percurso
# ----------------------------
def draw_path_tree(adj_path, route_json_path, output_png):
    """
    Gera um subgrafo APENAS com as arestas do percurso (árvore usada no caminho).
    O JSON do percurso vem do seu Dijkstra/solve.py:

    {
      "path": ["A", "B", "C", "D"],
      "edges": [["A","B"], ["B","C"], ["C","D"]]
    }
    """

    # ---- 1. carregar adjacências ----
    with open(adj_path, "r", encoding="utf-8") as f:
        adj_dict = {}
        for line in f.readlines()[1:]:
            o, d, w = [x.strip() for x in line.split(",")]
            w = float(w)
            if o not in adj_dict:
                adj_dict[o] = {}
            if d not in adj_dict:
                adj_dict[d] = {}
            adj_dict[o][d] = w
            adj_dict[d][o] = w

    # grafo completo
    G_full = build_graph_from_adjlist(adj_dict)

    # ---- 2. carregar percurso ----
    with open(route_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edges", [])
    if not edges:
        raise ValueError("JSON não contém 'edges'. Gere o percurso primeiro.")

    # subgrafo só com as arestas do percurso
    G_sub = nx.Graph()
    for u, v in edges:
        peso = G_full[u][v]["weight"]
        G_sub.add_edge(u, v, weight=peso)

    # ---- 3. desenhar ----
    pos = nx.spring_layout(G_sub, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw(
        G_sub,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=1200,
        font_size=9,
        edge_color="red",
        width=2.5
    )

    edge_labels = nx.get_edge_attributes(G_sub, "weight")
    nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=8)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Função 4: helper para CLI
# ----------------------------
def generate_route_tree(adj_csv, route_json, output_png):
    """
    Função simples para ser chamada pelo CLI:
    python cli.py arvore --adj data/adjacencias_bairros.csv --route out/percurso.json --out out/tree.png
    """
    draw_path_tree(adj_csv, route_json, output_png)
