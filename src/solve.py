import csv
import json
import os
from typing import Dict, Set, Tuple, List

import pandas as pd
from graphs.graph import Graph
from graphs.algorithms import dijkstra
from graphs.algorithms import dijkstra_path
from viz import construir_arestas_arvore_percurso, visualize_path_tree


BAIRROS_CSV = "data/bairros_unique.csv"
ADJ_CSV = "data/adjacencias_bairros.csv"
OUT_DIR = "out"
OUT_GLOBAL = os.path.join(OUT_DIR, "recife_global.json")
OUT_MICRO = os.path.join(OUT_DIR, "microrregioes.json")
OUT_EGO = os.path.join(OUT_DIR, "ego_bairro.csv")


def read_bairros_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} não encontrado.")
    df = pd.read_csv(path, encoding="utf-8")
    mapping = {}
    for _, row in df.iterrows():
        b = str(row["bairro"]).strip()
        mr = str(row["microrregiao"]).strip()
        mapping[b] = mr
    return mapping


def load_adjacencias_to_graph(adj_path: str) -> Tuple[Graph, Set[str]]:

    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"{adj_path} não encontrado.")

    g = Graph()
    names = set()

    with open(adj_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {h.lower().strip(): h for h in reader.fieldnames}
        origem_col = headers.get("origem")
        destino_col = headers.get("destino")
        log_col = headers.get("logradouro") or headers.get("logradouros") or headers.get("logradouro/observacao")
        peso_col = headers.get("peso")

        if origem_col is None or destino_col is None:
            raise ValueError("Arquivo de adjacências precisa ter colunas 'origem' e 'destino'.")

        for row in reader:
            a = str(row[origem_col]).strip()
            b = str(row[destino_col]).strip()
            names.add(a); names.add(b)
            meta = {}
            if log_col and row.get(log_col) and not pd.isna(row.get(log_col)):
                meta["logradouro"] = str(row.get(log_col)).strip()
            try:
                weight = float(row[peso_col]) if peso_col and row.get(peso_col) not in (None, "") else 1.0
            except Exception:
                weight = 1.0

            g.add_edge(a, b, weight=weight, meta=meta)

    return g, names


def graph_order(g: Graph) -> int:
    return len(g)


def graph_size(g: Graph) -> int:
    return len(g.get_edges())


def density(num_nodes: int, num_edges: int) -> float:
    if num_nodes < 2:
        return 0.0
    return (2.0 * num_edges) / (num_nodes * (num_nodes - 1))


def compute_and_save_global(g: Graph, out_path: str):
    V = graph_order(g)
    E = graph_size(g)
    dens = density(V, E)
    result = {"ordem": V, "tamanho": E, "densidade": dens}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[GLOBAL] ordem={V}, tamanho={E}, densidade={dens:.6f} -> {out_path}")
    return result


def compute_and_save_microrregioes(g: Graph, bairros_map: Dict[str, str], out_path: str):
    mr_to_bairros: Dict[str, Set[str]] = {}
    for bairro, mr in bairros_map.items():
        mr_to_bairros.setdefault(mr, set()).add(bairro)

    results = []
    for mr, bairros in sorted(mr_to_bairros.items()):
        sub_edge_count = 0
        for u, v, w, meta in g.get_edges():
            if u in bairros and v in bairros:
                sub_edge_count += 1
        ordem = len(bairros)
        tamanho = sub_edge_count
        dens = density(ordem, tamanho)
        results.append({"microrregiao": mr, "ordem": ordem, "tamanho": tamanho, "densidade": dens})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[MICRORREGIÕES] geradas {len(results)} entradas -> {out_path}")
    return results


def compute_and_save_ego(g: Graph, bairros_all: Set[str], out_path: str):
    rows: List[Dict] = []
    for bairro in sorted(bairros_all):
        if g.has_node(bairro):
            viz = set(g.neighbors(bairro))
        else:
            viz = set()
        grau = len(viz)
        ego_nodes = set(viz)
        ego_nodes.add(bairro)
        ego_edge_count = 0
        for u, v, w, meta in g.get_edges():
            if u in ego_nodes and v in ego_nodes:
                ego_edge_count += 1
        ordem_ego = len(ego_nodes)
        tamanho_ego = ego_edge_count
        dens_ego = density(ordem_ego, tamanho_ego)
        rows.append({
            "bairro": bairro,
            "grau": grau,
            "ordem_ego": ordem_ego,
            "tamanho_ego": tamanho_ego,
            "densidade_ego": dens_ego
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["bairro", "grau", "ordem_ego", "tamanho_ego", "densidade_ego"])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[EGO] {len(rows)} bairros processados -> {out_path}")
    return df

def compute_and_save_graus(g: Graph, bairros_all: Set[str], out_path: str):
    rows = []
    for bairro in sorted(bairros_all):
        grau = len(g.neighbors(bairro)) if g.has_node(bairro) else 0
        rows.append({"bairro": bairro, "grau": grau})

    import pandas as pd
    df = pd.DataFrame(rows, columns=["bairro", "grau"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[GRAUS] Arquivo gerado com {len(df)} bairros -> {out_path}")
    return df

def find_topological_highlights(df_ego, df_graus):
    row_denso = df_ego.loc[df_ego["densidade_ego"].idxmax()]
    bairro_mais_denso = row_denso["bairro"]
    dens_max = row_denso["densidade_ego"]

    row_grau = df_graus.loc[df_graus["grau"].idxmax()]
    bairro_maior_grau = row_grau["bairro"]
    grau_max = row_grau["grau"]

    print("\n== Rankings Topológicos (Parte 4) ==")
    print(f"• Bairro mais denso ..........: {bairro_mais_denso}  (densidade={dens_max:.4f})")
    print(f"• Bairro com maior grau ......: {bairro_maior_grau}  (grau={grau_max})")
    print("=====================================\n")

def run_metrics():
    print("== Iniciando cálculo de métricas (Parte 3) ==")

    bairros_map = read_bairros_map(BAIRROS_CSV)
    print(f"Carregados {len(bairros_map)} bairros de {BAIRROS_CSV}")

    g, names_in_adj = load_adjacencias_to_graph(ADJ_CSV)
    print(f"Grafo montado: {len(g)} nós, {len(g.get_edges())} arestas (contagem via get_edges)")

    names_not_in_bairros = sorted(list(names_in_adj - set(bairros_map.keys())))
    if names_not_in_bairros:
        print("ATENÇÃO: os seguintes nomes aparecem em adjacencias_bairros.csv mas NÃO em bairros_unique.csv:")
        for n in names_not_in_bairros[:40]:
            print("  -", n)
        print(f"  ... total {len(names_not_in_bairros)} nomes inconsistentes.")
    else:
        print("Nomes em adjacencias OK em relação a bairros_unique.csv")

    all_bairros = set(bairros_map.keys()).union(names_in_adj)

    compute_and_save_global(g, OUT_GLOBAL)

    compute_and_save_microrregioes(g, bairros_map, OUT_MICRO)

    compute_and_save_ego(g, all_bairros, OUT_EGO)
    
    df_ego = compute_and_save_ego(g, all_bairros, OUT_EGO)   # 7) graus por bairro (Parte 4.1)
    graus_csv_path = os.path.join(OUT_DIR, "graus.csv")
    df_graus = compute_and_save_graus(g, all_bairros, graus_csv_path)

    find_topological_highlights(df_ego, df_graus)


    print("== Métricas concluídas ==")

def calcular_distancias_enderecos(graph: Graph, path_enderecos="data/enderecos.csv"):
    print("== Parte 6.2: cálculo de distâncias entre endereços ==")

    pares = []
    with open(path_enderecos, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pares.append(row)

    saida_csv = "out/distancias_enderecos.csv"
    with open(saida_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "bairro_X", "bairro_Y", "custo", "caminho"])

        for p in pares:
            origem = p["bairro_X"].strip()
            destino = p["bairro_Y"].strip()

            if origem not in graph.adj or destino not in graph.adj:
                print(f"[AVISO] Bairro desconhecido: {origem} -> {destino}")
                continue

            dist, prev = dijkstra(graph, origem)
            custo = dist.get(destino, float("inf"))

            caminho = []
            if custo < float("inf"):
                atual = destino
                while atual is not None:
                    caminho.append(atual)
                    atual = prev.get(atual)
                caminho.reverse()

            w.writerow([
                p["X"],
                p["Y"],
                origem,
                destino,
                custo,
                " -> ".join(caminho) if caminho else ""
            ])

            if origem == "Nova Descoberta" and destino == "Boa Viagem":
                with open("out/percurso_nova_descoberta_setubal.json", "w", encoding="utf-8") as jf:
                    json.dump({
                        "origem": origem,
                        "destino": destino,
                        "custo": custo,
                        "caminho": caminho
                    }, jf, ensure_ascii=False, indent=4)

    print(f"[OK] Distâncias calculadas → {saida_csv}")


if __name__ == "__main__":
  if __name__ == "__main__":
    print("== Parte 3,4,6 ==")

    bairros_map = read_bairros_map(BAIRROS_CSV)
    print(f"Carregados {len(bairros_map)} bairros de {BAIRROS_CSV}")

    g, names_in_adj = load_adjacencias_to_graph(ADJ_CSV)
    print(f"Grafo montado: {len(g)} nós, {len(g.get_edges())} arestas")

    all_bairros = set(bairros_map.keys()).union(names_in_adj)

    compute_and_save_global(g, OUT_GLOBAL)
    compute_and_save_microrregioes(g, bairros_map, OUT_MICRO)

    df_ego = compute_and_save_ego(g, all_bairros, OUT_EGO)

    graus_csv_path = os.path.join(OUT_DIR, "graus.csv")
    df_graus = compute_and_save_graus(g, all_bairros, graus_csv_path)
    find_topological_highlights(df_ego, df_graus)

    calcular_distancias_enderecos(g, path_enderecos="data/enderecos.csv")

    calcular_distancias_enderecos(g, path_enderecos="data/enderecos.csv")

    # ========================================
    # PARTE 7: Transforme o percurso em árvore
    # ========================================
    
    # 1. Recarregar o caminho obrigatório (Nova Descoberta -> Boa Viagem)
    try:
        path_data_file = os.path.join(OUT_DIR, "percurso_nova_descoberta_setubal.json")
        with open(path_data_file, "r", encoding="utf-8") as f:
            path_data = json.load(f)
        caminho_obrig = path_data.get("caminho", [])
        
        # O requisito pede Boa Viagem (Setúbal), que foi mapeado para Boa Viagem
        origem = caminho_obrig[0] if caminho_obrig else "Nova Descoberta"
        destino = caminho_obrig[-1] if caminho_obrig else "Boa Viagem"

        if caminho_obrig and len(caminho_obrig) > 1:
            print(f"\n== Parte 7: Visualizando percurso de {origem} a {destino} ==")
            
            # 2. Construir as arestas do subgrafo
            edges_path = construir_arestas_arvore_percurso(g, caminho_obrig)
            
            # 3. Gerar a visualização
            output_html = os.path.join(OUT_DIR, "arvore_percurso.html")
            visualize_path_tree(caminho_obrig, edges_path, output_html)
        else:
            print(f"[AVISO] Caminho obrigatório ({origem} -> {destino}) não encontrado ou muito curto em {path_data_file}. (Caminho: {caminho_obrig})")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo de percurso obrigatório ({path_data_file}) não encontrado. Certifique-se de que 'calcular_distancias_enderecos' foi executado corretamente.")

    print("== Execução finalizada ==")

# ========================================
    # PARTE 8: Explorações e Visualizações Analíticas
    # ========================================
    print("\n== Parte 8: Visualizações Analíticas (3 Obrigatórias) ==")
    
    # 1. Mapa de Cores por Grau do Bairro (HTML Interativo - pyvis)
    from viz import visualize_degree_map
    viz_mapa_out = os.path.join(OUT_DIR, "mapa_grau.html")
    # df_graus já foi calculado e salvo em 'graus.csv'
    visualize_degree_map(g, df_graus, viz_mapa_out)

    # 2. Distribuição dos Graus (Histograma - Matplotlib)
    from viz import visualize_degree_histogram
    viz_hist_out = os.path.join(OUT_DIR, "histograma_graus.png")
    visualize_degree_histogram(df_graus, viz_hist_out)

    # 3. Subgrafo dos 10 Bairros com Maior Grau (HTML Interativo - pyvis)
    from viz import visualize_top_10_degree_subgraph
    viz_top10_out = os.path.join(OUT_DIR, "subgrafo_top10.html")
    visualize_top_10_degree_subgraph(g, df_graus, viz_top10_out)

    print("== Execução finalizada ==")
