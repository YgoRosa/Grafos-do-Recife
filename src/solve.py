import csv
import json
import os
from typing import Dict, Set, Tuple, List

import pandas as pd
from src.graphs.graph import Graph
from src.graphs.algorithms import dijkstra
from src.graphs.algorithms import dijkstra_path



# --- caminhos (ajuste se necessário) ---
BAIRROS_CSV = "data/bairros_unique.csv"
ADJ_CSV = "data/adjacencias_bairros.csv"
OUT_DIR = "out"
OUT_GLOBAL = os.path.join(OUT_DIR, "recife_global.json")
OUT_MICRO = os.path.join(OUT_DIR, "microrregioes.json")
OUT_EGO = os.path.join(OUT_DIR, "ego_bairro.csv")


def read_bairros_map(path: str) -> Dict[str, str]:
    """Lê bairros_unique.csv -> dict[bairro] = microrregiao (strings)."""
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
    """
    Lê adjacencias_bairros.csv com colunas: origem,destino,logradouro,peso
    Retorna (graph, set_nomes_encontrados_nas_arestas)
    """
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"{adj_path} não encontrado.")

    g = Graph()
    names = set()

    with open(adj_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # normalize header names: procura colunas por variante
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
            # le logradouro e peso se existirem
            meta = {}
            if log_col and row.get(log_col) and not pd.isna(row.get(log_col)):
                meta["logradouro"] = str(row.get(log_col)).strip()
            try:
                weight = float(row[peso_col]) if peso_col and row.get(peso_col) not in (None, "") else 1.0
            except Exception:
                weight = 1.0

            # adicionar aresta (Graph já espelha por ser não-direcionado)
            g.add_edge(a, b, weight=weight, meta=meta)

    return g, names


# -------------------------
# Métricas utilitárias
# -------------------------
def graph_order(g: Graph) -> int:
    # usamos __len__ implementado no Graph
    return len(g)


def graph_size(g: Graph) -> int:
    # contamos arestas via get_edges() (que já evita duplicatas)
    return len(g.get_edges())


def density(num_nodes: int, num_edges: int) -> float:
    if num_nodes < 2:
        return 0.0
    return (2.0 * num_edges) / (num_nodes * (num_nodes - 1))


# -------------------------
# 3.1 Métricas globais
# -------------------------
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


# -------------------------
# 3.2 Métricas por microrregião (subgrafo induzido)
# -------------------------
def compute_and_save_microrregioes(g: Graph, bairros_map: Dict[str, str], out_path: str):
    # agrupa bairros por microrregiao
    mr_to_bairros: Dict[str, Set[str]] = {}
    for bairro, mr in bairros_map.items():
        mr_to_bairros.setdefault(mr, set()).add(bairro)

    results = []
    for mr, bairros in sorted(mr_to_bairros.items()):
        # formar subgrafo induzido: apenas arestas com ambos endpoints em `bairros`
        sub_edge_count = 0
        # usa get_edges() do grafo para evitar duplicatas
        for u, v, w, meta in g.get_edges():
            if u in bairros and v in bairros:
                sub_edge_count += 1
        # ordem: número de bairros da microrregião que aparecem na lista de bairros_unique
        ordem = len(bairros)
        tamanho = sub_edge_count
        dens = density(ordem, tamanho)
        results.append({"microrregiao": mr, "ordem": ordem, "tamanho": tamanho, "densidade": dens})

    # salvar
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[MICRORREGIÕES] geradas {len(results)} entradas -> {out_path}")
    return results


# -------------------------
# 3.3 Ego-subrede por bairro
# -------------------------
def compute_and_save_ego(g: Graph, bairros_all: Set[str], out_path: str):
    """
    Para cada bairro v (em bairros_all), calcula: grau, ordem_ego, tamanho_ego, densidade_ego.
    Salva CSV com colunas: bairro,grau,ordem_ego,tamanho_ego,densidade_ego
    """
    rows: List[Dict] = []
    # Percorrer todos os bairros esperados (inclui isolados)
    for bairro in sorted(bairros_all):
        # grau: número de vizinhos no grafo (se o bairro não existir em g, grau=0)
        if g.has_node(bairro):
            viz = set(g.neighbors(bairro))
        else:
            viz = set()
        grau = len(viz)
        ego_nodes = set(viz)
        ego_nodes.add(bairro)
        # contar arestas internas do ego (usar get_edges para evitar duplic)
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

    # salvar CSV
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["bairro", "grau", "ordem_ego", "tamanho_ego", "densidade_ego"])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[EGO] {len(rows)} bairros processados -> {out_path}")
    return df

def compute_and_save_graus(g: Graph, bairros_all: Set[str], out_path: str):
    """
    Salva out/graus.csv com: bairro,grau
    """
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
    """
    Identifica:
      - bairro mais denso (maior densidade_ego)
      - bairro com maior grau
    Apenas imprime no console.
    """
    # Bairro mais denso
    row_denso = df_ego.loc[df_ego["densidade_ego"].idxmax()]
    bairro_mais_denso = row_denso["bairro"]
    dens_max = row_denso["densidade_ego"]

    # Bairro com maior grau
    row_grau = df_graus.loc[df_graus["grau"].idxmax()]
    bairro_maior_grau = row_grau["bairro"]
    grau_max = row_grau["grau"]

    print("\n== Rankings Topológicos (Parte 4) ==")
    print(f"• Bairro mais denso ..........: {bairro_mais_denso}  (densidade={dens_max:.4f})")
    print(f"• Bairro com maior grau ......: {bairro_maior_grau}  (grau={grau_max})")
    print("=====================================\n")

# -------------------------
# Orquestração principal
# -------------------------
def run_metrics():
    print("== Iniciando cálculo de métricas (Parte 3) ==")

    # 1) ler bairros (mapa bairro -> microrregiao)
    bairros_map = read_bairros_map(BAIRROS_CSV)
    print(f"Carregados {len(bairros_map)} bairros de {BAIRROS_CSV}")

    # 2) ler adjacências e montar grafo
    g, names_in_adj = load_adjacencias_to_graph(ADJ_CSV)
    print(f"Grafo montado: {len(g)} nós, {len(g.get_edges())} arestas (contagem via get_edges)")

    # 3) checar nomes em adj que não existem no bairros_unique
    names_not_in_bairros = sorted(list(names_in_adj - set(bairros_map.keys())))
    if names_not_in_bairros:
        print("ATENÇÃO: os seguintes nomes aparecem em adjacencias_bairros.csv mas NÃO em bairros_unique.csv:")
        for n in names_not_in_bairros[:40]:
            print("  -", n)
        print(f"  ... total {len(names_not_in_bairros)} nomes inconsistentes.")
    else:
        print("Nomes em adjacencias OK em relação a bairros_unique.csv")

    # incluir todos os bairros listados (mesmo isolados)
    all_bairros = set(bairros_map.keys()).union(names_in_adj)

    # 4) calcular e salvar global
    compute_and_save_global(g, OUT_GLOBAL)

    # 5) calcular e salvar microrregioes
    compute_and_save_microrregioes(g, bairros_map, OUT_MICRO)

    # 6) calcular e salvar ego por bairro
    compute_and_save_ego(g, all_bairros, OUT_EGO)
    
    df_ego = compute_and_save_ego(g, all_bairros, OUT_EGO)   # 7) graus por bairro (Parte 4.1)
    graus_csv_path = os.path.join(OUT_DIR, "graus.csv")
    df_graus = compute_and_save_graus(g, all_bairros, graus_csv_path)

    # 8) rankings (Parte 4.2)
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

            # Caso especial obrigatório
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

    # 1) ler mapa de bairros
    bairros_map = read_bairros_map(BAIRROS_CSV)
    print(f"Carregados {len(bairros_map)} bairros de {BAIRROS_CSV}")

    # 2) montar grafo a partir do CSV de adjacências
    g, names_in_adj = load_adjacencias_to_graph(ADJ_CSV)
    print(f"Grafo montado: {len(g)} nós, {len(g.get_edges())} arestas")

    # 3) conjunto completo de bairros (inclui isolados listados nas adjacências)
    all_bairros = set(bairros_map.keys()).union(names_in_adj)

    # 4) Parte 3: métricas
    compute_and_save_global(g, OUT_GLOBAL)
    compute_and_save_microrregioes(g, bairros_map, OUT_MICRO)

    # 5) Parte 3.3 (ego) — chama apenas 1 vez e guarda o dataframe
    df_ego = compute_and_save_ego(g, all_bairros, OUT_EGO)

    # 6) Parte 4: graus e rankings
    graus_csv_path = os.path.join(OUT_DIR, "graus.csv")
    df_graus = compute_and_save_graus(g, all_bairros, graus_csv_path)
    find_topological_highlights(df_ego, df_graus)

    # 7) Parte 6: distâncias entre endereços (usa data/enderecos.csv)
    calcular_distancias_enderecos(g, path_enderecos="data/enderecos.csv")

    print("== Execução finalizada ==")

