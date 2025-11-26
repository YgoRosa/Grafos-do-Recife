import csv
import json
import os
import time 
import random
import pandas as pd
from typing import Dict, Set, Tuple, List, Any

# --- IMPORTS CORRIGIDOS ---
from graphs.graph import Graph
from graphs.algorithms import Algorithms # Usamos a classe agora
# Certifique-se de que viz.py tem a função visualize_interactive_complete
from viz import (
    construir_arestas_arvore_percurso, 
    visualize_interactive_graph, # Nome atualizado da função interativa
    visualize_path_tree, 
    visualize_degree_map, 
    visualize_degree_histogram, 
    visualize_top_10_degree_subgraph, 
    visualize_parte2_degree_histogram,
    compute_parte2_degrees
)

# Configurações de Arquivos
BAIRROS_CSV = "data/bairros_unique.csv"
ADJ_CSV = "data/adjacencias_bairros.csv"
OUT_DIR = "out"
OUT_GLOBAL = os.path.join(OUT_DIR, "recife_global.json")
OUT_MICRO = os.path.join(OUT_DIR, "microrregioes.json")
OUT_EGO = os.path.join(OUT_DIR, "ego_bairro.csv")
AIRCRAFT_DATA_PATH = "data/dataset_parte2.csv"
CAMINHO_JSON = "out/percurso_nova_descoberta_setubal.json"

# =========================================================================
# FUNÇÕES DE CARREGAMENTO E MÉTRICAS DA PARTE 1
# =========================================================================

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
    
    # Parte 1: Grafo Não-Direcionado
    g = Graph(directed=False)
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
            
            # Não passamos directed aqui, pois definimos no construtor
            g.add_edge(a, b, weight=weight, meta=meta) 
            
    return g, names

# --- Funções de Métrica da Parte 1 ---

def graph_order(g: Graph) -> int: return len(g)
def graph_size(g: Graph) -> int: return len(g.get_edges())
def density(num_nodes: int, num_edges: int) -> float: 
    if num_nodes < 2: return 0.0
    return (2.0 * num_edges) / (num_nodes * (num_nodes - 1))

def compute_and_save_global(g: Graph, out_path: str):
    V = graph_order(g); E = graph_size(g); dens = density(V, E)
    result = {"ordem": V, "tamanho": E, "densidade": dens}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
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
            if u in bairros and v in bairros: sub_edge_count += 1
        ordem = len(bairros)
        tamanho = sub_edge_count
        dens = density(ordem, tamanho) 
        results.append({"microrregiao": mr, "ordem": ordem, "tamanho": tamanho, "densidade": dens})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)
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
        
        # Ajuste para contar arestas corretamente no subgrafo ego
        edges = g.get_edges()
        for u, v, w, meta in edges:
            if u in ego_nodes and v in ego_nodes: ego_edge_count += 1
            
        ordem_ego = len(ego_nodes)
        tamanho_ego = ego_edge_count
        dens_ego = density(ordem_ego, tamanho_ego)
        rows.append({"bairro": bairro, "grau": grau, "ordem_ego": ordem_ego, "tamanho_ego": tamanho_ego, "densidade_ego": dens_ego})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows, columns=["bairro", "grau", "ordem_ego", "tamanho_ego", "densidade_ego"])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[EGO] {len(rows)} bairros processados -> {out_path}")
    return df

def compute_and_save_graus(g: Graph, bairros_all: Set[str], out_path: str):
    rows = []
    for bairro in sorted(bairros_all):
        grau = len(g.neighbors(bairro)) if g.has_node(bairro) else 0
        rows.append({"bairro": bairro, "grau": grau})
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
    print(f"• Bairro mais denso ..........: {bairro_mais_denso} \t(densidade={dens_max:.4f})")
    print(f"• Bairro com maior grau ......: {bairro_maior_grau} \t(grau={grau_max})")
    print("=====================================\n")

def calcular_distancias_enderecos(graph: Graph, path_enderecos="data/enderecos.csv"):
    print("== Parte 6.2: cálculo de distâncias entre endereços ==")
    
    if not os.path.exists(path_enderecos):
        print(f"[AVISO] {path_enderecos} não encontrado.")
        return

    pares = []
    with open(path_enderecos, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader: pares.append(row)
    
    saida_csv = "out/distancias_enderecos.csv"
    NOME_SETUBAL_PADRONIZADO = "Setúbal" # Ajuste se no seu grafo for "Boa Viagem (Setúbal)"
    
    with open(saida_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "bairro_X", "bairro_Y", "custo", "caminho"])
        
        for p in pares:
            origem = p.get("bairro_X", "").strip() or p.get("bairro_origem", "").strip()
            destino_original = p.get("bairro_Y", "").strip() or p.get("bairro_destino", "").strip()
            
            # Normalização para Setúbal
            destino_final = destino_original
            if "boa viagem" in destino_original.lower() and graph.has_node(NOME_SETUBAL_PADRONIZADO):
                destino_final = NOME_SETUBAL_PADRONIZADO
            
            if origem not in graph.adj or destino_final not in graph.adj: 
                print(f"[SKIP] {origem} ou {destino_final} não encontrados.")
                continue
            
            # CHAMADA ATUALIZADA: Algorithms.dijkstra
            res = Algorithms.dijkstra(graph, origem, destino_final)
            custo = res["cost"]
            caminho = res["path"]
            
            caminho_str = " -> ".join(caminho) if caminho else ""
            w.writerow([p.get("X",""), p.get("Y",""), origem, destino_original, custo, caminho_str])
            
            # Salvar JSON Obrigatório
            if "nova descoberta" in origem.lower() and ("boa viagem" in destino_original.lower() or "setúbal" in destino_original.lower()):
                with open("out/percurso_nova_descoberta_setubal.json", "w", encoding="utf-8") as jf:
                    json.dump({"caminho": caminho, "custo": custo, "origem": origem, "destino": destino_final}, jf, ensure_ascii=False, indent=4)
                print("   -> JSON do percurso obrigatório salvo.")
                
    print(f"[OK] Distâncias calculadas → {saida_csv}")

# =========================================================================
# FUNÇÕES DE CARREGAMENTO E COMPARAÇÃO DA PARTE 2
# =========================================================================

def load_parte2_graph(data_path: str) -> Graph:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset da Parte 2 não encontrado em {data_path}")
    
    # IMPORTANTE: Definimos o grafo como DIRIGIDO aqui no construtor
    g = Graph(directed=True)
    
    df = pd.read_csv(data_path, encoding='utf-8')
    ORIGIN_COL = 'usg_apt'; DEST_COL = 'fg_apt'; VOLUME_COL = 'Total'
    required_cols = [ORIGIN_COL, DEST_COL, VOLUME_COL]

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"O CSV da Parte 2 deve conter as colunas: {required_cols}")
    
    df_rotas = df.groupby([ORIGIN_COL, DEST_COL])[VOLUME_COL].sum().reset_index()
    df_rotas.dropna(subset=[VOLUME_COL], inplace=True)
    df_rotas = df_rotas[df_rotas[VOLUME_COL] > 0]
    
    for _, row in df_rotas.iterrows():
        origem = str(row[ORIGIN_COL]).strip(); destino = str(row[DEST_COL]).strip()
        volume = row[VOLUME_COL]
        weight = 1.0 / volume
        
        # Não passamos directed=True aqui, pois já foi definido no Graph()
        g.add_edge(origem, destino, weight=weight) 
        
    print(f"[P2 LOAD] Grafo Aéreo montado: {len(g)} nós, {len(g.get_edges())} arestas.")
    return g

def run_parte2_comparison(g: Graph):
    """ Executa os 4 algoritmos no grafo maior e coleta métricas de desempenho. """
    print("\n" + "="*50)
    print("== PARTE 2: COMPARAÇÃO DE ALGORITMOS ==")
    print("="*50)
    
    nodes = g.get_nodes()
    if len(nodes) < 5:
        print("[AVISO] Grafo da Parte 2 muito pequeno.")
        return
        
    report: List[Dict] = []
    
    # ----------------------------------------------------
    # Estratégia Inteligente de Pares (Evitar custos infinitos)
    # ----------------------------------------------------
    print("   -> Gerando pares conectados para teste...")
    test_pairs = []
    test_sources = [] # Vamos guardar fontes válidas para usar no BFS/DFS também
    attempts = 0
    
    # Tenta achar pares (origem, destino) onde realmente existe caminho
    # Limitamos attempts para não travar se o grafo for muito desconexo
    while len(test_pairs) < 5 and attempts < 50:
        src = random.choice(nodes)
        
        # Faz um BFS rápido para ver quem esse nó alcança
        bfs_data = Algorithms.bfs(g, src)
        reachable = bfs_data.get("order", [])
        
        # Se alcança alguém (além de si mesmo)
        if len(reachable) > 2:
            dst = random.choice(reachable)
            if src != dst and (src, dst) not in test_pairs:
                test_pairs.append((src, dst))
                # Guarda essa origem como uma boa candidata para os testes de BFS/DFS
                if src not in test_sources:
                    test_sources.append(src)
        attempts += 1
    
    # Se faltar fontes para o BFS/DFS, completa com aleatórios
    while len(test_sources) < 3:
        test_sources.append(random.choice(nodes))

    # ----------------------------------------------------
    # I. Testes BFS e DFS 
    # ----------------------------------------------------
    print("\n--- 1. Testes BFS/DFS ---")
    for source in test_sources[:3]: # Usa as fontes que sabemos que alcançam algo
        # --- BFS ---
        start_time = time.perf_counter()
        res_bfs = Algorithms.bfs(g, source)
        elapsed = time.perf_counter() - start_time
        
        count = len(res_bfs.get("order", []))
        report.append({"alg": "BFS", "origem": source, "destino": "N/A", "tempo_s": elapsed})
        print(f"BFS ({source}): {count} nós alcançados em {elapsed:.6f}s")
        
        # --- DFS ---
        start_time = time.perf_counter()
        res_dfs = Algorithms.dfs(g, source) 
        elapsed = time.perf_counter() - start_time
        
        count = len(res_dfs.get("order", []))
        cycle = res_dfs.get("has_cycle", False)
        report.append({"alg": "DFS", "origem": source, "destino": "N/A", "tempo_s": elapsed, "ciclo_detectado": cycle})
        print(f"DFS ({source}): {count} nós descobertos em {elapsed:.6f}s (Ciclo: {cycle})")

    # ----------------------------------------------------
    # II. Testes Dijkstra (Pesos Positivos)
    # ----------------------------------------------------
    print("\n--- 2. Testes Dijkstra (5 pares) ---")
    for i, (origem, destino) in enumerate(test_pairs):
        start_time = time.perf_counter()
        res_dijk = Algorithms.dijkstra(g, origem, destino)
        elapsed = time.perf_counter() - start_time
        
        custo = res_dijk.get("cost")
        # Tratamento de erro ou infinito
        if "error" in res_dijk:
            custo = "Erro"
            
        report.append({"alg": "Dijkstra", "origem": origem, "destino": destino, "custo": custo, "tempo_s": elapsed})
        
        # Formatação bonita para o print
        val_custo = f"{custo:.4f}" if isinstance(custo, (int, float)) else str(custo)
        print(f"Dijkstra ({i+1}): {origem} -> {destino} Custo={val_custo} em {elapsed:.6f}s")
    # ----------------------------------------------------
    # III. Testes Bellman-Ford (Casos Negativos Obrigatórios)
    # ----------------------------------------------------
    print("\n--- 3. Testes Bellman-Ford (Cenários de Teste) ---")
    
    if len(test_sources) >= 1:
        # Pega 3 nós quaisquer para formar ciclo artificial
        if len(nodes) >= 3:
            A, B, C = random.sample(nodes, 3)
        else:
            A = B = C = test_sources[0]
            
        # --- Caso 1: Peso Negativo sem Ciclo Negativo ---
        g_neg_weight = g.copy()
        g_neg_weight.add_edge(A, B, weight=-0.01) # Bônus pequeno

        start_time = time.perf_counter()
        res_bf1 = Algorithms.bellman_ford(g_neg_weight, A, B)
        elapsed_1 = time.perf_counter() - start_time

        cycle_1 = res_bf1.get("negative_cycle")
        report.append({"alg": "Bellman-Ford", "caso": "Peso Negativo (Sem Ciclo)", "origem": A, 
                    "tempo_s": elapsed_1, "ciclo_detectado": cycle_1})
        print(f"BF (Peso Negativo): Ciclo detectado={cycle_1} em {elapsed_1:.6f}s")
        
        # --- Caso 2: Ciclo Negativo Detectado ---
        g_neg_cycle = g.copy()
        # Força ciclo A->B->C->A com soma negativa
        g_neg_cycle.add_edge(A, B, weight=1.0)
        g_neg_cycle.add_edge(B, C, weight=1.0)
        g_neg_cycle.add_edge(C, A, weight=-5.0) 
        
        start_time = time.perf_counter()
        res_bf2 = Algorithms.bellman_ford(g_neg_cycle, A, C)
        elapsed_2 = time.perf_counter() - start_time

        cycle_2 = res_bf2.get("negative_cycle")
        report.append({"alg": "Bellman-Ford", "caso": "Ciclo Negativo Detectado", "origem": A, 
                    "tempo_s": elapsed_2, "ciclo_detectado": cycle_2})
        print(f"BF (Ciclo Negativo): Ciclo detectado={cycle_2} em {elapsed_2:.6f}s")

    # ----------------------------------------------------
    # IV. SALVAR RELATÓRIO
    # ----------------------------------------------------
    final_report = []
    dataset_name = "Grafo Aéreo"
    
    for entry in report:
        tempo_s = entry.get("tempo_s", 0)
        new_entry = {
            "algoritmo": entry.get("alg"),
            "dataset": dataset_name,
            "origem": entry.get("origem"),
            "destino": entry.get("destino", "N/A"),
            "tempo_execucao_segundos": round(tempo_s, 6),
        }
        
        custo = entry.get("custo")
        if custo is not None:
            if custo == float('inf'): new_entry["custo_caminho"] = "NÃO ALCANÇÁVEL"
            elif isinstance(custo, (int, float)): new_entry["custo_caminho"] = round(custo, 6)
            else: new_entry["custo_caminho"] = str(custo)
        
        if "ciclo_detectado" in entry:
            new_entry["detectou_ciclo_negativo"] = entry["ciclo_detectado"]
        if "caso" in entry:
            new_entry["caso_teste"] = entry["caso"]
            
        final_report.append(new_entry)

    output_path = os.path.join(OUT_DIR, "parte2_report.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)
        print(f"\n[OK] Relatório salvo em {output_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar relatório: {e}")

# =========================================================================
# BLOCO PRINCIPAL
# =========================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("== PARTE 1: GRAPHO DOS BAIRROS DO RECIFE ==")
    print("="*60)

    try:
        # 1. Carregamento e Métricas
        bairros_map = read_bairros_map(BAIRROS_CSV)
        print(f"Carregados {len(bairros_map)} bairros.")
        g, names_in_adj = load_adjacencias_to_graph(ADJ_CSV)
        print(f"Grafo montado: {len(g)} nós, {len(g.get_edges())} arestas")
        
        all_bairros = set(bairros_map.keys()).union(names_in_adj)
        
        compute_and_save_global(g, OUT_GLOBAL)
        compute_and_save_microrregioes(g, bairros_map, OUT_MICRO)
        df_ego = compute_and_save_ego(g, all_bairros, OUT_EGO)
        
        graus_csv_path = os.path.join(OUT_DIR, "graus.csv")
        df_graus = compute_and_save_graus(g, all_bairros, graus_csv_path)
        
        find_topological_highlights(df_ego, df_graus)
        
        # 2. Distâncias
        calcular_distancias_enderecos(g, path_enderecos="data/enderecos.csv")

        # 3. Visualizações Parte 1
        caminho_obrig = []
        if os.path.exists(CAMINHO_JSON):
            with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
                path_data = json.load(f)
            caminho_obrig = path_data.get("caminho", [])
            
            if caminho_obrig:
                edges_path = construir_arestas_arvore_percurso(g, caminho_obrig)
                visualize_path_tree(caminho_obrig, edges_path, os.path.join(OUT_DIR, "arvore_percurso.html"))

        print("\n== Parte 8/9: Visualizações Extras ==")
        visualize_degree_map(g, df_graus, os.path.join(OUT_DIR, "mapa_grau.html"))
        visualize_degree_histogram(df_graus, os.path.join(OUT_DIR, "histograma_graus.png"))
        visualize_top_10_degree_subgraph(g, df_graus, os.path.join(OUT_DIR, "subgrafo_top10.html"))
        
        # Grafo Interativo Completo
        map_micro = {b: m for b, m in bairros_map.items()} 
        visualize_interactive_graph(g, df_ego, map_micro, caminho_obrig, os.path.join(OUT_DIR, "grafo_interativo.html"))

    except Exception as e:
        print(f"[ERRO PARTE 1] {e}")
        import traceback
        traceback.print_exc()

    # --- PARTE 2 ---
    print("\n" + "="*60)
    print("== PARTE 2: COMPARAÇÃO DE ALGORITMOS (Aéreo) ==")
    print("="*60)
    
    try:
        g_aereo = load_parte2_graph(AIRCRAFT_DATA_PATH)
        run_parte2_comparison(g_aereo)
        
        # Viz extra da Parte 2
        df_graus_p2 = compute_parte2_degrees(g_aereo)
        visualize_parte2_degree_histogram(df_graus_p2, os.path.join(OUT_DIR, "p2_histograma_graus_saida.png"))
        
    except Exception as e:
         print(f"[ERRO PARTE 2] {e}")
         import traceback
         traceback.print_exc()

    print("\n== EXECUÇÃO GLOBAL FINALIZADA ==")