import csv
import json
import os
from typing import Dict, Set, Tuple, List, Any
import time 
import random
import pandas as pd
from graphs.graph import Graph
from graphs.algorithms import dijkstra, bfs, dfs, bellman_ford 
# Certifique-se de que estas funções estão definidas em src/viz.py
from viz import (
    construir_arestas_arvore_percurso, visualize_interactive_graph, visualize_path_tree, visualize_degree_map, 
    visualize_degree_histogram, visualize_top_10_degree_subgraph, visualize_parte2_degree_histogram,
    compute_parte2_degrees
)


BAIRROS_CSV = "data/bairros_unique.csv"
ADJ_CSV = "data/adjacencias_bairros.csv"
OUT_DIR = "out"
OUT_GLOBAL = os.path.join(OUT_DIR, "recife_global.json")
OUT_MICRO = os.path.join(OUT_DIR, "microrregioes.json")
OUT_EGO = os.path.join(OUT_DIR, "ego_bairro.csv")
# AJUSTE O NOME DO ARQUIVO AQUI:
AIRCRAFT_DATA_PATH = "data/dataset_parte2.csv"

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
            # Chama g.add_edge com directed=False por padrão (Parte 1)
            g.add_edge(a, b, weight=weight, meta=meta) 
    return g, names

# --- Funções de Métrica da Parte 1 (Restauradas) ---

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
        for u, v, w, meta in g.get_edges():
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
    pares = []
    with open(path_enderecos, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader: pares.append(row)
    saida_csv = "out/distancias_enderecos.csv"
    with open(saida_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "bairro_X", "bairro_Y", "custo", "caminho"])
        for p in pares:
            origem = p["bairro_X"].strip(); destino = p["bairro_Y"].strip()
            if origem not in graph.adj or destino not in graph.adj: continue
            dist, prev = dijkstra(graph, origem)
            custo = dist.get(destino, float("inf")); caminho = []
            if custo < float("inf"):
                atual = destino
                while atual is not None: caminho.append(atual); atual = prev.get(atual)
                caminho.reverse()
            w.writerow([p["X"], p["Y"], origem, destino, custo, " -> ".join(caminho) if caminho else ""])
            if origem == "Nova Descoberta" and destino == "Boa Viagem":
                with open("out/percurso_nova_descoberta_setubal.json", "w", encoding="utf-8") as jf:
                    json.dump({"origem": origem, "destino": destino, "custo": custo, "caminho": caminho}, jf, ensure_ascii=False, indent=4)
    print(f"[OK] Distâncias calculadas → {saida_csv}")

# =========================================================================
# FUNÇÕES DE CARREGAMENTO E COMPARAÇÃO DA PARTE 2
# =========================================================================

def load_parte2_graph(data_path: str) -> Graph:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset da Parte 2 não encontrado em {data_path}")
    g = Graph()
    df = pd.read_csv(data_path, encoding='utf-8')
    ORIGIN_COL = 'usg_apt'; 
    DEST_COL = 'fg_apt'; 
    VOLUME_COL = 'Total'
    required_cols = [ORIGIN_COL, DEST_COL, VOLUME_COL]

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"O CSV da Parte 2 deve conter as colunas: {required_cols}. Verifique o nome real da coluna de volume.")
    df_rotas = df.groupby([ORIGIN_COL, DEST_COL])[VOLUME_COL].sum().reset_index()
    df_rotas.dropna(subset=[VOLUME_COL], inplace=True)
    df_rotas = df_rotas[df_rotas[VOLUME_COL] > 0]
    for _, row in df_rotas.iterrows():
        origem = str(row[ORIGIN_COL]).strip(); destino = str(row[DEST_COL]).strip()
        volume = row[VOLUME_COL]
        weight = 1.0 / volume
        # ADICIONANDO directed=True para a Parte 2
        g.add_edge(origem, destino, weight=weight, directed=True) 
    print(f"[P2 LOAD] Grafo Aéreo montado: {len(g)} nós, {len(g.get_edges())} arestas.")
    return g

def run_parte2_comparison(g: Graph):
    """ Executa os 4 algoritmos no grafo maior e coleta métricas de desempenho. """
    print("\n" + "="*50)
    print("== PARTE 2: COMPARAÇÃO DE ALGORITMOS ==")
    print("="*50)
    nodes = g.get_nodes()
    if len(nodes) < 10:
        print("[AVISO] Grafo da Parte 2 muito pequeno. Não é adequado para análise de desempenho.")
        return
    report: List[Dict] = []
    test_sources = random.sample(nodes, min(len(nodes), 3))
    test_pairs = []
    
    # Tentaremos gerar 5 pares. Limitamos as tentativas para evitar loops infinitos em grafos estranhos.
    max_desired_pairs = 5
    attempts = 0
    max_attempts = len(nodes) * 5
    
    while len(test_pairs) < max_desired_pairs and attempts < max_attempts:
        origem = random.choice(nodes)
        destino = random.choice(nodes)
        
        # 1. Garante que Origem != Destino
        # 2. Garante que o par ainda não foi testado
        if origem != destino and (origem, destino) not in test_pairs:
            test_pairs.append((origem, destino))
            
        attempts += 1
    # ----------------------------------------------------
    # I. Testes BFS e DFS 
    # ----------------------------------------------------
    print("\n--- 1. Testes BFS/DFS (a partir de 3 fontes) ---")
    for source in test_sources:
        # --- BFS ---
        start_time = time.perf_counter()
        dist, prev = bfs(g, source)
        elapsed = time.perf_counter() - start_time
        report.append({"alg": "BFS", "origem": source, "destino": "N/A", "tempo_s": elapsed})
        print(f"BFS ({source}): {len(dist)} nós alcançados em {elapsed:.6f}s")
        # --- DFS ---
        start_time = time.perf_counter()
        disc, fin, cycle = dfs(g, source) 
        elapsed = time.perf_counter() - start_time
        report.append({"alg": "DFS", "origem": source, "destino": "N/A", "tempo_s": elapsed, "ciclo_detectado": cycle})
        print(f"DFS ({source}): {len(disc)} nós descobertos em {elapsed:.6f}s (Ciclo: {cycle})")

    # ----------------------------------------------------
    # II. Testes Dijkstra e Bellman-Ford (Caminho Mínimo)
    # ----------------------------------------------------
    print("\n--- 2. Testes Dijkstra (5 pares, pesos >= 0) ---")
    for i, (origem, destino) in enumerate(test_pairs):
        start_time = time.perf_counter()
        try:
            dist, prev = dijkstra(g, origem)
            custo = dist.get(destino, float('inf'))
            elapsed = time.perf_counter() - start_time
            report.append({"alg": "Dijkstra", "origem": origem, "destino": destino, "custo": custo, "tempo_s": elapsed})
            print(f"Dijkstra ({i+1}): {origem} -> {destino} Custo={custo:.4f} em {elapsed:.6f}s")
        except ValueError as e:
             print(f"Dijkstra falhou para {origem} -> {destino}: {e}")
             report.append({"alg": "Dijkstra", "origem": origem, "destino": destino, "custo": "Erro", "tempo_s": elapsed})

    # ----------------------------------------------------
    # III. Testes Bellman-Ford (Casos Negativos Obrigatórios)
    # ----------------------------------------------------
    print("\n--- 3. Testes Bellman-Ford (Casos Negativos) ---")
    # Aqui é onde você deveria criar uma cópia do grafo e manipular arestas para os testes.
    
    try:
        A, B, C = test_sources
    except ValueError:
        print("[AVISO BF] Não há 3 fontes distintas para formar um ciclo de teste.")
        return 

    # --- Caso 1: Peso Negativo sem Ciclo Negativo ---
    
    # 1. Clonar o grafo base (pesos inversos > 0)
    g_neg_weight = g.copy()
    
    # 2. Injetar um peso negativo em uma aresta (sem criar um ciclo negativo)
    # Exemplo: A -> B agora tem um 'bônus' de tempo/custo. Usamos -0.01 como peso negativo.
    # O peso original é necessário para garantir que o ciclo não se torne negativo.
    # Vamos salvar os dados originais para restaurar depois, se necessário, mas para BF puro, basta adicionar.
    
    # Adicionar a aresta negativa A -> B (Se a aresta não existir, o BF ainda roda)
    # Usamos o método add_edge com directed=True, já que g_aereo é dirigido.
    g_neg_weight.add_edge(A, B, weight=-0.01, directed=True) 

    origem_bf_1 = A
    start_time = time.perf_counter()
    dist_1, prev_1, cycle_1 = bellman_ford(g_neg_weight, origem_bf_1)
    elapsed_1 = time.perf_counter() - start_time

    report.append({"alg": "Bellman-Ford", "caso": "Peso Negativo (Sem Ciclo)", "origem": origem_bf_1, 
                   "tempo_s": elapsed_1, "ciclo_detectado": cycle_1})
    print(f"BF (Peso Negativo): Ciclo detectado={cycle_1} em {elapsed_1:.6f}s (Custo de A->B: -0.01)")
    
    
    # --- Caso 2: Ciclo Negativo Detectado ---
    
    # 1. Clonar o grafo novamente
    g_neg_cycle = g.copy()
    
    # 2. Injetar pesos que garantam um ciclo negativo (A -> B -> C -> A)
    # O grafo tem pesos inversos (muito pequenos, ex: 0.0001). 
    # Para criar um ciclo negativo, a soma dos 3 trechos precisa ser menor que zero.
    
    # Passo A: Cria arestas A->B, B->C, C->A (ou garante que existam)
    # Estes valores são arbitrários e forçam um ciclo negativo,
    # garantindo que |V| relaxamentos farão o loop.
    CUSTO_AB = 1.0
    CUSTO_BC = 1.0
    CUSTO_CA = -3.0  # O valor mais pesado (negativo) para garantir 1.0 + 1.0 + (-3.0) = -1.0
    
    g_neg_cycle.add_edge(A, B, weight=CUSTO_AB, directed=True)
    g_neg_cycle.add_edge(B, C, weight=CUSTO_BC, directed=True)
    g_neg_cycle.add_edge(C, A, weight=CUSTO_CA, directed=True) # <- Aresta que fecha o ciclo negativo
    
    origem_bf_2 = A
    start_time = time.perf_counter()
    dist_2, prev_2, cycle_2 = bellman_ford(g_neg_cycle, origem_bf_2)
    elapsed_2 = time.perf_counter() - start_time

    report.append({"alg": "Bellman-Ford", "caso": "Ciclo Negativo Detectado", "origem": origem_bf_2, 
                   "tempo_s": elapsed_2, "ciclo_detectado": cycle_2})
    print(f"BF (Ciclo Negativo): Ciclo detectado={cycle_2} em {elapsed_2:.6f}s (Ciclo: A->B->C->A, Soma: -1.0)")

    # ----------------------------------------------------
    # IV. SALVAR RELATÓRIO (Parte 2, Ponto 3) - CORRIGIDO
    # ----------------------------------------------------
    
    final_report = []
    dataset_name = "Grafo Aéreo (Rotas e Volume de Tráfego)"
    
    for entry in report:
        # --- 1. ARREDONDAMENTO DE TEMPO ---
        tempo_s = entry.get("tempo_s")
        if isinstance(tempo_s, (int, float)):
            # Arredonda para 6 casas decimais, suficiente para a análise
            tempo_s = round(tempo_s, 6) 
        
        new_entry = {
            "algoritmo": entry.get("alg"),
            "dataset": dataset_name,
            "origem": entry.get("origem"),
            "destino": entry.get("destino", "N/A"),
            "tempo_execucao_segundos": tempo_s, # Usa o valor arredondado
        }
        
        # --- 2. TRATAMENTO DE CUSTO (Infinity e Arredondamento) ---
        custo = entry.get("custo")
        if custo is not None:
            if custo == "Erro":
                 new_entry["status"] = "ERRO: Dijkstra não suporta peso negativo ou caminho não encontrado."
            elif custo == float('inf'): 
                 # Converte o valor float('inf') para uma string JSON válida
                 new_entry["custo_caminho"] = "NÃO ALCANÇÁVEL"
            elif isinstance(custo, (int, float)):
                 # Arredonda custos válidos, se existirem
                 new_entry["custo_caminho"] = round(custo, 6)
            else:
                 new_entry["custo_caminho"] = custo
        
        if entry.get("ciclo_detectado") is not None:
            new_entry["detectou_ciclo_negativo"] = entry["ciclo_detectado"]
            
        if entry.get("caso"):
            new_entry["caso_teste"] = entry["caso"]
            
        final_report.append(new_entry)

    # 3. Salvar o arquivo out/parte2_report.json
    output_path = os.path.join(OUT_DIR, "parte2_report.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Não é necessário allow_nan=False aqui, pois tratamos float('inf') antes.
            json.dump(final_report, f, indent=4, ensure_ascii=False)
            
        print(f"\n[OK] Relatório de desempenho salvo em {output_path}")

    except Exception as e:
        print(f"[ERRO DE IO] Falha ao salvar {output_path}: {e}")

    

# =========================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# =========================================================================

if __name__ == "__main__":
    
    # --- PARTE 1: EXECUÇÃO COMPLETA ---
    print("\n" + "="*60)
    print("== PARTE 1: GRAPHO DOS BAIRROS DO RECIFE ==")
    print("="*60)

    # 1. Carregamento e Métricas
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
    
    # 2. Distâncias e Percurso (Parte 6 e 7)
    calcular_distancias_enderecos(g, path_enderecos="data/enderecos.csv")
    calcular_distancias_enderecos(g, path_enderecos="data/enderecos.csv") 

    # --- Chamada da Parte 7 (Árvore de Percurso) ---
    try:
        path_data_file = os.path.join(OUT_DIR, "percurso_nova_descoberta_setubal.json")
        with open(path_data_file, "r", encoding="utf-8") as f:
            path_data = json.load(f)
        caminho_obrig = path_data.get("caminho", [])
        
        origem = caminho_obrig[0] if caminho_obrig else "Nova Descoberta"
        destino = caminho_obrig[-1] if caminho_obrig else "Boa Viagem"

        if caminho_obrig and len(caminho_obrig) > 1:
            print(f"\n== Parte 7: Visualizando percurso de {origem} a {destino} ==")
            edges_path = construir_arestas_arvore_percurso(g, caminho_obrig)
            output_html = os.path.join(OUT_DIR, "arvore_percurso.html")
            visualize_path_tree(caminho_obrig, edges_path, output_html)
        
    except Exception as e:
        print(f"[AVISO/ERRO P7] Falha na visualização da Parte 7: {e}")
        # Definir caminho_obrigatório para a Parte 9
        caminho_obrig = [] 


    # 3. Visualizações Analíticas (Parte 8 e 9)
    # Requer que você tenha a função load_combined_metrics definida, mas para simplificar, 
    # vamos chamar as visualizações que você já tinha:
    
    print("\n== Parte 8: Visualizações Analíticas (3 Obrigatórias) ==")
    # 1. Mapa de Cores por Grau
    viz_mapa_out = os.path.join(OUT_DIR, "mapa_grau.html")
    visualize_degree_map(g, df_graus, viz_mapa_out)

    # 2. Distribuição dos Graus
    viz_hist_out = os.path.join(OUT_DIR, "histograma_graus.png")
    visualize_degree_histogram(df_graus, viz_hist_out)

    # 3. Subgrafo dos 10 Bairros
    viz_top10_out = os.path.join(OUT_DIR, "subgrafo_top10.html")
    visualize_top_10_degree_subgraph(g, df_graus, viz_top10_out)

    # Parte 9: Grafo Interativo Completo
    # Requer que você tenha a função load_combined_metrics definida.
    # Exemplo: visualize_interactive_graph(g, node_metrics, caminho_obrig, os.path.join(OUT_DIR, "grafo_interativo.html"))
    print("\n== Parte 9: Grafo Interativo Completo ==")
    out_grafo_interativo = os.path.join(OUT_DIR, "grafo_interativo.html")

    visualize_interactive_graph(g, df_ego, df_graus, caminho_obrig, out_grafo_interativo)

    # --- PARTE 2: EXECUÇÃO E COMPARAÇÃO ---
    print("\n" + "="*60)
    print("== PARTE 2: COMPARAÇÃO DE ALGORITMOS (Aéreo) ==")
    print("="*60)
    
    try:
        # 1. Carregar o grafo aéreo (g_aereo é um grafo dirigido)
        g_aereo = load_parte2_graph(AIRCRAFT_DATA_PATH)
        
        # 2. Rodar os testes e a comparação de desempenho
        run_parte2_comparison(g_aereo)
        
    except FileNotFoundError as e:
        print(f"[ERRO CRÍTICO] Falha ao carregar o grafo da Parte 2: {e}")
        print("Certifique-se de que o nome do arquivo e o caminho estão corretos.")
    except ValueError as e:
         print(f"[ERRO DE DADOS P2] Falha no processamento do CSV da Parte 2: {e}")
    except Exception as e:
         print(f"[ERRO GERAL P2] Ocorreu um erro geral na Parte 2: {e}")

    print("\n--- 4. Visualização da Estrutura (Histograma) ---")
    
    # ----------------------------------------------------
    # IV. VISUALIZAÇÕES ADICIONAIS (Parte 2, Requisito 4)
    # ----------------------------------------------------
    # 1. Calcular Graus de Saída
    df_graus_p2 = compute_parte2_degrees(g)
    
    # 2. Gerar Histograma
    viz_hist_p2_out = os.path.join(OUT_DIR, "p2_histograma_graus_saida.png")
    visualize_parte2_degree_histogram(df_graus_p2, viz_hist_p2_out)

    print("\n== EXECUÇÃO GLOBAL FINALIZADA ==")