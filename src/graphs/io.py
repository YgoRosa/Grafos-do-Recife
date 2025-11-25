import pandas as pd
import unicodedata
import os
from .graph import Graph

# Em src/graphs/io.py
import pandas as pd
import unicodedata
import os
from .graph import Graph # Mantenha a importação relativa

# ... (Sua função normalizar_subbairros)

# Em src/graphs/io.py

# Em src/graphs/io.py

# Em src/graphs/io.py

# Em src/graphs/io.py

def clean_bairros(input_path: str, output_path: str):
    """
    Limpa e normaliza o CSV de bairros WIDE, forçando o uso da primeira linha
    (os IDs de Microrregião) como cabeçalho para realizar o MELT.
    """
    import pandas as pd
    import os
    
    print(f"\n--- INICIANDO LEITURA E MELT ROBUSTO DO CSV: {input_path} ---")
    
    # 1. Leitura: Lê o arquivo SEM cabeçalho, forçando o Pandas a ler todas as colunas
    try:
        df_raw = pd.read_csv(input_path, sep=',', encoding='utf-8-sig', header=None) 
    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível ler o CSV sem cabeçalho. Erro: {e}")
        return pd.DataFrame() 

    # 2. Atribui a primeira linha como os IDs de Microrregião
    # Remove colunas que são totalmente NaN (geralmente criadas por vírgulas extras)
    df_raw = df_raw.dropna(axis=1, how='all')

    # A primeira linha agora contém os IDs (1.1, 1.2, etc.)
    microrregiao_ids = df_raw.iloc[0].astype(str).str.strip().tolist()
    
    # Remove a primeira linha e atribui os IDs como cabeçalho
    df_data = df_raw[1:].reset_index(drop=True)
    df_data.columns = microrregiao_ids
    
    # 3. Reorganiza (Melt) o DataFrame do formato WIDE para LONG
    df_long = pd.melt(
        df_data, 
        id_vars=[], 
        value_vars=microrregiao_ids, 
        var_name='microrregiao', 
        value_name='bairro'
    )
    
    # 4. Limpeza e Normalização
    df_long = df_long.dropna(subset=["bairro"]).copy() 
    
    # Garante que as strings sejam limpas e remove duplicatas
    df_long['bairro'] = df_long['bairro'].astype(str).str.strip()
    df_long['microrregiao'] = df_long['microrregiao'].astype(str).str.strip()
    
    df_unique = df_long.drop_duplicates(subset=["bairro"])

    # 5. Saída e Debug
    
    print(f"DEBUG: {len(df_unique)} bairros únicos sobreviveram. (Esperado: 95)")
    print(f"DEBUG: Microrregiões únicas encontradas: {df_unique['microrregiao'].unique()}")
    print("--------------------------------------------------\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_unique[['bairro', 'microrregiao']].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ CSV de bairros limpo (LONG) e salvo em: {output_path}")
    print(f"Total de bairros únicos: {len(df_unique)}")
    return df_unique

# --- ADICIONE ESTA FUNÇÃO AO SEU IO.PY ---

def add_edges_to_graph(g: Graph, adjacencias_csv_path: str):
    """Lê o CSV de adjacências e adiciona as arestas e seus pesos ao grafo."""
    try:
        df_adj = pd.read_csv(adjacencias_csv_path)
    except Exception as e:
        print(f"ERRO FATAL (ARESTAS): Falha ao ler {adjacencias_csv_path}. Motivo: {e}")
        return # Simplesmente sai da função.
    
    df_adj = pd.read_csv(adjacencias_csv_path)

    # 1. Normalização dos Nomes (para bater com os nós do grafo)
    df_adj["origem"] = df_adj["origem"].apply(normalizar_subbairros)
    df_adj["destino"] = df_adj["destino"].apply(normalizar_subbairros)
    
    # Requisito 6/7: Tratar Setúbal como sub-bairro
    df_adj["origem"] = df_adj["origem"].apply(normalizar_subbairros)
    df_adj["destino"] = df_adj["destino"].apply(normalizar_subbairros)
    
    # 2. Adiciona as Arestas
    for index, row in df_adj.iterrows():
        origem = row["origem"]
        destino = row["destino"]
        peso = row["peso"] # Assume que a coluna 'peso' já está no seu adjacencias_bairros.csv [cite: 65, 101]
        logradouro = row.get("logradouro", "")
        
        # Garante que os nós existem no grafo antes de adicionar a aresta
        if g.has_node(origem) and g.has_node(destino):
            # Grafo não-direcionado, adicione a aresta.
            # Se for ponderado, use o peso. [cite: 69]
            g.add_edge(origem, destino, weight=peso, logradouro=logradouro)

    print(f"✅ {g.get_num_edges()} arestas carregadas no grafo.")


# --- MODIFIQUE/SUBSTITUA A FUNÇÃO carregar_bairros EXISTENTE ---

# Em src/graphs/io.py

# Em src/graphs/io.py

def load_nodes_with_metrics(bairros_unique_csv_path: str) -> Graph:
    g = Graph(is_directed=False)
    try:
        import pandas as pd
        df = pd.read_csv(bairros_unique_csv_path)
    except Exception as e:
        print(f"ERRO FATAL (NÓS): Falha ao ler {bairros_unique_csv_path}. Motivo: {e}")
        return g # Retorna grafo vazio para evitar o NoneType

    for index, row in df.iterrows():
        bairro = row["bairro"]
        microrregiao_raw = row["microrregiao"] 
        microrregiao_str = None
        
        # 1. Trata Nulos/NaN
        if not pd.isna(microrregiao_raw):
            
            # 2. Simplifica a conversão de float para string para IDs como 1.1, 6.3
            if isinstance(microrregiao_raw, (int, float)):
                # Converte para string e remove zeros/pontos desnecessários, padronizando.
                # Ex: 1.1 -> "1.1"; 2.0 -> "2"
                microrregiao_str = str(microrregiao_raw).strip().rstrip('0').rstrip('.')
            else:
                # Trata como string simples (se já era string)
                microrregiao_str = str(microrregiao_raw).strip()
            
            # 3. Filtra strings nulas remanescentes
            if microrregiao_str.lower() in ('nan', 'none', ''): 
                microrregiao_str = None
        
        # ⚠️ CORREÇÃO CRÍTICA FINAL: Garante que um valor NÃO-NULO seja armazenado
        final_region_id = microrregiao_str if microrregiao_str is not None else "SEM REGIAO"
            
        g.add_node(bairro, metrics={"microrregiao": final_region_id})

    print(f"✅ {g.get_num_nodes()} bairros carregados no grafo com métricas.")
    return g


# --- NOVA FUNÇÃO PRINCIPAL (Wrapper para o Streamlit) ---

def load_recife_graph(bairros_input_csv_path: str, adjacencias_csv_path: str) -> Graph:
    """
    Função principal que carrega os dados brutos e constrói o objeto Graph completo.
    É a função que o seu app.py está tentando chamar.
    """
    
    # Garante que o CSV único (intermediário) existe
    BAIRROS_UNIQUE_PATH = "data/bairros_unique.csv"
    if not os.path.exists(BAIRROS_UNIQUE_PATH):
        print(f"⚠️ Gerando {BAIRROS_UNIQUE_PATH} a partir do CSV de entrada...")
        clean_bairros(bairros_input_csv_path, BAIRROS_UNIQUE_PATH)

    # 1. Carrega o grafo com os nós e métricas
    G = load_nodes_with_metrics(BAIRROS_UNIQUE_PATH)
    
    # 2. Adiciona as arestas (conexões)
    add_edges_to_graph(G, adjacencias_csv_path)
    
    return G

# ... (O resto do seu código, incluindo __name__ == "__main__":, permanece)
def normalizar_subbairros(nome: str) -> str:

    if pd.isna(nome):
        return None

    nome = nome.strip().title()

    # Remove acentos para comparar de forma uniforme
    nome_sem_acento = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("utf-8")

    if "setubal" in nome_sem_acento.lower():
        return "Boa Viagem (Setúbal)"

    return nome

if __name__ == "__main__":
    input_csv = "data/bairros_recife.csv"
    output_csv = "data/bairros_unique.csv"
    clean_bairros(input_csv, output_csv)
