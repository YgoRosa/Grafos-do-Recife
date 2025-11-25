import streamlit as st
# Importa o pacote graphs/ e seus módulos (io, algorithms, etc.)
from graphs import io, algorithms 

import viz 

try:
    G_recife = io.load_recife_graph('data/bairros_recife.csv', 'data/adjacencias_bairros.csv')
    # Pre-calcular métricas globais, ego-subredes, etc.
    # Exemplo: global_metrics = algorithms.calculate_global_metrics(G_recife)
except Exception as e:
    st.error(f"Erro ao carregar os dados do Recife: {e}")
    G_recife = None

# --- Barra de Navegação (Navbar) ---
st.sidebar.title("Projeto Final: Grafos do Recife")
screen = st.sidebar.selectbox(
    "Escolha a Tela:",
    ("1. Grafo Todo: Análise Métrica",
     "2. Interatividade: Requisitos do Ponto 9",
     "3. Algoritmos: Comparação (Parte 2)")
)

# --- Definição das Telas ---

def screen_grafo_todo():
    """Implementa a Tela 1: Grafo Todo."""
    st.header("🗺️ Grafo dos Bairros do Recife: Análise Métrica")
    st.markdown("---")
    
    if G_recife:
        # 1. Métricas Globais (Seção 3.1)
        st.subheader("Métricas Globais (Recife)")
        # Carregar ou calcular: Ordem, Tamanho, Densidade [cite: 79]
        global_metrics = algorithms.calculate_global_metrics(G_recife)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Ordem (Nós)", global_metrics["ordem"])
        
        with col2:
            st.metric("Tamanho (Arestas)", global_metrics["tamanho"])
            
        with col3:
            # Exibe a densidade formatada em 4 casas decimais
            st.metric("Densidade", f"{global_metrics['densidade']:.4f}")
        # [FIM NOVO CÓDIGO]
        
        # 2. Métricas por Microrregião (Seção 3.2)
        st.subheader("Métricas por Microrregião")
        # Caminho de saída (Certifique-se que o diretório 'out' existe)
        OUT_JSON_PATH = "out/microrregioes.json"
        
        try:
            df_microrregiao = algorithms.extract_and_calculate_microrregiao_metrics(
                G_recife,
                OUT_JSON_PATH
            )
            
            # Exibir o DataFrame como uma tabela interativa
            st.dataframe(df_microrregiao, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao calcular métricas por microrregião: {e}")
            
        # [FIM NOVO CÓDIGO]

        st.markdown("---")

        # 3. Graus e Rankings (Seção 4)
        st.subheader("Ranking de Bairros")
        # Mostrar Bairro com maior grau e Bairro mais denso [cite: 88]
        # Exemplo: viz.render_degree_ranking('out/graus.csv')

        # 4. Visualizações Analíticas (Seção 8)
        st.subheader("Explorações e Insights")
        # Exemplo: viz.render_degree_map(G_recife) # Mapa de cores por grau [cite: 118]
        # Exemplo: viz.render_degree_distribution('out/graus.csv') # Histograma [cite: 121]
    else:
        st.warning("Grafo do Recife não carregado. Verifique os arquivos de entrada.")


def screen_interatividade_ponto9():
    """Implementa a Tela 2: Grafo Interativo e Percurso."""
    st.header("🖱️ Interatividade e Percurso: Requisitos 7 e 9")
    st.markdown("---")
    
    # 1. Grafo Interativo (Seção 9)
    st.subheader("Grafo Interativo do Recife")
    st.markdown("Esta visualização deve ser o conteúdo do `out/grafo_interativo.html`.")
    st.markdown("Inclui **Tooltips** (grau, microrregião, densidade_ego), **Caixa de Busca** e **Botão para Realçar Percurso**[cite: 127, 128, 129].")
    #st.components.v1.html(open('out/grafo_interativo.html', 'r').read(), height=600)

    st.markdown("---")
    
    # 2. Percurso Nova Descoberta → Setúbal (Seção 6 e 7)
    st.subheader("Percurso: Nova Descoberta → Boa Viagem (Setúbal)")
    st.markdown("Caminho encontrado por Dijkstra:")
    # Carregar e mostrar custo e caminho [cite: 109, 110]
    # Exemplo: st.json(io.load_json('out/percurso_nova_descoberta_setubal.json'))
    
    st.subheader("Árvore de Caminho (Percurso)")
    st.markdown("Visualização estática ou interativa do subgrafo do percurso (`out/arvore_percurso.html|png`)[cite: 114].")
    # Exemplo: viz.render_route_tree('out/arvore_percurso.html')


def screen_algoritmos_parte2():
    """Implementa a Tela 3: Comparação de Algoritmos (Parte 2)."""
    st.header("⚙️ Comparação de Algoritmos (Parte 2)")
    st.markdown("---")

    # 1. Descrição do Dataset (Parte 2)
    st.subheader("Dataset Maior Escolhido")
    st.markdown("Descreva $|V|$, $|E|$, tipo (dirigido/ponderado) e distribuição de graus[cite: 137].")

    st.markdown("---")
    
    # 2. Execução e Resultados
    st.subheader("Resultados dos Algoritmos")
    st.markdown("**BFS/DFS:** Camadas e ciclos a partir de $>3$ fontes[cite: 139].")
    st.markdown("**Dijkstra:** Custo e percurso para $\ge5$ pares com pesos $\ge0$[cite: 139].")
    st.markdown("**Bellman-Ford:** Resultados com peso negativo e **detecção de ciclo negativo** (flag)[cite: 140, 163].")

    st.markdown("---")

    # 3. Métricas de Desempenho (Seção 141)
    st.subheader("Tabela de Desempenho")
    # Carregar e mostrar a tabela de tempos e memória [cite: 141, 157]
    # Exemplo: st.json(io.load_json('out/parte2_report.json'))
    
    # 4. Discussão Crítica (Seção 143)
    st.subheader("Análise Crítica")
    st.markdown("Discuta quando e por que cada algoritmo é mais adequado; limites do design de pesos[cite: 143].")
    
    # 5. Visualização (Seção 142)
    st.subheader("Visualização Adicional (Parte 2)")
    st.markdown("Ex.: Amostra do grafo, distribuição de graus, *heatmap* de distâncias[cite: 142].")


# --- Orquestração da Escolha ---

if screen == "1. Grafo Todo: Análise Métrica":
    screen_grafo_todo()
elif screen == "2. Interatividade: Requisitos do Ponto 9":
    screen_interatividade_ponto9()
elif screen == "3. Algoritmos: Comparação (Parte 2)":
    screen_algoritmos_parte2()

# Para executar, assumindo que você está usando Streamlit:
# Instale: pip install streamlit
# Execute no terminal: streamlit run src/streamlit_app.py