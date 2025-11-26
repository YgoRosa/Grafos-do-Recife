import os
import json
import pandas as pd
from pyvis.network import Network
from typing import List, Tuple, Dict
from graphs.graph import Graph 
import webbrowser
import matplotlib.pyplot as plt
import numpy as np 


def construir_arestas_arvore_percurso(graph, path_nodes: List[str]) -> List[Tuple[str, str, float, Dict]]:
    """
    Constrói a lista de arestas que formam o percurso a partir de uma lista de nós.
    Usa o método get_edge_data da sua classe Graph.
    """
    edges = []
    # Itera sobre os nós consecutivos no caminho
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        # O método get_edge_data retorna (peso, meta) ou None.
        data = graph.get_edge_data(u, v)

        if data is not None:
            weight, meta = data
            # Como o percurso é (u, v), a aresta deve ser salva nessa ordem (u, v)
            edges.append((u, v, weight, meta))
        else:
            # Caso não encontre a aresta (nunca deve acontecer se o Dijkstra for correto)
            print(f"[ERRO - VIZ] Aresta esperada ({u} -> {v}) não encontrada no grafo.")
            
    return edges

def visualize_path_tree(path_nodes: List[str], path_edges: List[Tuple[str, str, float, Dict]], output_file: str):
    """
    Gera a visualização interativa do subgrafo do percurso.
    """
    
    net = Network(height="750px", width="100%", 
                  directed=False, 
                  heading="Percurso: Nova Descoberta → Setúbal") 

    # 1. Adicionar Nós
    for node in path_nodes:
        # Destacar origem e destino
        color = '#38761d' if node == path_nodes[0] else \
                '#cc0000' if node == path_nodes[-1] else \
                '#3c78d8' # Cor para nós intermediários
                
        title_text = f"Bairro: **{node}**"
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=15)

    # 2. Adicionar Arestas
    
    # Parâmetros de escala:
    # Ajuste o fator e os limites conforme a faixa de pesos que você definiu na Seção 5.
    FATOR_ESCALA = 4.0 # Multiplica o peso para amplificar a diferença visual.
    MIN_WIDTH = 2.0    # Espessura mínima para qualquer aresta do caminho.
    MAX_WIDTH = 15.0   # Espessura máxima para evitar linhas muito grandes.

    for u, v, weight, meta in path_edges:
        
        # 1. Calcular a espessura proporcional ao peso
        proportional_width = weight * FATOR_ESCALA
        
        # 2. Aplicar limites
        edge_width = max(MIN_WIDTH, min(MAX_WIDTH, proportional_width))
        
        # Rótulo para o Tooltip
        label = f"Custo: {weight:.2f}"
        if "logradouro" in meta:
            label += f"<br>Via: {meta['logradouro']}"

        # Adicionar aresta com a espessura calculada
        net.add_edge(source=u, to=v, weight=weight, title=label, 
                     color='#ff9900', 
                     value=weight, 
                     width=edge_width # <-- A espessura agora é dinâmica!
                    )

    # 3. Salvar o HTML, forçando a codificação UTF-8 para evitar o UnicodeEncodeError
    
    # 3a. Gera o HTML do gráfico como uma string (Notebook=False para HTML autônomo)
    html_content = net.generate_html(notebook=False)
    
    # 3b. Escreve a string em um arquivo, especificando explicitamente a codificação UTF-8
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        
        # Opcional: tentar abrir o navegador, mas a escrita é o que importa para o projeto
        # import webbrowser
        # webbrowser.open(output_file)

        print(f"[VIZ] Árvore de percurso interativa gerada → {output_file}")

        webbrowser.open(output_file)
    except Exception as e:
        print(f"[ERRO DE ESCRITA] Não foi possível salvar ou escrever o arquivo HTML: {e}")

def visualize_degree_map(graph: Graph, df_graus: pd.DataFrame, output_file: str):
    """
    1. Mapa de Cores por Grau do Bairro (Visualização Analítica 1)
    Gera uma visualização do grafo onde o tamanho/cor do nó é proporcional ao seu grau.
    """
    print(f"[VIZ] Gerando Mapa de Cores por Grau → {output_file}")
    
    # Prepara a rede Pyvis
    net = Network(height="750px", width="100%", 
                  directed=False, 
                  heading="Grafo de Bairros do Recife: Visualização de Grau") 

    # Normaliza o Grau para Escala de Cores/Tamanhos
    max_degree = df_graus['grau'].max()
    min_degree = df_graus['grau'].min()
    degree_range = max_degree - min_degree
    
    # Dicionário de Grau para acesso rápido
    degree_map = df_graus.set_index('bairro')['grau'].to_dict()

    # Define a escala de cor (verde escuro para alto grau, amarelo claro para baixo)
    def degree_to_color(degree):
        if degree_range == 0:
            norm_degree = 0.5
        else:
            norm_degree = (degree - min_degree) / degree_range
        # Mapeia 0 (min_degree) para claro (amarelo) e 1 (max_degree) para escuro (verde/azul)
        # Escolha um esquema de cor BGR (R, G, B) em hexadecimal, por exemplo:
        r = int(255 * (1 - norm_degree)) # Diminui Vermelho
        g = int(255 * norm_degree)       # Aumenta Verde
        b = int(100 * (1 - norm_degree)) # Diminui Azul
        return f'#{r:02x}{g:02x}{b:02x}' # Ex: #FF0064 para baixo, #00FF00 para alto

    # 1. Adicionar Nós
    for node in graph.adj.keys():
        degree = degree_map.get(node, 0)
        color = degree_to_color(degree)
        size = 10 + (degree * 2) # Tamanho proporcional ao grau (Ajuste o fator)
        title_text = f"Bairro: **{node}**<br>Grau: {degree}"
        
        net.add_node(n_id=node, label=node, title=title_text, color=color, size=size)

    # 2. Adicionar Arestas
    for u, v, weight, meta in graph.get_edges():
        net.add_edge(source=u, to=v, weight=weight, title=f"Custo: {weight:.2f}", color='#999999', value=weight, width=1)

    # 3. Salvar o HTML
    html_content = net.generate_html(notebook=False)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Visualização de Grau gerada em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")


def visualize_degree_histogram(df_graus: pd.DataFrame, output_file: str):
    """
    2. Distribuição dos Graus (Visualização Analítica 2)
    Gera um histograma da distribuição de graus dos bairros.
    """
    print(f"[VIZ] Gerando Histograma de Graus → {output_file}")
    
    degrees = df_graus['grau'].dropna().tolist()
    
    if not degrees:
        print("[AVISO VIZ] Não há dados de grau para gerar o histograma.")
        return

    plt.figure(figsize=(10, 6))
    
    # Calcula os bins (caixas/intervalos)
    bins = np.arange(min(degrees), max(degrees) + 1.5) - 0.5
    
    plt.hist(degrees, bins=bins, color='#4682B4', edgecolor='black', rwidth=0.9)
    
    plt.title('Distribuição de Graus dos Bairros do Recife', fontsize=16)
    plt.xlabel('Grau (Número de Interconexões)', fontsize=12)
    plt.ylabel('Frequência (Número de Bairros)', fontsize=12)
    plt.xticks(np.arange(min(degrees), max(degrees) + 1)) # Garante ticks em números inteiros
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    try:
        plt.savefig(output_file)
        plt.close() # Fecha a figura para liberar memória
        print(f"[OK] Histograma de Graus gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")


def visualize_top_10_degree_subgraph(graph: Graph, df_graus: pd.DataFrame, output_file: str):
    """
    3. Subgrafo dos 10 Bairros com Maior Grau (Visualização Analítica 3)
    Gera uma visualização contendo apenas os 10 nós de maior grau e as arestas entre eles.
    """
    print(f"[VIZ] Gerando Subgrafo Top 10 por Grau → {output_file}")
    
    # 1. Seleciona os Top 10 Bairros por Grau
    top_10_df = df_graus.sort_values(by='grau', ascending=False).head(10)
    top_nodes = set(top_10_df['bairro'].tolist())
    top_degree_map = top_10_df.set_index('bairro')['grau'].to_dict()

    if not top_nodes:
        print("[AVISO VIZ] Não há nós suficientes para o Top 10.")
        return

    # 2. Prepara a rede Pyvis para o subgrafo
    net = Network(height="750px", width="100%", 
                  directed=False, 
                  heading="Subgrafo dos 10 Bairros Mais Conectados (Maior Grau)") 
    
    # 3. Adiciona Nós (apenas o Top 10)
    max_degree = top_10_df['grau'].max() if not top_10_df.empty else 1
    
    for node in top_nodes:
        degree = top_degree_map.get(node, 0)
        size = 15 + (degree / max_degree) * 20 # Escala de 15 a 35
        title_text = f"Bairro: **{node}**<br>Grau: {degree}"
        
        # Cor de destaque para o núcleo
        net.add_node(n_id=node, label=node, title=title_text, color='#FFC107', size=size) 

    # 4. Adiciona Arestas (apenas as que conectam nós do Top 10)
    for u, v, weight, meta in graph.get_edges():
        if u in top_nodes and v in top_nodes:
             net.add_edge(source=u, to=v, weight=weight, title=f"Custo: {weight:.2f}", color='#FF9800', value=weight, width=2)

    # 5. Salvar o HTML
    html_content = net.generate_html(notebook=False)
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(html_content)
        print(f"[OK] Subgrafo Top 10 gerado em {output_file}")
    except Exception as e:
        print(f"[ERRO VIZ] Falha ao salvar {output_file}: {e}")

def compute_parte2_degrees(g: Graph) -> pd.DataFrame:
    """
    Calcula o Grau de Saída (Out-Degree) para o grafo dirigido da Parte 2.
    """
    rows = []
    for node in g.get_nodes():
        # g.degree(node) retorna o grau de saída em um grafo dirigido
        grau_saida = g.degree(node) 
        rows.append({"aeroporto": node, "grau_saida": grau_saida})

    df = pd.DataFrame(rows, columns=["aeroporto", "grau_saida"])
    return df

def visualize_parte2_degree_histogram(df_graus_p2: pd.DataFrame, output_file: str):
    """
    Gera um histograma da Distribuição de Graus de Saída (Out-Degree) do grafo aéreo.
    """
    print(f"[VIZ] Gerando Histograma de Graus (Parte 2) → {output_file}")
    
    degrees = df_graus_p2['grau_saida'].dropna().tolist()
    
    if not degrees:
        print("[AVISO VIZ P2] Não há dados de grau para gerar o histograma da Parte 2.")
        return

    plt.figure(figsize=(10, 6))
    
    # Calcula os bins (caixas/intervalos)
    bins = np.arange(min(degrees), max(degrees) + 1.5) - 0.5
    
    plt.hist(degrees, bins=bins, color='#CC5500', edgecolor='black', rwidth=0.9) # Cor Laranja
    
    plt.title('Distribuição de Graus de Saída (Rede Aérea)', fontsize=16)
    plt.xlabel('Grau de Saída (Número de Rotas Internacionais)', fontsize=12)
    plt.ylabel('Frequência (Número de Aeroportos)', fontsize=12)
    plt.xticks(np.arange(min(degrees), max(degrees) + 1, step=5))
    plt.grid(axis='y', alpha=0.75)
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
    df_graus: pd.DataFrame,
    caminho_obrig: list,
    output_file: str
):
    """
    Versão robusta do grafo interativo:
    - usa as colunas reais existentes em df_ego (ordem_ego, tamanho_ego, densidade_ego)
    - busca microrregião em data/bairros_unique.csv se necessário
    - adiciona caixa de busca e botão para destacar o caminho obrigatório
    """
    print("[VIZ] Gerando grafo interativo completo...")

    # --- Preparar dicionários a partir dos DataFrames (tolerante a nomes de colunas) ---
    # graus
    if "bairro" in df_graus.columns and "grau" in df_graus.columns:
        grau_map = df_graus.set_index("bairro")["grau"].to_dict()
    else:
        grau_map = {}
    # ego: tentamos várias colunas possíveis
    ego_size_col = None
    for cand in ("ordem_ego", "tamanho_ego", "ego_size", "grau", "degree"):
        if cand in df_ego.columns:
            ego_size_col = cand
            break
    dens_col = None
    for cand in ("densidade_ego", "densidade", "density", "densidade"):
        if cand in df_ego.columns:
            dens_col = cand
            break

    ego_map = {}
    dens_map = {}
    if ego_size_col:
        ego_map = df_ego.set_index("bairro")[ego_size_col].to_dict()
    if dens_col:
        dens_map = df_ego.set_index("bairro")[dens_col].to_dict()

    # microrregiao: se estiver no df_ego, usamos; senão tentamos carregar data/bairros_unique.csv
    micro_map = {}
    if "microrregiao" in df_ego.columns:
        micro_map = df_ego.set_index("bairro")["microrregiao"].to_dict()
    else:
        # fallback: attempt reading data/bairros_unique.csv
        try:
            import pandas as _pd
            br = _pd.read_csv("data/bairros_unique.csv", encoding="utf-8")
            if "bairro" in br.columns and "microrregiao" in br.columns:
                micro_map = br.set_index("bairro")["microrregiao"].to_dict()
        except Exception:
            micro_map = {}

    # --- Construir pyvis network ---
    net = Network(height="900px", width="100%", directed=False, notebook=False,
                  heading="Grafo dos Bairros do Recife — Interativo")
    try:
        net.barnes_hut()
    except Exception:
        pass  # alguns ambientes podem não suportar física avançada

    import math
    nodes_list = list(graph.get_nodes())
    N = len(nodes_list)

    pos_map = {}
    radius = 1000  # ajusta se quiser o círculo maior/menor

    for i, node in enumerate(nodes_list):
        angle = 2 * math.pi * i / N
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pos_map[node] = (x, y)

    # Adicionar nós
    for node in graph.get_nodes():
        grau = int(grau_map.get(node, 0)) if node in grau_map else 0
        ego_val = ego_map.get(node, None)
        dens_val = dens_map.get(node, None)
        mic = micro_map.get(node, "")

        # visual sizing
        size = 12 + min(grau, 20)
        color = "#3c78d8"
        if caminho_obrig and node in caminho_obrig:
            color = "#ffcc00"
            size = max(size, 18)

        title_lines = [f"<b>{node}</b>", f"Grau: {grau}"]
        if mic:
            title_lines.append(f"Microrregião: {mic}")
        if ego_val is not None:
            title_lines.append(f"Ordem/Ego: {ego_val}")
        if dens_val is not None:
            try:
                title_lines.append(f"Densidade ego: {float(dens_val):.4f}")
            except Exception:
                title_lines.append(f"Densidade ego: {dens_val}")

        tooltip = "<br>".join(title_lines)

        x, y = pos_map[node]
        net.add_node(
            n_id=node,
            label=node,
            title=tooltip,
            size=size,
            color=color,
            x=x,
            y=y,
            physics=False  # impede bagunçar o círculo
        )

    # Adicionar arestas
    for u, v, weight, meta in graph.get_edges():
        meta = meta or {}
        log = meta.get("logradouro", "")
        obs = meta.get("observacao", "")
        title = f"Peso: {weight}"
        if log:
            title += f"<br>Via: {log}"
        if obs:
            title += f"<br>{obs}"
        net.add_edge(source=u, to=v, value=weight if weight is not None else 1.0,
                     title=title, width=1, color="#97C2FC")

    # Gerar HTML base
    html_str = net.generate_html()

    # Preparar JSON do caminho obrigatório (lista de nomes)
    import json
    caminho_json = json.dumps(caminho_obrig or [], ensure_ascii=False)

    # Injetar caixa de busca e botão de destaque
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

    // === BUSCA: DESTACA NÓ + VIZINHOS ===
    function doSearch() {{
        const q = document.getElementById('nodeSearch').value.trim().toLowerCase();
        if (!q) return;

        const nodes = network.body.data.nodes.get();
        const edges = network.body.data.edges.get();

        // encontrar nó por busca direta ou parcial
        let found = nodes.find(n => n.label.toLowerCase() === q) ||
                    nodes.find(n => n.label.toLowerCase().includes(q));

        if (!found) {{
            alert("Bairro não encontrado: " + q);
            return;
        }}

        // RESETAR TODOS OS NÓS E ARESTAS
        for (const n of nodes) {{
            network.body.data.nodes.update({{ id: n.id, color: "#3c78d8", size: n.size }});
        }}
        for (const e of edges) {{
            network.body.data.edges.update({{ id: e.id, color: "#97C2FC", width: 1 }});
        }}

        // DESTACAR O NÓ BUSCADO
        network.body.data.nodes.update({{ id: found.id, color: "#ff0000", size: 28 }});

        // ENCONTRAR VIZINHOS
        const neighbors = [];
        for (const e of edges) {{
            if (e.from === found.id) {{
                neighbors.push(e.to);
                network.body.data.edges.update({{ id: e.id, color: "#ff0000", width: 4 }});
            }}
            if (e.to === found.id) {{
                neighbors.push(e.from);
                network.body.data.edges.update({{ id: e.id, color: "#ff0000", width: 4 }});
            }}
        }}

        // DESTACAR VIZINHOS (AMARELO)
        for (const nb of neighbors) {{
            network.body.data.nodes.update({{ id: nb, color: "#ff9900", size: 22 }});
        }}

        // CENTRALIZAR NA BUSCA
        network.focus(found.id, {{ scale: 1.4, animation: {{ duration: 300 }} }});
        network.selectNodes([found.id]);
    }}


    // === DESTACAR CAMINHO OBRIGATÓRIO ===
    function highlightPath() {{

        if (!requiredPath || requiredPath.length < 2) {{
            alert("Caminho obrigatório não disponível.");
            return;
        }}

        const nodes = network.body.data.nodes.get();
        const edges = network.body.data.edges.get();

        // RESETAR TUDO
        for (const n of nodes) {{
            network.body.data.nodes.update({{ id: n.id, color: "#3c78d8", size: n.size }});
        }}
        for (const e of edges) {{
            network.body.data.edges.update({{ id: e.id, color: "#97C2FC", width: 1 }});
        }}

        // DESTACAR NÓS DO CAMINHO
        for (let i = 0; i < requiredPath.length; i++) {{
            const name = requiredPath[i];
            const nodeObj = nodes.find(n => n.label === name);
            if (nodeObj) {{
                network.body.data.nodes.update({{
                    id: nodeObj.id,
                    color: "#ffcc00",
                    size: 25
                }});
            }}

            // DESTACAR ARESTAS ENTRE OS NÓS DO CAMINHO
            if (i < requiredPath.length - 1) {{
                const a = requiredPath[i];
                const b = requiredPath[i + 1];

                for (const e of edges) {{
                    if ((e.from === a && e.to === b) || (e.from === b && e.to === a)) {{
                        network.body.data.edges.update({{
                            id: e.id,
                            color: "#ff0000",
                            width: 5
                        }});
                    }}
                }}
            }}
        }}

        // FOCAR NO PRIMEIRO NÓ DO CAMINHO
        const first = requiredPath[0];
        const firstNode = nodes.find(n => n.label === first);
        if (firstNode) {{
            network.focus(firstNode.id, {{ scale: 1.3, animation: {{ duration: 300 }} }});
        }}
    }}

    </script>
    """


    # inserir antes do </body>
    if "</body>" in html_str:
        html_str = html_str.replace("</body>", js_extra + "\n</body>")
    else:
        html_str = html_str + js_extra

    # salvar
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"[OK] Grafo interativo salvo em {output_file}")
    except Exception as e:
        print(f"[ERRO] falha ao salvar {output_file}: {e}")

