# Grafos do Recife  
O projeto tem como objetivo modelar as conexões dos bairros da cidade de Recife e das pontes aéreas como grafos, aplicar os algoritmos estudados de busca e menor caminho (BFS, DFS, Dijkstra e Bellman-Ford) para extrair as métricas e calcular as melhores e mais eficientes rotas, além de comparar os desempenhos de cada um dos algoritmos e a aplicação destes em diferentes cenários.
<br>

# Como executar o projeto:
**1. Certifique-se de ter o Python (versão 3.x) instalado.**  

**2. Criar Ambiente Virtual**    
    Navegue até a pasta raiz do projeto e crie o ambiente:

    python -m venv .venv 

**3. Ativação do Ambiente**

**Windows:**  

    venv\Scripts\activate 

**Linux/Mac:**  

    source .venv/bin/activate

**4. Instalação de Dependências**  

      pip install -r requirements.txt 

**5. O projeto é executado através do arquivo *solve.py*:**  

      python src/solve.py 
  
  A execução carregará os dados, construirá os grafos, aplicará todos os algoritmos configurados (cálculos de rankings, distâncias, comparações) e salvará os resultados na pasta out/.  

  **6. Execução dos testes**  
  Certifique-se de que o ambiente virtual está ativo e execute o pytest a partir da raiz do projeto: 

      pytest

  
  Se der erro, executar:  
  
      python -m pytest -v 
  também na raiz do projeto, com o ambiente virtual ativado. O sistema executará todos os arquivos em tests/.  
<br>

# Estrutura do projeto
     grafos-do-recife/
    ├─ README.md
    ├─ requirements.txt
    ├─ data/
    │ ├─ bairros_recife.csv 
    │ ├─ adjacencias_bairros.csv 
    │ ├─ enderecos.csv
    │ ├─ bairros_unique.csv
    │ └─ dataset_parte2
    ├─ out/ 
    │ └─ .gitkeep
    ├─ src/
    │ ├─ solve.py
    │ └─ viz.py
    │ ├─ graphs/
    │ │ ├─ io.py 
    │ │ ├─ graph.py 
    │ │ └─ algorithms.py
    └─ tests/
    ├─ test_bfs.py
    ├─ test_dfs.py
    ├─ test_dijkstra.py
    └─ test_bellman_ford.py
<br>

# Time 
[Beatriz Pereira](https://github.com/biapereira2)  

[Manuela Cavalcanti](https://github.com/Manuelaamorim)  

[Ygor Rosa](https://github.com/YgoRosa)  




