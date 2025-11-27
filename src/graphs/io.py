import pandas as pd
import unicodedata
import os
from src.graphs.graph import Graph

def normalize_bairro_name(name: str) -> str:
    if pd.isna(name):
        return None
    name = name.strip().title()
    return name

def melt_bairros(input_path: str, output_path: str):

    df = pd.read_csv(input_path)

    df_melted = df.melt(var_name="microrregiao", value_name="bairro")

    df_melted = df_melted.dropna(subset=["bairro"])

    df_melted["bairro"] = df_melted["bairro"].apply(normalize_bairro_name)

    df_unique = df_melted.drop_duplicates(subset=["bairro"])

    df_unique = df_unique[["bairro", "microrregiao"]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_unique.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ CSV derretido salvo em: {output_path}")
    print(f"Total de bairros únicos: {len(df_unique)}")


def carregar_bairros(csv_path: str) -> Graph:
    df = pd.read_csv(csv_path)
    g = Graph()

    for bairro in df["bairro"]:
        g.add_node(bairro)

    print(f"✅ {len(g)} bairros carregados no grafo.")
    return g

def normalizar_subbairros(nome: str) -> str:

    if pd.isna(nome):
        return None

    nome = nome.strip().title()

    nome_sem_acento = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("utf-8")

    if "setubal" in nome_sem_acento.lower():
        return "Boa Viagem (Setúbal)"

    return nome

if __name__ == "__main__":
    input_csv = "data/bairros_recife.csv"
    output_csv = "data/bairros_unique.csv"
    melt_bairros(input_csv, output_csv)