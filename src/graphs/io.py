import pandas as pd
import unicodedata
import os

def normalize_bairro_name(name: str) -> str:
    """Remove acentos, espaços extras e coloca tudo em minúsculas."""
    if pd.isna(name):
        return None
    name = name.strip().title()
    return name

def melt_bairros(input_path: str, output_path: str):
    """
    Derrete o CSV de microrregiões para uma lista única de bairros.

    Args:
        input_path (str): Caminho do CSV original (bairros agrupados).
        output_path (str): Caminho para salvar o CSV derretido.
    """
    # Lê o CSV original
    df = pd.read_csv(input_path)

    # "Derrete" todas as colunas num formato (microrregiao, bairro)
    df_melted = df.melt(var_name="microrregiao", value_name="bairro")

    # Remove linhas vazias
    df_melted = df_melted.dropna(subset=["bairro"])

    # Normaliza os nomes dos bairros
    df_melted["bairro"] = df_melted["bairro"].apply(normalize_bairro_name)

    # Remove duplicatas
    df_unique = df_melted.drop_duplicates(subset=["bairro"])

    df_unique = df_unique[["bairro", "microrregiao"]]

    # Salva o resultado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_unique.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ CSV derretido salvo em: {output_path}")
    print(f"Total de bairros únicos: {len(df_unique)}")

if __name__ == "__main__":
    input_csv = "data/bairros_recife.csv"
    output_csv = "data/bairros_unique.csv"
    melt_bairros(input_csv, output_csv)
