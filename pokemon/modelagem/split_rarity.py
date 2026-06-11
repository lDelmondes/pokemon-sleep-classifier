"""
Variante experimental do split: filtra o catalogo para apenas as raridades
de layout NORMAL (Common, Uncommon, Rare), usado no experimento de recorte
da arte. Mantem o split.py de producao intacto.
"""
from pokemon.caminhos import RAW, LABELS
import pandas as pd
from sklearn.model_selection import train_test_split

RARIDADES_PERMITIDAS = ["Common", "Uncommon", "Rare"]

def carregar_dados_rotulados():
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    cat = cat[cat["image_url"].notna()].copy()
    cat = cat[cat["rarity"].isin(RARIDADES_PERMITIDAS)].copy()   # << filtro novo
    positivos = set(pd.read_csv(LABELS / "gabarito.csv")["card_id"])
    cat["label"] = cat["card_id"].isin(positivos).astype(int)
    return cat[["card_id", "set_codigo", "era", "label"]].reset_index(drop=True)

def dividir(df, random_state=42):
    treino_val, teste = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=random_state)
    treino, val = train_test_split(
        treino_val, test_size=0.1765, stratify=treino_val["label"],
        random_state=random_state)
    return treino.reset_index(drop=True), val.reset_index(drop=True), teste.reset_index(drop=True)

def main():
    df = carregar_dados_rotulados()
    treino, val, teste = dividir(df)
    print(f"Raridades: {RARIDADES_PERMITIDAS}")
    for nome, grupo in [("TREINO", treino), ("VALIDACAO", val), ("TESTE", teste)]:
        n, pos = len(grupo), int(grupo["label"].sum())
        print(f"{nome:10}: {n:4} cartas | {pos:2} positivos ({pos/n:.1%})")

if __name__ == "__main__":
    main()