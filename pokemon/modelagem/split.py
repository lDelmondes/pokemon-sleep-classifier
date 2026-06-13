"""

Divide as cartas em treino/validacao/teste de forma ESTRATIFICADA (mantendo a proporcao de positivos em cada grupo). Split unico, com random_state fixo para ser reproduzivel.

Modulo importado pelo script de treino.

"""
from pokemon.caminhos import RAW, LABELS
import pandas as pd
from sklearn.model_selection import train_test_split

CATALOGO_DEV = RAW / "catalogo_desenvolvimento.csv"

def carregar_dados_rotulados():
    """card_id + label para as cartas COM imagem do universo de DESENVOLVIMENTO."""
    cat = pd.read_csv(CATALOGO_DEV)
    cat = cat[cat["image_url"].notna()].copy()
    positivos = set(pd.read_csv(LABELS / "gabarito.csv")["card_id"])
    cat["label"] = cat["card_id"].isin(positivos).astype(int)
    return cat[["card_id", "set_codigo", "era", "label"]].reset_index(drop=True)

def dividir(df, random_state=42):
    """Split estratificado 70/15/15. Estratifica pela coluna label para
    manter a proporcao de positivos nos tres grupos."""
    # 1o corte: separa 15% para TESTE
    treino_val, teste = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=random_state)
    # 2o corte: do restante (85%), separa ~15/85 para VALIDACAO
    treino, val = train_test_split(
        treino_val, test_size=0.1765, stratify=treino_val["label"],
        random_state=random_state)
    return treino.reset_index(drop=True), val.reset_index(drop=True), teste.reset_index(drop=True)

def main():
    df = carregar_dados_rotulados()
    treino, val, teste = dividir(df)
    for nome, grupo in [("TREINO", treino), ("VALIDACAO", val), ("TESTE", teste)]:
        n, pos = len(grupo), int(grupo["label"].sum())
        print(f"{nome:10}: {n:4} cartas | {pos:2} positivos ({pos/n:.1%})")

if __name__ == "__main__":
    main()