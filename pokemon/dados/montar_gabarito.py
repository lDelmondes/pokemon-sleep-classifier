"""

Consolida o gabarito de positivos (olhos fechados) a partir de uma fonte unica:
data/labels/positivos_ids.csv  (um card_id por linha, os 44 rotulados a mao)

Valida cada id contra o catalogo completo, marca se tem imagem, e salva o gabarito final. Avisa ids inexistentes (typos de transcricao).

Saida: data/labels/gabarito.csv

"""
from pokemon.caminhos import RAW, LABELS
import pandas as pd

def main():
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    com_img = set(cat.loc[cat["image_url"].notna(), "card_id"])

    # Fonte unica de positivos: uma coluna card_id
    ids = set(pd.read_csv(LABELS / "positivos_ids.csv")["card_id"])

    cat_ids = set(cat["card_id"])
    inexistentes = sorted(ids - cat_ids)
    validos = sorted(ids & cat_ids)

    gab = cat[cat["card_id"].isin(validos)][
        ["card_id", "set_codigo", "era", "numero", "nome"]].copy()
    gab["tem_imagem"] = gab["card_id"].isin(com_img)
    gab.to_csv(LABELS / "gabarito.csv", index=False, encoding="utf-8")

    print(f"Positivos unicos informados: {len(ids)}")
    print(f"  Validos (existem no catalogo): {len(validos)}")
    print(f"  Com imagem (entram no treino): {int(gab['tem_imagem'].sum())}")
    if inexistentes:
        print(f"  INEXISTENTES no catalogo (investigar): {inexistentes}")
    print(f"\nDistribuicao por set:")
    print(gab.groupby(["era", "set_codigo"]).size().to_string())
    print(f"\nGabarito salvo em {LABELS / 'gabarito.csv'}")

if __name__ == "__main__":
    main()