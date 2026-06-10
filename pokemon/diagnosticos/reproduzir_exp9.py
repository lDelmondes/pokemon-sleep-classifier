"""
Reproduz o estado do Exp. 9 (catalogo 4.530, gabarito 212) para re-rodar e
checar se o PR-AUC 0.615 foi representativo ou sorte. NAO destrutivo:
- filtra o catalogo removendo os 8 sets do lote 3
- usa o backup de 212 como fonte de positivos
Restauracao no fim: comandos no final deste arquivo (comentados).
"""
from pokemon.caminhos import RAW, LABELS
import pandas as pd

SETS_LOTE3 = ["FST", "ASR", "LOT", "TEU", "PRC", "PHF", "LTR", "DRX"]

def main():
    # 1. Catalogo atual (5.802) -> filtra fora os 8 sets do lote 3
    cat = pd.read_csv(RAW / "catalogo_5802.csv")  # o backup que voce fez
    antes = len(cat)
    cat_filtrado = cat[~cat["set_codigo"].isin(SETS_LOTE3)].copy()
    depois = len(cat_filtrado)
    cat_filtrado.to_csv(RAW / "catalogo_completo.csv", index=False, encoding="utf-8")
    print(f"Catalogo: {antes} -> {depois} ({antes-depois} removidas do lote 3)")
    print(f"Sets restantes: {cat_filtrado['set_codigo'].nunique()}")

if __name__ == "__main__":
    main()