"""
Reverte o projeto ao estado do Exp. 9 (catalogo ~4.530 sem o lote 3, gabarito 212)
para re-rodar o treino e medir a distribuicao do PR-AUC. NAO DESTRUTIVO:
- le do backup catalogo_5802.csv (nao toca no original ate o fim)
- usa positivos_ids_backup.csv (212) como fonte
- regenera catalogo_completo.csv filtrado e gabarito.csv
Restaurar depois com exp9_restaurar.py.
"""
from pokemon.caminhos import RAW, LABELS
import sys
import pandas as pd

SETS_LOTE3 = ["FST", "ASR", "LOT", "TEU", "PRC", "PHF", "LTR", "DRX"]

def main():
    # Travas: os backups TEM que existir, senao aborta sem mexer em nada
    bkp_cat = RAW / "catalogo_5802.csv"
    bkp_278 = LABELS / "positivos_ids_278.csv"
    bkp_212 = LABELS / "positivos_ids_backup.csv"
    for f in (bkp_cat, bkp_278, bkp_212):
        if not f.exists():
            print(f"ABORTADO: backup ausente -> {f}")
            sys.exit(1)

    # 1. Filtra o catalogo: remove os 8 sets do lote 3 (le do backup 5802)
    cat = pd.read_csv(bkp_cat)
    antes = len(cat)
    cat_filtrado = cat[~cat["set_codigo"].isin(SETS_LOTE3)].copy()
    cat_filtrado.to_csv(RAW / "catalogo_completo.csv", index=False, encoding="utf-8")
    print(f"Catalogo: {antes} -> {len(cat_filtrado)} cartas "
          f"({cat_filtrado['set_codigo'].nunique()} sets)")

    # 2. Fonte de positivos = 212 (backup). Copia por cima do ativo.
    ids_212 = pd.read_csv(bkp_212)
    ids_212.to_csv(LABELS / "positivos_ids.csv", index=False, encoding="utf-8")
    print(f"positivos_ids.csv <- backup de {len(ids_212)} positivos")

    print("\nEstado Exp. 9 montado. Agora rode:")
    print("  python -m pokemon.dados.montar_gabarito   (deve dar 212, zero inexistentes)")
    print("  python -m pokemon.modelagem.split          (deve dar ~4530 cartas, 212 pos)")
    print("  python -m pokemon.modelagem.treinar_finetuning_siglip   (3x, anote PR-AUC)")

if __name__ == "__main__":
    main()