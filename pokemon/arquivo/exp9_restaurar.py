"""
Restaura o estado de producao (catalogo 5.802 completo, gabarito 278) apos o
teste de reproducao do Exp. 9. NAO DESTRUTIVO: le dos backups.
"""
from pokemon.caminhos import RAW, LABELS
import sys
import pandas as pd

def main():
    bkp_cat = RAW / "catalogo_5802.csv"
    bkp_278 = LABELS / "positivos_ids_278.csv"
    for f in (bkp_cat, bkp_278):
        if not f.exists():
            print(f"ABORTADO: backup ausente -> {f}")
            sys.exit(1)

    # Restaura catalogo completo
    cat = pd.read_csv(bkp_cat)
    cat.to_csv(RAW / "catalogo_completo.csv", index=False, encoding="utf-8")
    print(f"catalogo_completo.csv <- {len(cat)} cartas (5802)")

    # Restaura fonte de 278
    ids_278 = pd.read_csv(bkp_278)
    ids_278.to_csv(LABELS / "positivos_ids.csv", index=False, encoding="utf-8")
    print(f"positivos_ids.csv <- {len(ids_278)} positivos (278)")

    print("\nEstado de producao restaurado. Rode para reconciliar o gabarito:")
    print("  python -m pokemon.dados.montar_gabarito   (deve dar 278, zero inexistentes)")

if __name__ == "__main__":
    main()