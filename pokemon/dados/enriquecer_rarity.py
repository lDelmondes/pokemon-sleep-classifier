"""
Adiciona a coluna 'rarity' ao catalogo, lendo do cache local (data/cache_cards/).
Nao toca a API. So enriquece os card_id que ja estao no catalogo (Pokemon),
entao trainers/energies do cache sao ignorados naturalmente pela juncao.

Saida: sobrescreve catalogo_completo.csv com a coluna 'rarity'.
"""
from pokemon.caminhos import RAW
import json
import pandas as pd

def main():
    cache = RAW.parent / "cache_cards"
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    print(f"Catalogo: {len(cat)} cartas de Pokemon")

    rarities = {}
    faltando = []
    for cid in cat["card_id"]:
        arq = cache / f"{cid}.json"
        if not arq.exists():
            faltando.append(cid)
            rarities[cid] = None
            continue
        dados = json.loads(arq.read_text(encoding="utf-8"))
        rarities[cid] = dados.get("rarity")

    cat["rarity"] = cat["card_id"].map(rarities)

    if faltando:
        print(f"AVISO: {len(faltando)} cartas sem JSON no cache (rarity=None): {faltando[:10]}")

    # distribuicao na base REAL de Pokemon (deve diferir do cache bruto)
    print("\nDistribuicao de rarity na base de Pokemon:")
    print(cat["rarity"].value_counts(dropna=False).to_string())

    # quantos POSITIVOS por raridade (util para a decisao do filtro full-art)
    if "label" in cat.columns:
        print("\nPositivos (label=1) por rarity:")
        print(cat[cat["label"] == 1]["rarity"].value_counts(dropna=False).to_string())

    cat.to_csv(RAW / "catalogo_completo.csv", index=False, encoding="utf-8")
    print(f"\nCatalogo salvo com a coluna 'rarity'.")

if __name__ == "__main__":
    main()