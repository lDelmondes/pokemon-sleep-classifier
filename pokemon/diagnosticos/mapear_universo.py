"""
Mapeia o universo COMPLETO de sets das 6 eras-alvo (bw, xy, sm, swsh, sv, me)
na TCGdex, e cruza com os sets que ja temos no catalogo para listar o que FALTA
ingerir. Nao baixa nada — so inventaria.
"""
from pokemon.caminhos import RAW
import requests
import pandas as pd

SERIES_ALVO = {
    "bw": "Black & White",
    "xy": "XY",
    "sm": "Sun & Moon",
    "swsh": "Sword & Shield",
    "sv": "Scarlet & Violet",
    "me": "Mega Evolution",
}

def main():
    # 1. Universo completo: todos os sets das 6 series
    universo = []
    for serie_id, serie_nome in SERIES_ALVO.items():
        r = requests.get(f"https://api.tcgdex.net/v2/en/series/{serie_id}")
        dados = r.json()
        for s in dados.get("sets", []):
            universo.append({
                "serie": serie_id,
                "set_id": s["id"],                      # id interno (sv01, me04...)
                "set_nome": s["name"],
                "cartas": s.get("cardCount", {}).get("total", "?"),
            })
    uni = pd.DataFrame(universo)
    print(f"Universo das 6 eras: {len(uni)} sets, "
          f"{uni['cartas'].apply(lambda x: x if isinstance(x,int) else 0).sum()} cartas (total bruto)\n")

    # 2. O que ja temos: descobrir como o catalogo identifica os sets
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    print(f"Catalogo atual: {len(cat)} cartas")
    print(f"Colunas: {cat.columns.tolist()}")
    # Tenta casar pelo set_codigo. Mostra os valores unicos para vermos o formato.
    tenho_codigos = set(cat["set_codigo"].unique())
    print(f"set_codigo unicos no catalogo ({len(tenho_codigos)}): {sorted(tenho_codigos)}\n")

    # 3. Cruzamento: o set_id da API (sv01) provavelmente NAO casa direto com
    #    set_codigo do catalogo (SVI). Mostramos os dois lado a lado para decidir.
    #    Extrai o card_id de exemplo de cada set do catalogo para achar o prefixo real.
    cat["prefixo_id"] = cat["card_id"].str.rsplit("-", n=1).str[0]  # sv01-035 -> sv01
    prefixos_catalogo = set(cat["prefixo_id"].unique())
    print(f"Prefixos reais dos card_id no catalogo ({len(prefixos_catalogo)}): {sorted(prefixos_catalogo)}\n")

    # Agora cruza pelo set_id da API contra os prefixos reais do card_id
    uni["ja_tenho"] = uni["set_id"].isin(prefixos_catalogo)

    tenho = uni[uni["ja_tenho"]]
    falta = uni[~uni["ja_tenho"]]

    print(f"=== RESUMO ===")
    print(f"Sets que JA TENHO: {len(tenho)}")
    print(f"Sets que FALTAM:   {len(falta)}")
    falta_cartas = falta["cartas"].apply(lambda x: x if isinstance(x,int) else 0).sum()
    print(f"Cartas a ingerir (estimativa): {falta_cartas}\n")

    print("=== SETS FALTANTES (por era) ===")
    for serie_id in SERIES_ALVO:
        sub = falta[falta["serie"] == serie_id]
        if len(sub):
            print(f"\n{serie_id} ({SERIES_ALVO[serie_id]}): {len(sub)} sets faltando")
            for _, r in sub.iterrows():
                print(f"  {r['set_id']:8} {r['cartas']:>4} cartas  {r['set_nome']}")

    # salva o inventario para a ingestao usar depois
    uni.to_csv(RAW / "inventario_sets.csv", index=False, encoding="utf-8")
    print(f"\nInventario salvo em inventario_sets.csv")

if __name__ == "__main__":
    main()