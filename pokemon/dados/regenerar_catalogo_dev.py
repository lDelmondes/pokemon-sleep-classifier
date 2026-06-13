"""
Regenera o catalogo do universo de DESENVOLVIMENTO (5.836 cartas, 38 sets) de
forma reproduzivel, a partir da lista canonica de sets abaixo. Substitui o
shutil.copy manual do backup pre-ingestao: agora o universo de dev nasce de
codigo, nao de um arquivo orfao.

Fonte: os 38 sets originais (ex-ingestao_38sets.py). Le tudo do cache em disco
(data/cache_cards/), entao roda offline e em segundos se o cache existe; se nao,
carta_completa busca da TCGdex e cacheia. NAO baixa imagens (ja estao no disco).

Uso:
    python -m pokemon.dados.regenerar_catalogo_dev            # gera o CSV
    python -m pokemon.dados.regenerar_catalogo_dev --validar  # compara com o atual, NAO sobrescreve
"""
import sys
import pandas as pd
from pokemon.caminhos import RAW
from pokemon.dados.tcgdex_utils import cartas_do_set, carta_completa

# --- Lista canonica do universo de DESENVOLVIMENTO: 38 sets ---
# (set_id_tcgdex: (set_codigo, era)) — copiada verbatim do ingestao_38sets.py
SETS = {
    "sv01": ("SVI", "SV"), "sv02": ("PAL", "SV"), "sv03.5": ("MEW", "SV"),
    "sv05": ("TEF", "SV"), "sv06": ("TWM", "SV"), "sv07": ("SCR", "SV"),
    "sv08": ("SSP", "SV"), "sv10.5w": ("WHT", "SV"), "svp": ("SVP", "SV"),
    "swsh5": ("BST", "SWSH"), "swsh7": ("EVS", "SWSH"), "swsh9": ("BRS", "SWSH"),
    "swsh11": ("LOR", "SWSH"), "swsh12.5": ("CRZ", "SWSH"), "swsh8": ("FST", "SWSH"),
    "swsh10": ("ASR", "SWSH"),
    "sm6": ("FLI", "SM"), "sm10": ("UNB", "SM"), "sm11": ("UNM", "SM"),
    "sm115": ("HIF", "SM"), "sma": ("HFV", "SM"), "sm12": ("CEC", "SM"),
    "sm8": ("LOT", "SM"), "sm9": ("TEU", "SM"),
    "xy2": ("FLF", "XY"), "xy6": ("ROS", "XY"), "xy7": ("AOR", "XY"),
    "xy8": ("BKT", "XY"), "xy11": ("STS", "XY"), "xy5": ("PRC", "XY"),
    "xy4": ("PHF", "XY"),
    "bw3": ("NVI", "BW"), "bw4": ("NXD", "BW"), "bw7": ("BCR", "BW"),
    "bw9": ("PLF", "BW"), "bw10": ("PLB", "BW"), "bw11": ("LTR", "BW"),
    "bw6": ("DRX", "BW"),
}

SAIDA = RAW / "catalogo_desenvolvimento.csv"


def extrair_rarity(full):
    """Le a rarity do objeto da carta, normalizando qualquer forma de 'vazio'
    (None, 'None', '', 'nan') para None real — pra o fillna('nan') depois casar
    com o catalogo atual. Promos (svp-*) e energias costumam vir sem rarity."""
    r = full.get("rarity")
    if r is None or str(r).strip().lower() in ("none", "nan", ""):
        return None
    return r


def montar():
    linhas = []
    for sid, (codigo, era) in SETS.items():
        print(f"=== {codigo} ({sid}, {era}) ===", flush=True)
        for resumo in cartas_do_set(sid):
            full = carta_completa(resumo["id"])
            if full.get("category") != "Pokemon":
                continue
            linhas.append({
                "card_id": full["id"],
                "set_codigo": codigo,
                "era": era,
                "numero": full.get("localId"),
                "nome": full["name"],
                "image_url": (f"{full['image']}/high.png" if full.get("image") else None),
                "rarity": extrair_rarity(full),
                "label": 0,
            })
    df = pd.DataFrame(linhas)
    df["rarity"] = df["rarity"].fillna("nan")     # <-- None -> "nan", igual ao catalogo atual
    return df.sort_values("card_id").reset_index(drop=True)


def validar(df_novo):
    """Compara o regenerado com o catalogo_desenvolvimento.csv atual. Nao escreve nada."""
    if not SAIDA.exists():
        print("Nao ha catalogo atual pra comparar. Rode sem --validar pra criar o primeiro.")
        return
    atual = pd.read_csv(SAIDA).sort_values("card_id").reset_index(drop=True)
    print(f"\n--- VALIDACAO ---")
    print(f"Linhas:  atual={len(atual)}  regenerado={len(df_novo)}  "
          f"{'OK' if len(atual) == len(df_novo) else 'DIVERGE'}")

    ids_atual, ids_novo = set(atual["card_id"]), set(df_novo["card_id"])
    so_atual, so_novo = ids_atual - ids_novo, ids_novo - ids_atual
    if so_atual: print(f"  Cards SO no atual ({len(so_atual)}): {sorted(so_atual)[:10]}")
    if so_novo:  print(f"  Cards SO no regenerado ({len(so_novo)}): {sorted(so_novo)[:10]}")

    # rarity: a coluna critica. Compara carta a carta nos IDs em comum.
    comum = sorted(ids_atual & ids_novo)
    a = atual.set_index("card_id").loc[comum, "rarity"].fillna("nan").astype(str)
    n = df_novo.set_index("card_id").loc[comum, "rarity"].fillna("nan").astype(str)
    difere = (a != n)
    print(f"Rarity: {int(difere.sum())} divergencias em {len(comum)} cards comuns "
          f"{'OK' if difere.sum() == 0 else 'DIVERGE'}")
    if difere.sum():
        ex = a[difere].index[:8]
        for cid in ex:
            print(f"  {cid}: atual='{a[cid]}'  regenerado='{n[cid]}'")
    print("Se tudo OK, o script e fiel: pode adotar como fonte do universo de dev.")


def main():
    df = montar()
    print(f"\nTotal regenerado: {len(df)} cartas de Pokemon, {df['rarity'].notna().sum()} com rarity")
    if "--validar" in sys.argv:
        validar(df)
    else:
        df.to_csv(SAIDA, index=False, encoding="utf-8")
        print(f"Salvo: {SAIDA}")
        print(df.groupby(["era", "set_codigo"]).size().to_string())


if __name__ == "__main__":
    main()