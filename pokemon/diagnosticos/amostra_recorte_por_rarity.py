"""
Para cada raridade do catalogo, pega N cartas e salva a versao RECORTADA (55%
superior) numa pasta, para inspecao visual: o rosto/olho do Pokemon sobrevive
ao recorte simples mesmo nas raridades full-art/special?
Saida: data/amostra_recorte/<rarity>/<card_id>.png
"""
from pokemon.caminhos import RAW, IMAGES
import pandas as pd
from PIL import Image

N_POR_RARIDADE = 5
FRACAO = 0.55

def main():
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    cat = cat[cat["image_url"].notna()]
    base = RAW.parent / "amostra_recorte"
    base.mkdir(exist_ok=True)

    print("Raridades no catalogo:")
    print(cat["rarity"].value_counts(dropna=False).to_string())
    print(f"\nGerando {N_POR_RARIDADE} recortes por raridade em {base}/ ...")

    for rar, grupo in cat.groupby("rarity", dropna=False):
        nome_rar = str(rar).replace(" ", "_").replace("/", "_")
        pasta = base / nome_rar
        pasta.mkdir(exist_ok=True)
        amostra = grupo.head(N_POR_RARIDADE)
        for _, r in amostra.iterrows():
            cid = r["card_id"]
            arq = IMAGES / f"{cid}.png"
            if not arq.exists():
                continue
            img = Image.open(arq).convert("RGB")
            w, h = img.size
            img.crop((0, 0, w, int(h * FRACAO))).save(pasta / f"{cid}.png")
        print(f"  {nome_rar}: {len(amostra)} cartas")

if __name__ == "__main__":
    main()