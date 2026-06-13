"""Pré-flight Exp.16: checa cobertura + gera contact-sheets dos 311 positivos
JÁ recortados pela janela da raridade deles. Rode e olhe os PNGs ANTES de treinar.
   python -m pokemon.diagnosticos.preflight_recorte_positivos
"""
from __future__ import annotations
import math
import pandas as pd
from PIL import Image
from pokemon.caminhos import RAW, IMAGES, ROOT
from pokemon.modelagem.recorte_rarity import checar_cobertura, recortar, LIMITE_INFERIOR

CATALOGO = RAW / "catalogo_completo.csv"               # ajuste se o nome diferir
POSITIVOS = ROOT / "data" / "labels" / "positivos_ids.csv"
SAIDA = ROOT / "data" / "preflight_exp16"
MINIATURA, COLUNAS = (180, 250), 6

def path_imagem(card_id):                              # ajuste o padrão de nome se preciso
    p = IMAGES / f"{card_id}.png"
    return p if p.exists() else None

def contact_sheet(imgs, destino):
    cols = COLUNAS; rows = math.ceil(len(imgs) / cols); cw, ch = MINIATURA
    sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
    for i, im in enumerate(imgs):
        sheet.paste(im.convert("RGB").resize(MINIATURA), ((i % cols) * cw, (i // cols) * ch))
    destino.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destino); print(f"  -> {destino.name}: {len(imgs)} cartas")

def main():
    cat = pd.read_csv(CATALOGO)
    cat["rarity"] = cat["rarity"].fillna("nan").astype(str)
    checar_cobertura(cat)                              # GATE
    pos_ids = set(pd.read_csv(POSITIVOS)["card_id"].astype(str))  # confirme o nome da coluna
    pos = cat[cat["card_id"].astype(str).isin(pos_ids)]
    print(f"Positivos: {len(pos)} (esperado ~311)")
    for rar, grupo in pos.groupby("rarity"):
        imgs = []
        for cid in grupo["card_id"].astype(str):
            p = path_imagem(cid)
            if p:
                with Image.open(p) as im:
                    imgs.append(recortar(im.convert("RGB"), rar))
        if imgs:
            contact_sheet(imgs, SAIDA / f"pos_{rar}_base{LIMITE_INFERIOR[rar]}.png")
    print(f"\nAbra {SAIDA}. Procure positivo SEM cara/olho visível: decapitado por 0.48.")

if __name__ == "__main__":
    main()