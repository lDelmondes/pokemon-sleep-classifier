"""
Gera a galeria visual dos ACHADOS do modelo: as top cartas que o modelo
classificou como olho fechado e que NAO estavam no gabarito (descobertas
genuinas). Grid 5x4 = 20 cartas, recortadas COMO O MODELO AS VE.
Saida: docs/img/galeria_achados.png

Recorte por raridade (mesma fonte do treino/inferencia, recorte_rarity.py).
As alturas variam com a raridade; cada carta e CENTRALIZADA na sua celula.
"""
from pokemon.caminhos import RAW, IMAGES, ROOT
import pandas as pd
from PIL import Image

from pokemon.modelagem.recorte_rarity import janela_para_rarity

GRID_COLS, GRID_ROWS = 5, 4
N = GRID_COLS * GRID_ROWS
CARD_W = 240          # largura de cada carta no grid
PAD = 12              # espaco entre celulas
FUNDO = (30, 30, 30)  # cinza-escuro do fundo

def recortar(card_id, rarity):
    img = Image.open(IMAGES / f"{card_id}.png").convert("RGB")
    topo, base = janela_para_rarity(rarity)
    w, h = img.size
    return img.crop((0, int(h * topo), w, int(h * base)))

def main():
    lista = pd.read_csv(RAW.parent / "lista_compras.csv")
    lista["rarity"] = lista["rarity"].fillna("nan").astype(str)
    achados = lista[~lista["ja_gabarito"]].sort_values("score", ascending=False).head(N)
    print(f"Top {N} achados (nao-gabarito) selecionados.")

    recortes = []
    for _, r in achados.iterrows():
        rec = recortar(r["card_id"], r["rarity"])
        prop = CARD_W / rec.width
        rec = rec.resize((CARD_W, int(rec.height * prop)))
        recortes.append(rec)

    # altura da celula = a maior carta. Cada carta e centralizada dentro dela.
    card_h = max(r.height for r in recortes)
    grid_w = GRID_COLS * CARD_W + (GRID_COLS + 1) * PAD
    grid_h = GRID_ROWS * card_h + (GRID_ROWS + 1) * PAD

    canvas = Image.new("RGB", (grid_w, grid_h), FUNDO)
    for i, rec in enumerate(recortes):
        col, row = i % GRID_COLS, i // GRID_COLS
        cel_x = PAD + col * (CARD_W + PAD)
        cel_y = PAD + row * (card_h + PAD)
        # offset para centralizar: horizontal (todas tem CARD_W, entao 0) e
        # vertical (sobra (card_h - altura) dividida em cima/baixo)
        off_x = cel_x + (CARD_W - rec.width) // 2
        off_y = cel_y + (card_h - rec.height) // 2
        canvas.paste(rec, (off_x, off_y))

    saida = ROOT / "docs" / "img" / "galeria_achados.png"
    canvas.save(saida)
    print(f"Galeria salva: {saida} ({grid_w}x{grid_h})")

if __name__ == "__main__":
    main()