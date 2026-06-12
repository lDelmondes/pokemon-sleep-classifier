"""
Gera a galeria visual dos ACHADOS do modelo: as top cartas que o modelo
classificou como olho fechado e que NAO estavam no gabarito (descobertas
genuinas). Grid 5x4 = 20 cartas, recortadas como o modelo as ve.
Saida: docs/img/galeria_achados.png
"""
from pokemon.caminhos import RAW, IMAGES, LABELS, ROOT
import pandas as pd
from PIL import Image

GRID_COLS, GRID_ROWS = 5, 4
N = GRID_COLS * GRID_ROWS
CORTE_TOPO, LIMITE_INFERIOR = 0.085, 0.55
CARD_W = 240   # largura de cada carta no grid
PAD = 12       # espaco entre cartas

def recortar(card_id):
    img = Image.open(IMAGES / f"{card_id}.png").convert("RGB")
    w, h = img.size
    return img.crop((0, int(h * CORTE_TOPO), w, int(h * LIMITE_INFERIOR)))

def main():
    # le a lista de compras (ja pontuada) e pega os achados: score alto + NAO gabarito
    lista = pd.read_csv(RAW.parent / "lista_compras.csv")
    achados = lista[~lista["ja_gabarito"]].sort_values("score", ascending=False).head(N)
    print(f"Top {N} achados (nao-gabarito) selecionados.")

    # recorta todas e descobre a altura (todas tem ~mesma proporcao apos recorte)
    recortes = []
    for _, r in achados.iterrows():
        rec = recortar(r["card_id"])
        # redimensiona para largura fixa, mantendo proporcao
        prop = CARD_W / rec.width
        rec = rec.resize((CARD_W, int(rec.height * prop)))
        recortes.append(rec)

    card_h = max(r.height for r in recortes)
    grid_w = GRID_COLS * CARD_W + (GRID_COLS + 1) * PAD
    grid_h = GRID_ROWS * card_h + (GRID_ROWS + 1) * PAD

    canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    for i, rec in enumerate(recortes):
        col, row = i % GRID_COLS, i // GRID_COLS
        x = PAD + col * (CARD_W + PAD)
        y = PAD + row * (card_h + PAD)
        canvas.paste(rec, (x, y))

    saida = ROOT / "docs" / "img" / "galeria_achados.png"
    canvas.save(saida)
    print(f"Galeria salva: {saida} ({grid_w}x{grid_h})")

if __name__ == "__main__":
    main()