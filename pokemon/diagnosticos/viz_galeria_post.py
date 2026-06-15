"""
Gera a galeria VISUAL para o post de LinkedIn: 12 cartas (grid 3x4) que o modelo
classificou como olho fechado e que NAO estavam no gabarito (achados genuinos),
curadas a mao entre as de score alto.

Diferente da galeria do README (viz_galeria_achados.py), esta:
  - tem selecao FIXA e curada (lista IDS abaixo), nao o top-N automatico;
  - mostra o SCORE (probabilidade) embaixo de cada carta;
  - tem TITULO embutido na propria imagem (viaja junto se repostarem).

Saida: docs/img/galeria_post_linkedin.png
"""
from pokemon.caminhos import RAW, IMAGES, ROOT
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pokemon.modelagem.recorte_rarity import janela_para_rarity

# --- Selecao curada (ordem = ordem de exibicao no grid, 3 colunas x 4 linhas) ---
IDS = [
    "xy0-26", "smp-SM41", "me02.5-165", "sv10.5b-117",
    "swsh10.5-013", "me03-095", "xy10-71", "sv03-081",
    "xy10-66", "g1-58", "sv10.5b-097", "sv04.5-232",
]

GRID_COLS, GRID_ROWS = 3, 4
CARD_W = 340            # largura de cada carta (apos recorte lateral)
PAD = 20               # espaco entre celulas
MARGEM = 28            # margem externa do grid
FAIXA_SCORE = 58       # altura da faixa de score abaixo de cada carta
TITULO_H = 200         # altura da area de titulo no topo (cabe 2 linhas + sub)
FUNDO = (24, 24, 24)   # cinza-escuro (faz as artes saltarem)
TEXTO = (235, 235, 235)
TEXTO_SUB = (160, 160, 160)
CORTE_LATERAL = 0.07   # recorte cosmetico: 7% de cada lado, tira a moldura

T_TITULO = "Cartas que o modelo identificou como Pokémon de olhos fechados"
T_SUB = "número = probabilidade estimada pelo modelo"


def _fonte(tam, bold=False):
    """Carrega DejaVu pelo caminho ABSOLUTO (via matplotlib, que sempre o tem).
    Passar so o nome do .ttf falha em muitos ambientes e cai no default feio,
    pequeno e sem acentos."""
    import matplotlib.font_manager as fm
    nome = "DejaVu Sans"
    try:
        caminho = fm.findfont(fm.FontProperties(family=nome,
                              weight=("bold" if bold else "normal")))
        return ImageFont.truetype(caminho, tam)
    except Exception:
        return ImageFont.load_default()


def recortar(card_id, rarity):
    img = Image.open(IMAGES / f"{card_id}.png").convert("RGB")
    topo, base = janela_para_rarity(rarity)
    w, h = img.size
    esq = int(w * CORTE_LATERAL)
    dir_ = int(w * (1 - CORTE_LATERAL))
    return img.crop((esq, int(h * topo), w - esq, int(h * base)))


def main():
    lista = pd.read_csv(RAW.parent / "lista_compras.csv")
    lista["rarity"] = lista["rarity"].fillna("nan").astype(str)

    faltando = set(IDS) - set(lista["card_id"])
    if faltando:
        raise ValueError(f"IDs ausentes na lista_compras: {sorted(faltando)}")

    sel = lista[lista["card_id"].isin(IDS)].set_index("card_id").loc[IDS].reset_index()

    # recorta e redimensiona todas para CARD_W de largura
    recortes, scores = [], []
    for _, r in sel.iterrows():
        rec = recortar(r["card_id"], r["rarity"])
        prop = CARD_W / rec.width
        rec = rec.resize((CARD_W, int(rec.height * prop)))
        recortes.append(rec)
        scores.append(r["score"])

    # altura da celula = maior carta + faixa de score
    card_h = max(r.height for r in recortes)
    cel_h = card_h + FAIXA_SCORE

    grid_w = GRID_COLS * CARD_W + (GRID_COLS - 1) * PAD + 2 * MARGEM
    grid_h = TITULO_H + GRID_ROWS * cel_h + (GRID_ROWS - 1) * PAD + 2 * MARGEM

    canvas = Image.new("RGB", (grid_w, grid_h), FUNDO)
    draw = ImageDraw.Draw(canvas)

    # --- titulo embutido (quebra em 2 linhas se nao couber) ---
    f_titulo = _fonte(38, bold=True)
    f_sub = _fonte(26)
    larg_util = grid_w - 2 * MARGEM

    def cabe(txt, fonte):
        b = draw.textbbox((0, 0), txt, font=fonte)
        return (b[2] - b[0]) <= larg_util

    titulo_linhas = [T_TITULO]
    if not cabe(T_TITULO, f_titulo):
        # quebra no espaco mais proximo do meio
        palavras = T_TITULO.split()
        meio = len(palavras) // 2
        titulo_linhas = [" ".join(palavras[:meio]), " ".join(palavras[meio:])]

    y = 38
    for ln in titulo_linhas:
        draw.text((MARGEM, y), ln, font=f_titulo, fill=TEXTO)
        y += 46
    draw.text((MARGEM, y + 6), T_SUB, font=f_sub, fill=TEXTO_SUB)

    # --- grid ---
    f_score = _fonte(28, bold=True)
    y0 = TITULO_H + MARGEM
    for i, (rec, sc) in enumerate(zip(recortes, scores)):
        col, row = i % GRID_COLS, i // GRID_COLS
        cel_x = MARGEM + col * (CARD_W + PAD)
        cel_y = y0 + row * (cel_h + PAD)
        # carta centralizada verticalmente na area de imagem da celula
        off_y = cel_y + (card_h - rec.height) // 2
        canvas.paste(rec, (cel_x, off_y))
        # score centralizado na faixa abaixo
        txt = f"{sc:.3f}"
        bbox = draw.textbbox((0, 0), txt, font=f_score)
        tw = bbox[2] - bbox[0]
        tx = cel_x + (CARD_W - tw) // 2
        ty = cel_y + card_h + (FAIXA_SCORE - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((tx, ty), txt, font=f_score, fill=TEXTO)

    saida = ROOT / "docs" / "img" / "galeria_post_linkedin.png"
    canvas.save(saida)
    print(f"Galeria do post salva: {saida} ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()