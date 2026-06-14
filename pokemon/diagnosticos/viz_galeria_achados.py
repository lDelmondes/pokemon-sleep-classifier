"""
Gera a galeria visual dos ACHADOS do modelo: cartas que o modelo classificou como
olho fechado e que NAO estavam no gabarito (descobertas genuinas). Grid 5x4.
Saida: docs/img/galeria_achados.png

Esta e a galeria de VITRINE (abertura do README): alem do recorte por raridade
(igual ao modelo ve), aplica um recorte LATERAL cosmetico para remover a moldura
da carta, e permite curadoria manual (trocar/excluir cartas por apresentacao).
NADA disso afeta o modelo ou o pipeline — e so a imagem de vitrine.
"""
from pokemon.caminhos import RAW, IMAGES, ROOT
import pandas as pd
from PIL import Image

from pokemon.modelagem.recorte_rarity import janela_para_rarity

GRID_COLS, GRID_ROWS = 5, 4
N = GRID_COLS * GRID_ROWS
CARD_W = 240            # largura de cada carta no grid (apos recorte lateral)
PAD = 18               # espaco entre celulas (respiro maior na vitrine)
MARGEM = PAD           # margem externa do grid
FUNDO = (30, 30, 30)   # cinza-escuro (faz as artes saltarem)
CORTE_LATERAL = 0.07   # recorte cosmetico: 7% de cada lado, tira a moldura

# --- Curadoria manual da VITRINE (so apresentacao) ---
# Remove cartas com humano visivel / texto ruidoso e poe substitutas escolhidas a mao.
EXCLUIR = {"me02.5-001", "sv08.5-002", "bw8-109"}
INCLUIR = ["sv10.5b-117", "me03-095", "swsh10.5-013"]


def recortar(card_id, rarity):
    img = Image.open(IMAGES / f"{card_id}.png").convert("RGB")
    topo, base = janela_para_rarity(rarity)
    w, h = img.size
    # recorte vertical (como o modelo ve) + recorte lateral cosmetico (so vitrine)
    esq = int(w * CORTE_LATERAL)
    dir_ = int(w * (1 - CORTE_LATERAL))
    return img.crop((esq, int(h * topo), dir_, int(h * base)))


def main():
    lista = pd.read_csv(RAW.parent / "lista_compras.csv")
    lista["rarity"] = lista["rarity"].fillna("nan").astype(str)

    # achados ordenados por score, fora os excluidos
    achados = (lista[~lista["ja_gabarito"]]
               .sort_values("score", ascending=False))
    achados = achados[~achados["card_id"].isin(EXCLUIR)]

    # garante que as cartas curadas (INCLUIR) entrem, no topo da selecao
    incluir_df = lista[lista["card_id"].isin(INCLUIR)].copy()
    faltando = set(INCLUIR) - set(incluir_df["card_id"])
    if faltando:
        raise ValueError(f"IDs de INCLUIR ausentes na lista_compras: {sorted(faltando)}")
    # ordena INCLUIR na ordem que voce pediu
    incluir_df = incluir_df.set_index("card_id").loc[INCLUIR].reset_index()

    # monta a selecao final: curadas + preenche o resto com os melhores achados
    ja = set(INCLUIR)
    resto = achados[~achados["card_id"].isin(ja)].head(N - len(INCLUIR))
    selecao = pd.concat([incluir_df, resto], ignore_index=True).head(N)
    print(f"{len(selecao)} cartas na vitrine ({len(INCLUIR)} curadas + {len(selecao)-len(INCLUIR)} por score).")

    recortes = []
    for _, r in selecao.iterrows():
        rec = recortar(r["card_id"], r["rarity"])
        prop = CARD_W / rec.width
        rec = rec.resize((CARD_W, int(rec.height * prop)))
        recortes.append(rec)

    card_h = max(r.height for r in recortes)
    grid_w = GRID_COLS * CARD_W + (GRID_COLS - 1) * PAD + 2 * MARGEM
    grid_h = GRID_ROWS * card_h + (GRID_ROWS - 1) * PAD + 2 * MARGEM

    canvas = Image.new("RGB", (grid_w, grid_h), FUNDO)
    for i, rec in enumerate(recortes):
        col, row = i % GRID_COLS, i // GRID_COLS
        cel_x = MARGEM + col * (CARD_W + PAD)
        cel_y = MARGEM + row * (card_h + PAD)
        off_x = cel_x + (CARD_W - rec.width) // 2
        off_y = cel_y + (card_h - rec.height) // 2
        canvas.paste(rec, (off_x, off_y))

    saida = ROOT / "docs" / "img" / "galeria_achados.png"
    canvas.save(saida)
    print(f"Galeria salva: {saida} ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()