"""
Visual 2 do post de LinkedIn: a reviravolta do recorte.

Layout empilhado, em duas metades claramente separadas:
  TOPO  - "o que muda na IMAGEM": a carta inteira -> seta -> so a arte recortada.
          (exemplo do tratamento aplicado a TODAS as cartas)
  BASE  - "o que muda no MODELO": barra de ganho 0.49 -> 0.68, com traducao.

A separacao explicita evita sugerir que ESTA carta gerou ESTE numero: a carta
ilustra o tratamento; o numero e o desempenho agregado do modelo.

Saida: docs/img/recorte_ganho_post.png
"""
from pokemon.caminhos import IMAGES, ROOT
from PIL import Image, ImageDraw, ImageFont

from pokemon.modelagem.recorte_rarity import janela_para_rarity

CARD_EXEMPLO = "me03-095"      # Komala (Illustration rare) - olho bem fechado
RARITY_EXEMPLO = "Illustration rare"

# === COMO MOSTRAR O "ANTES/DEPOIS" ===
# "224"      -> mostra ambos no tamanho que o MODELO ve (224x224). Conta a
#               historia fiel: o olho some quando a imagem inteira e encolhida,
#               e sobrevive quando e so a arte. Menos bonito (imagens pequenas).
# "original" -> mostra ambos grandes, em alta resolucao. Mais bonito, mas o
#               efeito "o olho some" nao aparece (parece so um zoom).
MODO_ANTES = "224"             # troque para "original" para comparar

# --- ganho do modelo (Exp.14, catalogo completo, MEDIA das 3 rodadas) ---
PR_ANTES = 0.49
PR_DEPOIS = 0.68

# --- dimensoes ---
LARGURA = 1080                 # largura do canvas (formato bom p/ feed)
CARD_H = 300                   # altura das cartas no modo "original"
TAM_224 = 300                  # tamanho exibido no modo "224" (a img e 224 mas
                               # ampliamos p/ mostrar; o BORRADO e o ponto)
MARGEM = 50
FUNDO = (24, 24, 24)
TEXTO = (235, 235, 235)
TEXTO_SUB = (160, 160, 160)
ROXO = (140, 90, 220)          # barra "depois"
CINZA_BARRA = (70, 70, 70)     # barra "antes"


def _fonte(tam, bold=False):
    import matplotlib.font_manager as fm
    try:
        caminho = fm.findfont(fm.FontProperties(family="DejaVu Sans",
                              weight=("bold" if bold else "normal")))
        return ImageFont.truetype(caminho, tam)
    except Exception:
        return ImageFont.load_default()


def _centraliza_x(draw, txt, fonte, x0, larg):
    b = draw.textbbox((0, 0), txt, font=fonte)
    return x0 + (larg - (b[2] - b[0])) // 2


def main():
    img = Image.open(IMAGES / f"{CARD_EXEMPLO}.png").convert("RGB")

    # versao recortada (so a arte, via janela por raridade)
    topo, base = janela_para_rarity(RARITY_EXEMPLO)
    w, h = img.size
    rec = img.crop((0, int(h * topo), w, int(h * base)))

    if MODO_ANTES == "original":
        # Simula o que o MODELO ve: tudo vira 224x224 antes de entrar na rede.
        # Reduzimos a 224 e reampliamos para exibir (nearest = mantem o "borrado"
        # visivel, mostrando que o detalhe se perdeu na carta inteira).
        lado = 224
        inteira_224 = img.resize((lado, lado), Image.LANCZOS)
        rec_224 = rec.resize((lado, lado), Image.LANCZOS)
        inteira = inteira_224.resize((TAM_224, TAM_224), Image.NEAREST)
        recortada = rec_224.resize((TAM_224, TAM_224), Image.NEAREST)
    else:  # "original"
        prop = CARD_H / img.height
        inteira = img.resize((int(img.width * prop), CARD_H), Image.LANCZOS)
        prop_r = CARD_H / rec.height
        recortada = rec.resize((int(rec.width * prop_r), CARD_H), Image.LANCZOS)

    card_disp_h = inteira.height   # altura real exibida (varia com o modo)

    # fontes
    f_secao = _fonte(34, bold=True)
    f_card = _fonte(24, bold=True)
    f_leg = _fonte(22)
    f_num = _fonte(46, bold=True)
    f_barra_lbl = _fonte(24, bold=True)

    # ---- monta canvas com altura calculada (usa card_disp_h, que varia c/ modo) ----
    H_TOPO_TIT = 80
    H_CARD_LBL = 36
    H_LEG = 40
    H_DIV = 60
    H_BASE_TIT = 50
    # cada barra agora tem rotulo ACIMA: rotulo(34) + barra(54) + espaco
    barra_h = 54
    H_ROTULO = 34
    H_BARRAS = 2 * (H_ROTULO + barra_h) + 30 + 24
    H_TRAD = 80   # duas linhas
    altura = (MARGEM + H_TOPO_TIT + H_CARD_LBL + card_disp_h + 16 + H_LEG
              + H_DIV + H_BASE_TIT + H_BARRAS + H_TRAD + MARGEM)

    canvas = Image.new("RGB", (LARGURA, altura), FUNDO)
    draw = ImageDraw.Draw(canvas)
    y = MARGEM

    # ===== METADE 1: a imagem =====
    titulo_topo = "O recorte: carta completa para só a arte"
    draw.text((_centraliza_x(draw, titulo_topo, f_secao, 0, LARGURA), y),
              titulo_topo, font=f_secao, fill=TEXTO)
    y += H_TOPO_TIT

    seta_w = 90
    gap = 30
    total_w = inteira.width + gap + seta_w + gap + recortada.width
    x0 = (LARGURA - total_w) // 2

    # rotulos sobre cada imagem
    draw.text((_centraliza_x(draw, "ANTES", f_card, x0, inteira.width), y),
              "ANTES", font=f_card, fill=TEXTO_SUB)
    x_rec = x0 + inteira.width + gap + seta_w + gap
    draw.text((_centraliza_x(draw, "DEPOIS", f_card, x_rec, recortada.width), y),
              "DEPOIS", font=f_card, fill=TEXTO_SUB)
    y += H_CARD_LBL

    canvas.paste(inteira, (x0, y))
    # seta (centralizada na altura das imagens)
    sy = y + card_disp_h // 2
    sx0 = x0 + inteira.width + gap
    draw.line((sx0, sy, sx0 + seta_w - 18, sy), fill=TEXTO, width=5)
    draw.polygon([(sx0 + seta_w - 18, sy - 14), (sx0 + seta_w, sy),
                  (sx0 + seta_w - 18, sy + 14)], fill=TEXTO)
    canvas.paste(recortada, (x_rec, y))
    y += card_disp_h + 16

    legenda = "exemplo do tratamento aplicado a todas as cartas antes de irem ao modelo"
    draw.text((_centraliza_x(draw, legenda, f_leg, 0, LARGURA), y),
              legenda, font=f_leg, fill=TEXTO_SUB)
    y += H_LEG

    # ===== DIVISOR =====
    y += H_DIV // 2
    draw.line((MARGEM, y, LARGURA - MARGEM, y), fill=(60, 60, 60), width=2)
    y += H_DIV // 2

    # ===== METADE 2: o ganho do modelo =====
    titulo_base = "O que isso mudou no modelo"
    draw.text((_centraliza_x(draw, titulo_base, f_secao, 0, LARGURA), y),
              titulo_base, font=f_secao, fill=TEXTO)
    y += H_BASE_TIT

    # barras com rotulo ACIMA (evita colisao com o texto)
    barra_x = MARGEM
    barra_max_w = LARGURA - 2 * MARGEM - 130   # deixa espaco p/ o numero a direita

    def barra(yb, label, valor, cor):
        draw.text((barra_x, yb), label, font=f_barra_lbl, fill=TEXTO_SUB)
        yb2 = yb + H_ROTULO
        draw.rounded_rectangle((barra_x, yb2, barra_x + barra_max_w, yb2 + barra_h),
                               radius=8, fill=(45, 45, 45))
        w_val = int(barra_max_w * valor)
        draw.rounded_rectangle((barra_x, yb2, barra_x + w_val, yb2 + barra_h),
                               radius=8, fill=cor)
        draw.text((barra_x + barra_max_w + 22, yb2 + (barra_h - 50) // 2),
                  f"{valor:.2f}", font=f_num, fill=TEXTO)

    barra(y, "antes do recorte", PR_ANTES, CINZA_BARRA)
    y += H_ROTULO + barra_h + 30
    barra(y, "depois do recorte", PR_DEPOIS, ROXO)
    y += H_ROTULO + barra_h + 24

    # traducao do numero, quebrada em 2 linhas para nao estourar a largura
    trad1 = "capacidade do modelo de ranquear corretamente as cartas (PR-AUC),"
    trad2 = "medida no conjunto inteiro"
    draw.text((_centraliza_x(draw, trad1, f_leg, 0, LARGURA), y),
              trad1, font=f_leg, fill=TEXTO_SUB)
    draw.text((_centraliza_x(draw, trad2, f_leg, 0, LARGURA), y + 32),
              trad2, font=f_leg, fill=TEXTO_SUB)

    saida = ROOT / "docs" / "img" / "recorte_ganho_post.png"
    canvas.save(saida)
    print(f"Visual 2 salvo: {saida} ({LARGURA}x{altura})")


if __name__ == "__main__":
    main()