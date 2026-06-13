"""
Gera o visual ANTES/DEPOIS do recorte: carta inteira (como o modelo via) ao
lado da arte recortada (8.5%-55%, como passou a ver). Ilustra a descoberta
central — resolucao era o gargalo, o recorte destravou.
Saida: docs/img/antes_depois_recorte.png
"""
from pokemon.caminhos import IMAGES, ROOT
from PIL import Image, ImageDraw, ImageFont

CARD_ID = "swsh10.5-055"        
CORTE_TOPO, LIMITE_INFERIOR = 0.085, 0.48
ALTURA = 500                # altura alvo de cada painel
PAD = 40
COR_FUNDO = (30, 30, 30)
COR_TEXTO = (230, 230, 230)

def main():
    orig = Image.open(IMAGES / f"{CARD_ID}.png").convert("RGB")
    w, h = orig.size

    # ANTES: carta inteira, redimensionada para ALTURA
    prop = ALTURA / orig.height
    antes = orig.resize((int(orig.width * prop), ALTURA))

    # DEPOIS: recorte 8.5%-55%, redimensionado para a mesma ALTURA
    rec = orig.crop((0, int(h * CORTE_TOPO), w, int(h * LIMITE_INFERIOR)))
    prop_r = ALTURA / rec.height
    depois = rec.resize((int(rec.width * prop_r), ALTURA))

    # canvas: antes | seta | depois, com espaco pra legendas em cima
    margem_topo = 50
    largura = antes.width + depois.width + PAD * 3
    altura = ALTURA + margem_topo + PAD
    canvas = Image.new("RGB", (largura, altura), COR_FUNDO)
    d = ImageDraw.Draw(canvas)

    x1 = PAD
    x2 = antes.width + PAD * 2
    canvas.paste(antes, (x1, margem_topo))
    canvas.paste(depois, (x2, margem_topo))

    # legendas
    try:
        fonte = ImageFont.truetype("arial.ttf", 22)
        fonte_seta = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        fonte = ImageFont.load_default()
        fonte_seta = fonte
    d.text((x1, 15), "ANTES: carta inteira (224x224)", fill=COR_TEXTO, font=fonte)
    d.text((x2, 15), "DEPOIS: so a arte", fill=COR_TEXTO, font=fonte)
    # seta entre os paineis
    d.text((antes.width + PAD * 1.3, margem_topo + ALTURA // 2 - 20),
           "->", fill=(78, 201, 176), font=fonte_seta)

    saida = ROOT / "docs" / "img" / "antes_depois_recorte.png"
    canvas.save(saida)
    print(f"Salvo: {saida} ({largura}x{altura})")

if __name__ == "__main__":
    main()