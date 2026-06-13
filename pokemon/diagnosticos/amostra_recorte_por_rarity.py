"""
Para cada raridade do catalogo completo, pega N cartas e gera uma TIRA
comparativa mostrando o mesmo recorte em VARIAS fracoes de limite inferior
(topo fixo em 8.5%). Permite decidir visualmente a janela ideal por raridade
para o recorte adaptativo.
Saida: data/amostra_recorte/<rarity>/<card_id>_comparativo.png
"""
from pokemon.caminhos import RAW, IMAGES
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

N_POR_RARIDADE = 6
CORTE_TOPO = 0.085
FRACOES_INFERIOR = [0.48, 0.55, 0.60, 0.65]   # janelas candidatas a comparar
ALTURA = 320

def main():
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    cat = cat[cat["image_url"].notna()]
    base = RAW.parent / "amostra_recorte"
    base.mkdir(exist_ok=True)

    print("Raridades no catalogo completo:")
    print(cat["rarity"].value_counts(dropna=False).to_string())
    print(f"\nGerando comparativos ({FRACOES_INFERIOR}) por raridade...\n")

    try:
        fonte = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        fonte = ImageFont.load_default()

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
            orig = Image.open(arq).convert("RGB")
            w, h = orig.size

            paineis = []
            for frac in FRACOES_INFERIOR:
                rec = orig.crop((0, int(h * CORTE_TOPO), w, int(h * frac)))
                prop = ALTURA / rec.height
                rec = rec.resize((int(rec.width * prop), ALTURA))
                paineis.append((frac, rec))

            # monta a tira horizontal com legendas
            larg_total = sum(p.width for _, p in paineis) + 20 * (len(paineis) + 1)
            tira = Image.new("RGB", (larg_total, ALTURA + 40), (30, 30, 30))
            d = ImageDraw.Draw(tira)
            x = 20
            for frac, p in paineis:
                tira.paste(p, (x, 35))
                d.text((x, 8), f"0.085-{frac}", fill=(78, 201, 176), font=fonte)
                x += p.width + 20
            tira.save(pasta / f"{cid}_comparativo.png")

        print(f"  {nome_rar}: {len(amostra)} cartas")

    print(f"\nComparativos em {base}/<raridade>/. "
          f"Abra e decida a janela (limite inferior) ideal de cada raridade.")

if __name__ == "__main__":
    main()