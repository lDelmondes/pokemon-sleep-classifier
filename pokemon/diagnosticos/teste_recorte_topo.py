"""
Testa o recorte COMBINADO (corta faixa de nome/HP no topo + mantem so a arte)
em tres fracoes de corte superior, para escolher a que remove o texto sem
decepar o rosto do Pokemon. Mantem o limite inferior em 55%.
Saida: data/teste_corte_topo/<corte>/<card_id>.png
"""
from pokemon.caminhos import RAW, IMAGES
import pandas as pd
from PIL import Image

# cartas de eras diferentes p/ ver variacao do tamanho da faixa de nome
CARTAS = ["sm6-89", "xy6-38", "bw11-94", "sv06-030", "swsh8-180"]
CORTES_TOPO = [0.085, 0.09, 0.10, 0.12]
LIMITE_INFERIOR = 0.55

def main():
    base = RAW.parent / "teste_corte_topo"
    base.mkdir(exist_ok=True)
    for corte in CORTES_TOPO:
        pasta = base / f"topo_{int(corte*100)}pct"
        pasta.mkdir(exist_ok=True)
        for cid in CARTAS:
            arq = IMAGES / f"{cid}.png"
            if not arq.exists():
                continue
            img = Image.open(arq).convert("RGB")
            w, h = img.size
            rec = img.crop((0, int(h * corte), w, int(h * LIMITE_INFERIOR)))
            rec.save(pasta / f"{cid}.png")
        print(f"Corte topo {int(corte*100)}%: salvo em {pasta}")

if __name__ == "__main__":
    main()