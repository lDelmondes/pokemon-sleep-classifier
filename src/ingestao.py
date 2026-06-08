"""

Ingestao generalizada de varios sets. Usa tcgdex_utils para falar com a API.

"""
from pathlib import Path
import pandas as pd
from tcgdex_utils import cartas_do_set, carta_completa, montar_url_imagem, baixar_imagem

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
RAW.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

SETS = {
    "sv10.5w": ("WHT", "SV"),
    "svp":     ("SVP", "SV"),
    "sv03.5":  ("MEW", "SV"),
    "sv07":    ("SCR", "SV"),
    "sv05":    ("TEF", "SV"),
    "sv02":    ("PAL", "SV"),
    "swsh5":   ("BST", "SWSH"),
    "sv01":     ("SVI", "SV"),
    "sv06":     ("TWM", "SV"),
    "sv08":     ("SSP", "SV"),
    "swsh7":    ("EVS", "SWSH"),
    "swsh9":    ("BRS", "SWSH"),
    "swsh11":   ("LOR", "SWSH"),
    "swsh12.5": ("CRZ", "SWSH"),
}

def main():
    linhas = []
    for sid, (codigo, era) in SETS.items():
        print(f"\n=== {codigo} ({sid}, era {era}) ===", flush=True)
        cartas = cartas_do_set(sid)
        n_poke = 0
        for j, resumo in enumerate(cartas, start=1):
            full = carta_completa(resumo["id"])
            if full.get("category") != "Pokemon":
                continue
            base = full.get("image")
            url_img = montar_url_imagem(full)
            destino = IMAGES / f"{full['id']}.png"
            if url_img:
                try:
                    baixar_imagem(url_img, destino)
                except Exception as e:
                    print(f"  ERRO img {full['id']}: {type(e).__name__}", flush=True)
            linhas.append({
                "card_id": full["id"],
                "set_codigo": codigo,
                "era": era,
                "numero": full.get("localId"),
                "nome": full["name"],
                "image_url": url_img,
                "label": 0,
            })
            n_poke += 1
            if j % 50 == 0:
                print(f"  [{j}/{len(cartas)}] processadas", flush=True)
        print(f"  {codigo}: {n_poke} cartas de Pokemon")

    df = pd.DataFrame(linhas)
    saida = RAW / "catalogo_completo.csv"
    df.to_csv(saida, index=False, encoding="utf-8")
    print(f"\n{len(df)} cartas de Pokemon salvas em {saida}")
    print(df.groupby(["era", "set_codigo"]).size().to_string())

if __name__ == "__main__":
    main()