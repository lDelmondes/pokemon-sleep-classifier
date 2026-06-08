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
    # --- Scarlet & Violet (SV) ---
    "sv01":     ("SVI", "SV"),   # Scarlet & Violet Base Set  
    "sv02":     ("PAL", "SV"),   # Paldea Envolved
    "sv03.5":   ("MEW", "SV"),   # Pokémon 151
    "sv05":     ("TEF", "SV"),   # Temporal Forces
    "sv06":     ("TWM", "SV"),   # Twilight Masquerade
    "sv07":     ("SCR", "SV"),   # Stellar Crown
    "sv08":     ("SSP", "SV"),   # Surging Sparks
    "sv10.5w":  ("WHT", "SV"),   # White Flare
    "svp":      ("SVP", "SV"),   # Scarlet & Violet Promos
    # --- Sword & Shield (SWSH) ---
    "swsh5":    ("BST", "SWSH"), # Battle Styles
    "swsh7":    ("EVS", "SWSH"), # Evolving Skies
    "swsh9":    ("BRS", "SWSH"), # Brilliant Stars
    "swsh11":   ("LOR", "SWSH"), # Lost Origin
    "swsh12.5": ("CRZ", "SWSH"), # Crown Zenith
    # --- Sun & Moon (SM) ---
    "sm3.5":    ("SLG", "SM"),   # Shining Legends
    "sm10":     ("UNB", "SM"),   # Unbroken Bonds
    "sm11":     ("UNM", "SM"),   # Unified Minds
    "sm115":    ("HIF", "SM"),   # Hidden Fates
    "sma":      ("HFV", "SM"),   # Hidden Fates Shiny Vault (codigo custom; subset do sm115)
    "sm12":     ("CEC", "SM"),   # Cosmic Eclipse
    # --- XY ---
    "xy2":      ("FLF", "XY"),   # Flashfire
    "xy6":      ("ROS", "XY"),   # Roaring Skies
    "xy7":      ("AOR", "XY"),   # Ancient Origins
    "xy8":      ("BKT", "XY"),   # BREAKthrough
    "xy11":     ("STS", "XY"),   # Steam Siege
    # --- Black & White (BW) ---
    "bw3":      ("NVI", "BW"),   # Noble Victories
    "bw4":      ("NXD", "BW"),   # Next Destinies
    "bw7":      ("BCR", "BW"),   # Boundaries Crossed
    "bw9":      ("PLF", "BW"),   # Plasma Freeze
    "bw10":     ("PLB", "BW"),   # Plasma Blast
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