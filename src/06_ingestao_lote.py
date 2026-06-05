"""

Ingestao generalizada: recebe um dicionario de sets {id_tcgdex: (codigo, era)}, baixa catalogo + imagens de cada um e ANEXA ao catalogo existente.
Reaproveita o cache: cartas/imagens ja baixadas nao sao rebaixadas.

"""
import json, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
CACHE = ROOT / "data" / "cache_cards"
for d in (RAW, IMAGES, CACHE):
    d.mkdir(parents=True, exist_ok=True)

API = "https://api.tcgdex.net/v2/en"

# Sets novos: id_tcgdex -> (codigo_marketplace, era)
SETS_NOVOS = {
    "sv07":  ("SCR", "SV"),
    "sv05":  ("TEF", "SV"),
    "sv02":  ("PAL", "SV"),
    "swsh5": ("BST", "SWSH"),
}

def get(url):
    r = requests.get(url, timeout=(10, 20)); r.raise_for_status()
    return r.json()

def carta_completa(card_id):
    f = CACHE / f"{card_id}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    dados = get(f"{API}/cards/{card_id}")
    f.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.1)
    return dados

def baixar_imagem(url, destino):
    if destino.exists():
        return
    r = requests.get(url, timeout=(10, 20)); r.raise_for_status()
    destino.write_bytes(r.content)
    time.sleep(0.1)

def main():
    linhas = []
    for sid, (codigo, era) in SETS_NOVOS.items():
        print(f"\n=== {codigo} ({sid}, era {era}) ===", flush=True)
        cartas = get(f"{API}/sets/{sid}")["cards"]
        n_poke = 0
        for j, resumo in enumerate(cartas, start=1):
            full = carta_completa(resumo["id"])
            if full.get("category") != "Pokemon":
                continue
            base = full.get("image")
            url_img = f"{base}/high.png" if base else None
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

    novos = pd.DataFrame(linhas)
    saida = RAW / "catalogo_novos.csv"
    novos.to_csv(saida, index=False, encoding="utf-8")
    print(f"\n{len(novos)} cartas dos sets novos salvas em {saida}")
    print(novos["set_codigo"].value_counts().to_string())

if __name__ == "__main__":
    main()