"""

Baixa metadados + URL de imagem das cartas de Pokemon dos sets do MVP via API gratuita da TCGdex (sem API key). Faz cache local de cada carta.

"""
import json, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache_cards"
RAW = ROOT / "data" / "raw"
CACHE.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

API = "https://api.tcgdex.net/v2/en"

def casar_set(nome: str, sid: str = ""):
    """Mapeia set para o codigo de marketplace."""
    if sid == "sv10.5w": return "WHT"
    if sid == "sv03.5":  return "MEW"
    if sid == "svp":     return "SVP"
    return None

def get(url):
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json()

def carta_completa(card_id: str):
    """Objeto completo da carta, com cache em disco."""
    f = CACHE / f"{card_id}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    dados = get(f"{API}/cards/{card_id}")
    f.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.1) 
    return dados

def main():
    todos_sets = get(f"{API}/sets")  # lista resumida de TODOS os sets
    alvos = {s["id"]: casar_set(s["name"], s["id"])
             for s in todos_sets if casar_set(s["name"], s["id"])}

    print("Sets encontrados:")
    for sid, codigo in alvos.items():
        print(f"  {codigo} -> id TCGdex '{sid}'")

    linhas = []
    for sid, codigo in alvos.items():
        cartas = get(f"{API}/sets/{sid}")["cards"]  # resumo das cartas do set
        for resumo in cartas:
            full = carta_completa(resumo["id"])
            if full.get("category") != "Pokemon":   # escopo: so Pokemon
                continue
            base = full.get("image")                 # URL-BASE, sem extensao
            linhas.append({
                "card_id": full["id"],
                "set_codigo": codigo,
                "numero": full.get("localId"),
                "nome": full["name"],
                "image_url": f"{base}/high.png" if base else None,
                "label": 0,   # voce marca 1 nos 19 positivos depois
            })
        n = sum(1 for l in linhas if l["set_codigo"] == codigo)
        print(f"  {codigo}: {n} cartas de Pokemon")

    df = pd.DataFrame(linhas)
    saida = RAW / "catalogo_mvp.csv"
    df.to_csv(saida, index=False, encoding="utf-8")
    print(f"\n{len(df)} cartas salvas em {saida}")

if __name__ == "__main__":
    main()