"""
00_ingestao_catalogo.py
Baixa metadados + URL de imagem das cartas de Pokemon dos sets do MVP
via API gratuita da TCGdex (sem API key). Faz cache local de cada carta.
"""
import json, time
from pathlib import Path
import requests
import pandas as pd

API = "https://api.tcgdex.net/v2/en"
CACHE = Path("data/cache_cards"); CACHE.mkdir(parents=True, exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)

def casar_set(nome: str):
    """Mapeia o nome do set (TCGdex) para o seu codigo de marketplace."""
    n = nome.lower()
    if "white flare" in n: return "WHT"
    if "promos" in n and "scarlet" in n: return "SVP"
    if n == "151": return "MEW"
    return None

def get(url):
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json()

def carta_completa(card_id: str):
    """Objeto completo da carta, com cache em disco (a TCGdex pede que
    voce cacheie em vez de rebaixar -- e fica instantaneo nas proximas vezes)."""
    f = CACHE / f"{card_id}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    dados = get(f"{API}/cards/{card_id}")
    f.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.1)  # cortesia com uma API gratuita
    return dados

def main():
    todos_sets = get(f"{API}/sets")  # lista resumida de TODOS os sets
    alvos = {s["id"]: casar_set(s["name"]) for s in todos_sets if casar_set(s["name"])}

    print("Sets encontrados (CONFIRA se os IDs estao certos):")
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
    df.to_csv("data/raw/catalogo_mvp.csv", index=False, encoding="utf-8")
    print(f"\n{len(df)} cartas salvas em data/raw/catalogo_mvp.csv")

if __name__ == "__main__":
    main()
