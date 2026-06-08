"""


Lista as series e seus sets na TCGdex para escolher quais ingerir.

Autocontido (so depende de requests).

"""
import requests

BASE = "https://api.tcgdex.net/v2/en"

# Palpite de IDs das series-alvo. CONFIRA contra a lista completa que o
# script imprime primeiro -- se algum estiver errado, corrija aqui.
SERIES_ALVO = ["sm", "xy", "bw"]  # Sun & Moon, XY, Black & White

def listar_series():
    r = requests.get(f"{BASE}/series", timeout=30)
    r.raise_for_status()
    return r.json()

def detalhar_serie(serie_id):
    r = requests.get(f"{BASE}/series/{serie_id}", timeout=30)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    print("=== TODAS AS SERIES (confira os IDs aqui) ===")
    for s in listar_series():
        print(f"  {s['id']:<8} {s['name']}")

    for serie_id in SERIES_ALVO:
        try:
            serie = detalhar_serie(serie_id)
        except requests.HTTPError:
            print(f"\n!! Serie '{serie_id}' nao encontrada -- veja a lista acima.")
            continue
        sets = serie.get("sets", [])
        print(f"\n=== {serie['name']} ({serie_id}) -- {len(sets)} sets ===")
        for st in sets:
            total = st.get("cardCount", {}).get("total", "?")
            print(f"  {st['id']:<10} {st['name']:<34} total={total}")