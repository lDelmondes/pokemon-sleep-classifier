from pathlib import Path
import requests

API = "https://api.tcgdex.net/v2/en"
sets = requests.get(f"{API}/sets", timeout=30).json()

# Mostra todo set cujo nome tem "promo"
for s in sets:
    n = s["name"].lower()
    if "promo" in n or "black star" in n:
        print(f"{s['id']:12} | {s['name']}")