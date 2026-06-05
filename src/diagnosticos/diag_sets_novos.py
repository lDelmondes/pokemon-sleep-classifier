from pathlib import Path
import requests

API = "https://api.tcgdex.net/v2/en"
sets = requests.get(f"{API}/sets", timeout=30).json()

alvos = ["stellar crown", "temporal forces", "paldea evolved", "battle styles"]

for s in sets:
    n = s["name"].lower()
    if any(a in n for a in alvos):
        print(f"{s['id']:12} | {s['name']}")