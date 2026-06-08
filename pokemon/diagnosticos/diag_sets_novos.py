from pathlib import Path
import requests

API = "https://api.tcgdex.net/v2/en"
sets = requests.get(f"{API}/sets", timeout=30).json()

alvos = ["scarlet & violet", "surging sparks", "twilight masquerade",
         "crown zenith", "evolving skies", "lost origin", "brilliant stars"]

for s in sets:
    n = s["name"].lower()
    if any(a in n for a in alvos):
        print(f"{s['id']:12} | {s['name']}")