import requests

BASE = "https://api.tcgdex.net/v2/en"
ERAS = ["swsh", "sm", "xy", "bw"]

def main():
    for eid in ERAS:
        serie = requests.get(f"{BASE}/series/{eid}", timeout=30).json()
        sets = serie.get("sets", [])
        print(f"\n=== {serie['name']} ({eid}) -- {len(sets)} sets ===")
        for st in sets:
            total = st.get("cardCount", {}).get("total", "?")
            print(f"  {st['id']:<10} {st['name']:<34} total={total}")

if __name__ == "__main__":
    main()