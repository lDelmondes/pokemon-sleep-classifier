"""

Lista positivos do gabarito por nome, para revisar a aplicacao consistente
das regras (ex: todos os Hisuian Growlithe, todos de um Pokemon de olho coberto).

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
gab = pd.read_csv(ROOT / "data" / "labels" / "gabarito.csv")

# nomes a investigar para aplicar regra consistente
suspeitos = ["growlithe", "kirlia", "solrock"]
for s in suspeitos:
    achados = gab[gab["nome"].str.lower().str.contains(s)]
    print(f"\n=== '{s}': {len(achados)} no gabarito ===")
    print(achados[["card_id", "nome"]].to_string(index=False))

# tambem mostra o gabarito inteiro ordenado por nome, para voce escanear
print("\n=== GABARITO COMPLETO (por nome) ===")
print(gab.sort_values("nome")[["card_id", "set_codigo", "nome"]].to_string(index=False))