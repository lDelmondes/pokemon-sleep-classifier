"""

Remove IDs especificos do positivos_ids.csv (limpeza de rotulos apos inspecao da cauda). 

Faz backup antes, por seguranca.

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
LABELS = ROOT / "data" / "labels"
arq = LABELS / "positivos_ids.csv"

# IDs a remover (decisao da inspecao da cauda - Exp. 7)
REMOVER = [
    "sv06-099",       # Hisuian Growlithe (olho coberto)
    "sv06-181",       # Hisuian Growlithe (olho coberto)
    "swsh11-083",     # Hisuian Growlithe (olho coberto)
    "sv01-212",       # Kirlia (olho de humano, nao Pokemon)
    "swsh12.5-GG15",  # Solrock (pose, olho nao fechado)
]

df = pd.read_csv(arq)
antes = len(df)
# backup
df.to_csv(LABELS / "positivos_ids_backup.csv", index=False)

df = df[~df["card_id"].isin(REMOVER)]
df.to_csv(arq, index=False)

print(f"Antes: {antes} | Removidos: {antes - len(df)} | Agora: {len(df)}")
nao_achados = set(REMOVER) - set(pd.read_csv(LABELS / 'positivos_ids_backup.csv')['card_id'])
if nao_achados:
    print(f"ATENCAO: IDs que nao estavam no arquivo: {nao_achados}")