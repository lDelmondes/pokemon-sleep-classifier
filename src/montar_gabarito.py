"""

Constroi o gabarito dos positivos (Pokemon de olhos fechados) a partir da lista (set, numero), resolve o card_id cruzando com o catalogo, e separa quais positivos tem imagem (entram na avaliacao do CLIP) dos que nao tem (revisao manual). 
Saida: data/labels/positivos.csv

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"
LABELS.mkdir(parents=True, exist_ok=True)

# Positivos confirmados: (set_codigo, numero).
POSITIVOS = [
    ("WHT", "141"), ("WHT", "105"), ("WHT", "112"), ("WHT", "091"),
    ("WHT", "096"), ("WHT", "028"), ("WHT", "119"), ("WHT", "036"),
    ("SVP", "014"), ("SVP", "022"), ("SVP", "041"), ("SVP", "051"),
    ("SVP", "122"), ("SVP", "173"), ("SVP", "189"),
    ("MEW", "013"), ("MEW", "063"), ("MEW", "086"), ("MEW", "166"),
]

def main():
    full = pd.read_csv(RAW / "catalogo_mvp.csv", dtype={"numero": str})
    com_img = pd.read_csv(RAW / "catalogo_com_imagem.csv", dtype={"numero": str})
    ids_com_img = set(com_img["card_id"])

    linhas, nao_encontrados = [], []
    for set_cod, numero in POSITIVOS:
        match = full[(full["set_codigo"] == set_cod) & (full["numero"] == numero)]
        if match.empty:
            nao_encontrados.append((set_cod, numero))
            continue
        r = match.iloc[0]
        linhas.append({
            "card_id": r["card_id"],
            "set_codigo": set_cod,
            "numero": numero,
            "nome": r["nome"],
            "tem_imagem": r["card_id"] in ids_com_img,
        })

    gab = pd.DataFrame(linhas)
    gab.to_csv(LABELS / "positivos.csv", index=False, encoding="utf-8")

    n_total = len(gab)
    n_img = int(gab["tem_imagem"].sum())
    print(f"Positivos resolvidos: {n_total}/{len(POSITIVOS)}")
    print(f"  Com imagem (entram na avaliacao do CLIP): {n_img}")
    print(f"  Sem imagem (revisao manual, fora do recall): {n_total - n_img}")
    if nao_encontrados:
        print(f"\n  NAO ENCONTRADOS no catalogo (investigar): {nao_encontrados}")
    print(f"\nGabarito salvo em {LABELS / 'positivos.csv'}")
    print(gab.to_string(index=False))

if __name__ == "__main__":
    main()