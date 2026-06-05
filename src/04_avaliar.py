"""

Cruza os scores do CLIP com o gabarito e mede recall/precision de cada prompt.
Logica do produto: humano-no-loop, alvo = nao perder positivo (recall manda). Por isso medimos "quantos positivos estao no top-K candidatos" para varios K.

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"

def main():
    scores = pd.read_csv(RAW / "scores_clip.csv")
    gab = pd.read_csv(LABELS / "positivos.csv")
    positivos = set(gab["card_id"])
    P = len(positivos)

    scores["eh_positivo"] = scores["card_id"].isin(positivos)

    testes = [c.replace("score_", "") for c in scores.columns if c.startswith("score_")]
    Ks = [10, 20, 30, 50, 100]

    for teste in testes:
        col = f"score_{teste}"
        ordenado = scores.sort_values(col, ascending=False).reset_index(drop=True)
        # posicao (rank) de cada positivo na lista ordenada por este score
        ranks = ordenado.index[ordenado["eh_positivo"]].tolist()

        print(f"\n===== PROMPT: {teste} =====")
        print(f"Posicao dos {P} positivos no ranking (1 = topo):")
        achados = ordenado[ordenado["eh_positivo"]].copy()
        achados["rank"] = [r + 1 for r in ranks]
        print(achados[["rank", "set_codigo", "numero", "nome", col]]
              .sort_values("rank").to_string(index=False))

        print(f"\n  K  | recall | precision | candidatos p/ revisar")
        for K in Ks:
            topK = ordenado.head(K)
            vp = int(topK["eh_positivo"].sum())   # positivos capturados no top-K
            recall = vp / P
            precision = vp / K
            print(f"  {K:3} |  {recall:4.0%}  |   {precision:4.0%}    | {K}")

if __name__ == "__main__":
    main()