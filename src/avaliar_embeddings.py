"""
avaliar_embeddings.py

Treina regressao logistica sobre embeddings (CLIP ou SigLIP) e mede recall/precision honestos com stratified 5-fold. Recebe qual .npz usar via argumento.
Uso:
  python src/avaliar_embeddings.py embeddings.npz          # CLIP
  python src/avaliar_embeddings.py embeddings_siglip.npz   # SigLIP2
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"

def main():
    # qual arquivo de embeddings usar (default: o do SigLIP)
    nome_npz = sys.argv[1] if len(sys.argv) > 1 else "embeddings_siglip.npz"
    print(f"Usando embeddings: {nome_npz}")

    dados = np.load(RAW / nome_npz, allow_pickle=True)
    X, card_ids = dados["X"], dados["card_ids"]
    print(f"Matriz: {X.shape[0]} cartas x {X.shape[1]} dimensoes")

    positivos = set(pd.read_csv(LABELS / "gabarito.csv")["card_id"])
    y = np.array([1 if cid in positivos else 0 for cid in card_ids])
    print(f"{len(y)} cartas | {y.sum()} positivos ({y.mean():.1%})")

    modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]

    ap = average_precision_score(y, probs)
    print(f"\nPR-AUC (average precision): {ap:.3f}")

    ordem = np.argsort(probs)[::-1]
    y_ord = y[ordem]
    P = int(y.sum())
    print(f"\n  K  | recall | precision")
    for K in [20, 30, 44, 60, 80, 100, 150]:
        vp = int(y_ord[:K].sum())
        print(f"  {K:3} |  {vp/P:4.0%}  |   {vp/K:4.0%}")
    ultimo = int(np.where(y_ord == 1)[0].max()) + 1
    print(f"\n  Para recall 100%: revisar as {ultimo} de {len(y)} cartas ({ultimo/len(y):.0%})")

if __name__ == "__main__":
    main()