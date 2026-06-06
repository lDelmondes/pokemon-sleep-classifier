"""

Treina regressao logistica sobre os embeddings do CLIP para classificar "olhos fechados".

Avalia com stratified 5-fold + class_weight='balanced'.

Gera probabilidades out-of-fold (cada carta pontuada por um modelo que NAO a viu no treino) e mede recall/precision honestos.

Saida: data/raw/predicoes.csv

"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"

def main():
    # 1. Carrega embeddings (X) e os card_id na mesma ordem
    dados = np.load(RAW / "embeddings.npz", allow_pickle=True)
    X, card_ids = dados["X"], dados["card_ids"]

    # 2. Monta o y (1 = positivo) cruzando com o gabarito
    positivos = set(pd.read_csv(LABELS / "gabarito.csv")["card_id"])
    y = np.array([1 if cid in positivos else 0 for cid in card_ids])
    print(f"{len(y)} cartas | {y.sum()} positivos ({y.mean():.1%})")

    # 3. Modelo + validacao
    modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # cada carta recebe a probabilidade prevista por um modelo que NAO a treinou
    probs = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]

    # 4. Metricas honestas (out-of-fold)
    ap = average_precision_score(y, probs)  # PR-AUC: a metrica certa p/ classe rara
    print(f"\nPR-AUC (average precision): {ap:.3f}")

    # quantos positivos caem no top-K do ranking por probabilidade
    ordem = np.argsort(probs)[::-1]
    y_ord = y[ordem]
    P = y.sum()
    print(f"\n  K  | recall | precision | candidatos p/ revisar")
    for K in [20, 30, 44, 60, 80, 100, 150]:
        vp = y_ord[:K].sum()
        print(f"  {K:3} |  {vp/P:4.0%}  |   {vp/K:4.0%}    | {K}")

    # K minimo para 100% de recall (pegar TODOS os positivos)
    ultimo = np.where(y_ord == 1)[0].max() + 1
    print(f"\n  Para recall 100% (todos os {P}): revisar as {ultimo} cartas melhor rankeadas")
    print(f"  ({ultimo} de {len(y)} = {ultimo/len(y):.0%} do catalogo)")

    # 5. Salva predicoes para inspecao
    out = pd.DataFrame({"card_id": card_ids, "y": y, "prob": probs})
    out.sort_values("prob", ascending=False).to_csv(
        RAW / "predicoes.csv", index=False, encoding="utf-8")
    print(f"\nPredicoes salvas em {RAW / 'predicoes.csv'}")

if __name__ == "__main__":
    main()