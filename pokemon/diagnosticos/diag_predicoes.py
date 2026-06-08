from pokemon.caminhos import RAW, LABELS
import pandas as pd


pred = pd.read_csv(RAW / "predicoes.csv")
cat = pd.read_csv(RAW / "catalogo_completo.csv")

# rank de cada carta (1 = maior probabilidade)
pred = pred.sort_values("prob", ascending=False).reset_index(drop=True)
pred["rank"] = pred.index + 1

# so os positivos, com id/nome/set/era, ordenados pelo rank que o modelo deu
pos = pred[pred["y"] == 1].merge(
    cat[["card_id", "set_codigo", "era", "numero", "nome"]],
    on="card_id", how="left")

print(f"Total de cartas: {len(pred)} | posicao dos {len(pos)} positivos:\n")
print(pos.sort_values("rank")[
    ["rank", "card_id", "era", "nome", "prob"]].to_string(index=False))

print("\nRank mediano dos positivos por era:")
print(pos.groupby("era")["rank"].median().to_string())

print("\nRank mediano dos positivos por set:")
print(pos.groupby("set_codigo")["rank"].median().sort_values().to_string())