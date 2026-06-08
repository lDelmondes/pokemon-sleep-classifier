from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
df = pd.read_csv(RAW / "catalogo_mvp.csv", dtype={"numero": str})

sem_img = df[df["image_url"].isna()].copy()
com_img = df[df["image_url"].notna()].copy()

print(f"Total:        {len(df)}")
print(f"Com imagem:   {len(com_img)}  -> vao para o pipeline automatico")
print(f"Sem imagem:   {len(sem_img)}  -> revisao manual")
print(f"\nDistribuicao das sem imagem por set:")
print(sem_img["set_codigo"].value_counts().to_string())

# Salva os dois grupos separados
com_img.to_csv(RAW / "catalogo_com_imagem.csv", index=False, encoding="utf-8")
sem_img.to_csv(RAW / "catalogo_revisao_manual.csv", index=False, encoding="utf-8")

print(f"\n28 cartas para revisao manual:")
print(sem_img[["numero", "nome"]].sort_values("numero").to_string(index=False))