"""

Passa cada imagem pelo CLIP e salva o embedding (vetor de 512 numeros).

Roda UMA vez; o resultado alimenta o treino. Lento em CPU (~10-20 min).

Saida: data/raw/embeddings.npz  (matriz de embeddings + lista de card_id)

"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"

def main():
    device = "cpu"
    print("Carregando CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval().to(device)

    df = pd.read_csv(RAW / "catalogo_completo.csv")
    df = df[df["image_url"].notna()].reset_index(drop=True)
    total = len(df)

    card_ids, vetores = [], []
    for i, row in enumerate(df.itertuples(), start=1):
        caminho = IMAGES / f"{row.card_id}.png"
        try:
            img = preprocess(Image.open(caminho).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb /= emb.norm(dim=-1, keepdim=True)   # normaliza (vetor unitario)
            card_ids.append(row.card_id)
            vetores.append(emb.squeeze(0).cpu().numpy())
        except Exception as e:
            print(f"  ERRO em {row.card_id}: {e}")
        if i % 100 == 0:
            print(f"[{i}/{total}] processadas", flush=True)

    X = np.vstack(vetores).astype(np.float32)
    ids = np.array(card_ids)
    np.savez(RAW / "embeddings.npz", X=X, card_ids=ids)
    print(f"\n{X.shape[0]} embeddings de dimensao {X.shape[1]} salvos em embeddings.npz")

if __name__ == "__main__":
    main()