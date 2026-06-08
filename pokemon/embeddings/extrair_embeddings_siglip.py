"""

Semelhante ao extrair_embeddings.py, mas usando SigLIP2 (melhor em detalhes finos) no lugar do CLIP ViT-B-32. 

Saida: data/raw/embeddings_siglip.npz

"""
from pokemon.caminhos import RAW, IMAGES
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

def main():
    device = "cuda"   # agora temos GPU
    print("Carregando SigLIP2 (primeira vez baixa os pesos)...")
    # Sintaxe do SigLIP: create_model_from_pretrained + prefixo hf-hub
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:timm/ViT-B-16-SigLIP2"
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
                emb /= emb.norm(dim=-1, keepdim=True)
            card_ids.append(row.card_id)
            vetores.append(emb.squeeze(0).cpu().numpy())
        except Exception as e:
            print(f"  ERRO em {row.card_id}: {e}")
        if i % 100 == 0:
            print(f"[{i}/{total}] processadas", flush=True)

    X = np.vstack(vetores).astype(np.float32)
    ids = np.array(card_ids)
    np.savez(RAW / "embeddings_siglip.npz", X=X, card_ids=ids)
    print(f"\n{X.shape[0]} embeddings de dimensao {X.shape[1]} salvos.")
    print("Repare na DIMENSAO: o SigLIP pode ter tamanho diferente do CLIP (512).")

if __name__ == "__main__":
    main()