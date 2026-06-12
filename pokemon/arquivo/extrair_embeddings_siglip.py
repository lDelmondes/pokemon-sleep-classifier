"""

Extrai embeddings SigLIP2 (ViT-B-16-SigLIP2) para o catalogo.

Funciona de maneira incrementar: reaproveita os embeddings ja existentes em embeddings_siglip.npz e processa apenas os card_id novos.

Saida: data/raw/embeddings_siglip.npz  (X: matriz N x 768, card_ids: N rotulos)

"""
from pokemon.caminhos import RAW, IMAGES
import sys
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image

NPZ = RAW / "embeddings_siglip.npz"

def carregar_existentes(forcar_tudo=False):
    """Devolve (dict card_id->vetor) do que ja foi processado. Vazio se
    nao existe .npz ou se forcar_tudo=True."""
    if forcar_tudo or not NPZ.exists():
        return {}
    dados = np.load(NPZ, allow_pickle=True)
    return dict(zip(dados["card_ids"], dados["X"]))

def main():
    forcar_tudo = "--tudo" in sys.argv
    device = "cuda"

    existentes = carregar_existentes(forcar_tudo)
    print(f"Embeddings ja em cache: {len(existentes)}"
          + (" (ignorado: --tudo)" if forcar_tudo else ""))

    df = pd.read_csv(RAW / "catalogo_completo.csv")
    df = df[df["image_url"].notna()].reset_index(drop=True)

    # quais faltam processar
    faltam = df[~df["card_id"].isin(existentes.keys())].reset_index(drop=True)
    total = len(faltam)
    print(f"Catalogo com imagem: {len(df)} | a processar agora: {total}")

    if total == 0:
        print("Nada novo a processar. Embeddings ja cobrem todo o catalogo.")
        return

    print("Carregando SigLIP2 (primeira vez baixa os pesos)...")
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:timm/ViT-B-16-SigLIP2"
    )
    model.eval().to(device)

    novos = dict(existentes)  # copia; vamos adicionar os novos aqui
    for i, row in enumerate(faltam.itertuples(), start=1):
        caminho = IMAGES / f"{row.card_id}.png"
        try:
            img = preprocess(Image.open(caminho).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb /= emb.norm(dim=-1, keepdim=True)
            novos[row.card_id] = emb.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"  ERRO em {row.card_id}: {e}")
        if i % 100 == 0:
            print(f"[{i}/{total}] processadas", flush=True)

    # Reconstroi X e card_ids ALINHADOS, na ordem do catalogo (estavel e auditavel)
    ordem = [cid for cid in df["card_id"] if cid in novos]
    X = np.vstack([novos[cid] for cid in ordem]).astype(np.float32)
    ids = np.array(ordem)

    # Verificacao de sanidade: X e ids tem que ter o mesmo tamanho
    assert X.shape[0] == len(ids), "Desalinhamento entre X e card_ids!"

    np.savez(NPZ, X=X, card_ids=ids)
    print(f"\n{X.shape[0]} embeddings de dimensao {X.shape[1]} salvos "
          f"({total} novos, {len(existentes)} reaproveitados).")

if __name__ == "__main__":
    main()