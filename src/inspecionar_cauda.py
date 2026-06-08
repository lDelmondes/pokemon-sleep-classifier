"""

Carrega o modelo SigLIP fine-tunado, pontua TODAS as cartas, e gera um HTML com os POSITIVOS pior rankeados (os que o modelo erra) para inspecao visual.

Objetivo: entender a cauda dificil e flagrar possiveis erros de rotulo.

"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from modelo_siglip import construir_modelo

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "data" / "labels"
MODELOS = ROOT / "data" / "modelos"

N_PIORES = 15   # quantos dos piores positivos mostrar

class InferDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(IMAGES / f"{row['card_id']}.png").convert("RGB")
        return self.preprocess(img), idx

def main():
    device = "cuda"
    # reconstroi a arquitetura e carrega os pesos treinados
    modelo, preprocess = construir_modelo(blocos_descongelados=2)
    modelo.load_state_dict(torch.load(MODELOS / "melhor_siglip.pt"))
    modelo.to(device).eval()

    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    cat = cat[cat["image_url"].notna()].reset_index(drop=True)
    positivos = set(pd.read_csv(LABELS / "gabarito.csv")["card_id"])

    loader = DataLoader(InferDataset(cat, preprocess), batch_size=32, shuffle=False)
    probs = np.zeros(len(cat))
    with torch.no_grad():
        for imgs, idxs in loader:
            p = torch.sigmoid(modelo(imgs.to(device))).cpu().numpy()
            probs[idxs.numpy()] = p

    cat["prob"] = probs
    cat["eh_positivo"] = cat["card_id"].isin(positivos)

    # ranking global (1 = maior prob)
    cat = cat.sort_values("prob", ascending=False).reset_index(drop=True)
    cat["rank"] = cat.index + 1

    # os positivos pior rankeados
    pos = cat[cat["eh_positivo"]].sort_values("prob").head(N_PIORES)

    cards = ""
    for _, r in pos.iterrows():
        img = (IMAGES / f"{r['card_id']}.png").resolve().as_uri()
        cards += f"""
        <div class="card">
          <img src="{img}"/>
          <div class="info"><b>{r['nome']}</b><br><code>{r['card_id']}</code><br>
          rank {int(r['rank'])}/{len(cat)} &middot; prob {r['prob']:.3f}</div>
        </div>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:16px; }}
      .card {{ background:#2d2d2d; padding:10px; border-radius:8px; width:210px; }}
      .card img {{ width:100%; border-radius:4px; }}
      .info {{ font-size:13px; margin-top:8px; }} code {{ color:#4ec9b0; }}
    </style>
    <h2>Os {N_PIORES} positivos que o modelo MAIS erra &mdash; olho fechado mesmo? ou rotulo errado?</h2>
    <div class="grid">{cards}</div>"""

    saida = LABELS / "cauda_dificil.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Gerado: {saida}")
    print(pos[["rank", "card_id", "nome", "prob"]].to_string(index=False))

if __name__ == "__main__":
    main()