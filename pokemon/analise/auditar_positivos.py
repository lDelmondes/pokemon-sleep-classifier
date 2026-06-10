"""

Audita o gabarito: pontua todas as cartas com o modelo de 278 e gera um HTML
com TODOS os positivos ordenados do pior para o melhor rank. Os do topo (rank
pior) sao os candidatos a erro de rotulo ou cauda dificil; voce inspeciona
visualmente e decide pela regra refinada (olho VISIVELMENTE fechado de um Pokemon).

O modelo PRIORIZA o que olhar; a decisao de marcar/desmarcar e sua.

Saida: data/labels/auditoria_positivos.html

"""
from pokemon.caminhos import RAW, IMAGES, LABELS, MODELOS
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from pokemon.modelagem.modelo_siglip import construir_modelo

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
    cat = cat.sort_values("prob", ascending=False).reset_index(drop=True)
    cat["rank"] = cat.index + 1

    # TODOS os positivos, ordenados do pior rank (pior = topo da auditoria) ao melhor
    pos = cat[cat["eh_positivo"]].sort_values("prob")  # menor prob primeiro

    cards = ""
    for _, r in pos.iterrows():
        img = (IMAGES / f"{r['card_id']}.png").resolve().as_uri()
        cards += f"""
        <div class="card">
          <img src="{img}"/>
          <div class="info"><b>{r['nome']}</b> <span class="set">{r['set_codigo']}</span><br>
          <code>{r['card_id']}</code><br>
          rank {int(r['rank'])}/{len(cat)} &middot; prob {r['prob']:.3f}</div>
        </div>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      h2 {{ position:sticky; top:0; background:#1e1e1e; padding:12px 0; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:16px; }}
      .card {{ background:#2d2d2d; padding:10px; border-radius:8px; width:200px; }}
      .card img {{ width:100%; border-radius:4px; }}
      .info {{ font-size:13px; margin-top:8px; }} code {{ color:#4ec9b0; }}
      .set {{ color:#888; font-size:11px; }}
    </style>
    <h2>AUDITORIA: {len(pos)} positivos, do que o modelo MAIS rejeita (topo) ao que aceita.
    Olho visivelmente fechado de um Pokemon? Se nao, candidato a remover.</h2>
    <div class="grid">{cards}</div>"""

    saida = LABELS / "auditoria_positivos.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Gerado: {saida} com {len(pos)} positivos ordenados por rank.")
    print("\nOs 20 piores (candidatos a erro de rotulo ou cauda dificil):")
    print(pos.head(20)[["rank", "card_id", "set_codigo", "nome", "prob"]].to_string(index=False))

if __name__ == "__main__":
    main()