"""

Espelho da auditoria de positivos: pontua todas as cartas e gera um HTML com os
NEGATIVOS de MAIOR score — cartas que voce NAO marcou como positivas mas o modelo
acha que sao. Candidatas a positivos que escaparam da varredura (falsos negativos
no gabarito). Voce inspeciona e marca os que forem olho fechado de verdade.

Saida: data/labels/auditoria_negativos.html

"""
from pokemon.caminhos import RAW, IMAGES, LABELS, MODELOS
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from pokemon.modelagem.modelo_siglip import construir_modelo

SCORE_MINIMO = 0.5   # revisa todos os negativos com score >= este valor

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

    # so os NEGATIVOS, ordenados por maior score (mais suspeitos primeiro)
    neg = cat[~cat["eh_positivo"]]
    neg = neg[neg["prob"] >= SCORE_MINIMO].sort_values("prob", ascending=False)
    print(f"Negativos com score >= {SCORE_MINIMO}: {len(neg)} cartas a revisar")

    cards = ""
    for _, r in neg.iterrows():
        img = (IMAGES / f"{r['card_id']}.png").resolve().as_uri()
        cards += f"""
        <label class="card">
          <input type="checkbox" value="{r['card_id']}">
          <img src="{img}"/>
          <div class="info"><b>{r['nome']}</b> <span class="set">{r['set_codigo']}</span><br>
          <code>{r['card_id']}</code><br>prob {r['prob']:.3f}</div>
        </label>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      #bar {{ position:sticky; top:0; background:#1e1e1e; padding:12px 0; z-index:10; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
      .card {{ background:#2d2d2d; padding:8px; border-radius:8px; width:180px; cursor:pointer; display:block; }}
      .card img {{ width:100%; border-radius:4px; }}
      .card input {{ transform:scale(1.5); margin-bottom:6px; }}
      .card input:checked ~ img {{ outline:3px solid #4ec9b0; }}
      .info {{ font-size:12px; margin-top:6px; }} code {{ color:#4ec9b0; }}
      .set {{ color:#888; font-size:11px; }}
      textarea {{ width:100%; height:70px; background:#111; color:#4ec9b0; border:1px solid #444; font-family:monospace; }}
    </style>
    <div id="bar">
      <h2>NEGATIVOS de maior score &mdash; o modelo acha que tem olho fechado. Marque os que VOCE concorda.</h2>
      <button onclick="gerar()">Gerar lista</button>
      <span id="c">0 marcados</span>
      <textarea id="saida" placeholder="card_id dos que voce confirma como positivo..."></textarea>
    </div>
    <div class="grid">{cards}</div>
    <script>
      const bs = () => [...document.querySelectorAll('input[type=checkbox]')];
      bs().forEach(b => b.addEventListener('change', () => {{
        document.getElementById('c').textContent = bs().filter(x=>x.checked).length + ' marcados';
      }}));
      function gerar() {{
        document.getElementById('saida').value = bs().filter(x=>x.checked).map(x=>x.value).join('\\n');
      }}
    </script>"""

    saida = LABELS / "auditoria_negativos.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Gerado: {saida} com os {len(neg)} negativos de maior score.")

if __name__ == "__main__":
    main()