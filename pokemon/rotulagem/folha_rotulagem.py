"""

Gera folhas HTML de rotulagem (uma por set), ordenadas por score_dormindo desc.

Marca os positivos no checkbox e o botao gera a lista de card_id para colar.

Edite SETS_PARA_ROTULAR e rode uma vez para gerar todas.

"""
from pokemon.caminhos import RAW, IMAGES, LABELS
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Sets novos a rotular nesta rodada
SETS_PARA_ROTULAR = ["SVI", "TWM", "SSP", "EVS", "BRS", "LOR", "CRZ"]

def calcular_scores():
    """Treina a logistica SigLIP em cima do que JA temos rotulado e gera
    uma probabilidade out-of-fold para cada carta, para ordenar a revisao.
    Cartas dos sets novos (ainda sem rotulo) sao pontuadas pelo modelo."""
    dados = np.load(RAW / "embeddings_siglip.npz", allow_pickle=True)
    X, card_ids = dados["X"], dados["card_ids"]
    positivos = set(pd.read_csv(LABELS / "gabarito.csv")["card_id"])
    y = np.array([1 if c in positivos else 0 for c in card_ids])

    modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    return dict(zip(card_ids, probs))

def gerar_folha(set_codigo, cat, scores):
    df = cat[(cat["set_codigo"] == set_codigo) & (cat["image_url"].notna())].copy()
    df["score"] = df["card_id"].map(scores).fillna(0.0)
    df = df.sort_values("score", ascending=False)

    cards = ""
    for _, r in df.iterrows():
        img = (IMAGES / f"{r['card_id']}.png").resolve().as_uri()
        cards += f"""
        <label class="card">
          <input type="checkbox" value="{r['card_id']}">
          <img src="{img}" loading="lazy"/>
          <div class="info">{r['nome']} #{r['numero']}<br>score {r['score']:.2f}</div>
        </label>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
      .card {{ background:#2d2d2d; padding:8px; border-radius:8px; width:170px; cursor:pointer; display:block; }}
      .card img {{ width:100%; border-radius:4px; }}
      .card input {{ transform:scale(1.5); margin-bottom:6px; }}
      .card input:checked ~ img {{ outline:3px solid #4ec9b0; }}
      .info {{ font-size:12px; margin-top:6px; }}
      #bar {{ position:sticky; top:0; background:#1e1e1e; padding:12px 0; z-index:10; border-bottom:1px solid #444; margin-bottom:14px; }}
      textarea {{ width:100%; height:70px; background:#111; color:#4ec9b0; border:1px solid #444; font-family:monospace; }}
      button {{ font-size:15px; padding:8px 16px; cursor:pointer; }}
    </style>
    <div id="bar">
      <h2>Rotulagem: {set_codigo} &mdash; marque os de OLHOS FECHADOS ({len(df)} cartas)</h2>
      <button onclick="gerar()">Gerar lista</button>
      <span id="c">0 marcados</span>
      <textarea id="saida" placeholder="card_id dos positivos aparece aqui..."></textarea>
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

    saida = LABELS / f"rotulagem_{set_codigo}.html"
    saida.write_text(html, encoding="utf-8")
    return len(df)

def main():
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    print("Calculando scores para ordenar a revisao...")
    scores = calcular_scores()
    for sc in SETS_PARA_ROTULAR:
        n = gerar_folha(sc, cat, scores)
        print(f"  rotulagem_{sc}.html  ({n} cartas)")
    print(f"\nFolhas geradas em {LABELS}. Abra cada uma no navegador.")

if __name__ == "__main__":
    main()