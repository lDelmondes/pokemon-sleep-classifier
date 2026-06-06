"""

Gera um HTML com os positivos PIOR rankeados pelo modelo (os que ele mais erra), para inspecao visual: entender POR QUE o modelo nao os enxerga.

Le predicoes.csv (precisa ter rodado treinar.py antes).

Saida: data/labels/piores_positivos.html

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "data" / "labels"

N_PIORES = 8   # quantos dos piores positivos mostrar

def main():
    pred = pd.read_csv(RAW / "predicoes.csv")
    cat = pd.read_csv(RAW / "catalogo_completo.csv")

    pred = pred.sort_values("prob", ascending=False).reset_index(drop=True)
    pred["rank"] = pred.index + 1

    # so positivos (y==1), pega os N de pior rank (prob mais baixa)
    pos = pred[pred["y"] == 1].merge(
        cat[["card_id", "era", "nome"]], on="card_id", how="left")
    piores = pos.sort_values("prob").head(N_PIORES)

    cards_html = ""
    for _, r in piores.iterrows():
        img = (IMAGES / f"{r['card_id']}.png").resolve().as_uri()
        cards_html += f"""
        <div class="card">
          <img src="{img}"/>
          <div class="info">
            <b>{r['nome']}</b> ({r['era']})<br>
            <code>{r['card_id']}</code><br>
            rank {int(r['rank'])} de {len(pred)} &middot; prob {r['prob']:.2f}
          </div>
        </div>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:18px; }}
      .card {{ background:#2d2d2d; padding:10px; border-radius:8px; width:230px; }}
      .card img {{ width:100%; border-radius:4px; }}
      .info {{ font-size:13px; margin-top:8px; line-height:1.5; }}
      code {{ color:#4ec9b0; }}
      h1 {{ font-size:18px; }}
    </style>
    <h1>Os {N_PIORES} positivos que o modelo MAIS erra &mdash; por que ele nao ve o olho fechado?</h1>
    <div class="grid">{cards_html}</div>"""

    saida = LABELS / "piores_positivos.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Folha gerada: {saida}")
    print(piores[["rank", "card_id", "nome", "prob"]].to_string(index=False))

if __name__ == "__main__":
    main()