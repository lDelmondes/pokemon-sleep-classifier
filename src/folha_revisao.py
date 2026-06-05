"""

Gera um HTML com as imagens dos positivos suspeitos (os que o CLIP afundou) lado a lado com seus scores, para revisao humana do rotulo.

Saida: data/labels/revisao_suspeitos.html
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "data" / "labels"

# Os positivos que afundaram nas duas listas (ranks altos = CLIP discorda).
SUSPEITOS = ["sv10.5w-119", "sv10.5w-105", "sv10.5w-112", "sv10.5w-141", "svp-022"]

def main():
    scores = pd.read_csv(RAW / "scores_clip.csv")
    sel = scores[scores["card_id"].isin(SUSPEITOS)].copy()

    cards_html = ""
    for _, r in sel.iterrows():
        # caminho ABSOLUTO da imagem local, para o navegador achar o arquivo
        img_path = (IMAGES / f"{r['card_id']}.png").resolve().as_uri()
        cards_html += f"""
        <div class="card">
          <img src="{img_path}" />
          <div class="info">
            <b>{r['nome']}</b> ({r['set_codigo']} #{r['numero']})<br>
            score olhos_fechados: {r['score_olhos_fechados']:.2f}<br>
            score dormindo: {r['score_dormindo']:.2f}
          </div>
        </div>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#222; color:#eee; padding:20px; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:20px; }}
      .card {{ background:#333; padding:12px; border-radius:8px; width:260px; }}
      .card img {{ width:100%; border-radius:4px; }}
      .info {{ margin-top:8px; font-size:14px; }}
      h1 {{ font-size:18px; }}
    </style>
    <h1>Revisao de rotulos suspeitos &mdash; os olhos estao FECHADOS?</h1>
    <div class="grid">{cards_html}</div>"""

    saida = LABELS / "revisao_suspeitos.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Folha de revisao gerada: {saida}")
    print("Abra esse arquivo no navegador (duplo clique) e revise cada carta.")

if __name__ == "__main__":
    main()