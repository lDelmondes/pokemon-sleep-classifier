"""

Gera um HTML de rotulagem para UM set: todas as cartas com imagem, ordenadas por score_dormindo (desc). Voce marca os positivos no checkbox e o botao gera a lista de card_id para colar no gabarito.
Uso: editar SET_ALVO abaixo e rodar.

"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "data" / "labels"
LABELS.mkdir(parents=True, exist_ok=True)

SET_ALVO = "PAL"   # <--- troque para SCR, TEF, PAL conforme for revisando

def main():
    cat = pd.read_csv(RAW / "catalogo_completo.csv")
    scores = pd.read_csv(RAW / "scores_clip.csv")

    df = cat[(cat["set_codigo"] == SET_ALVO) & (cat["image_url"].notna())].copy()
    df = df.merge(scores[["card_id", "score_dormindo"]], on="card_id", how="left")
    df = df.sort_values("score_dormindo", ascending=False)

    cards_html = ""
    for _, r in df.iterrows():
        img = (IMAGES / f"{r['card_id']}.png").resolve().as_uri()
        score = r["score_dormindo"]
        score_txt = f"{score:.2f}" if pd.notna(score) else "?"
        cards_html += f"""
        <label class="card">
          <input type="checkbox" value="{r['card_id']}">
          <img src="{img}" loading="lazy"/>
          <div class="info">{r['nome']} #{r['numero']}<br>dormindo: {score_txt}</div>
        </label>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
      .card {{ background:#2d2d2d; padding:8px; border-radius:8px; width:180px;
               cursor:pointer; display:block; }}
      .card img {{ width:100%; border-radius:4px; }}
      .card input {{ transform:scale(1.6); margin-bottom:6px; }}
      .card input:checked ~ img {{ outline:3px solid #4ec9b0; }}
      .info {{ font-size:12px; margin-top:6px; }}
      #bar {{ position:sticky; top:0; background:#1e1e1e; padding:12px 0;
              z-index:10; border-bottom:1px solid #444; margin-bottom:14px; }}
      textarea {{ width:100%; height:80px; background:#111; color:#4ec9b0;
                  border:1px solid #444; font-family:monospace; }}
      button {{ font-size:15px; padding:8px 16px; cursor:pointer; }}
    </style>
    <div id="bar">
      <h2>Rotulagem: {SET_ALVO} &mdash; marque os de OLHOS FECHADOS ({len(df)} cartas)</h2>
      <button onclick="gerar()">Gerar lista de positivos</button>
      <span id="contador">0 marcados</span>
      <textarea id="saida" placeholder="A lista de card_id aparece aqui..."></textarea>
    </div>
    <div class="grid">{cards_html}</div>
    <script>
      const boxes = () => [...document.querySelectorAll('input[type=checkbox]')];
      boxes().forEach(b => b.addEventListener('change', () => {{
        document.getElementById('contador').textContent =
          boxes().filter(x => x.checked).length + ' marcados';
      }}));
      function gerar() {{
        const ids = boxes().filter(x => x.checked).map(x => x.value);
        document.getElementById('saida').value = ids.join('\\n');
      }}
    </script>"""

    saida = LABELS / f"rotulagem_{SET_ALVO}.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Folha gerada: {saida}")
    print(f"{len(df)} cartas do set {SET_ALVO}, ordenadas por score_dormindo.")
    print("Abra no navegador, marque os positivos, clique em 'Gerar lista' e me traga o resultado.")

if __name__ == "__main__":
    main()