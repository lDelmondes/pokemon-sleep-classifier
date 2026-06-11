"""
Mostra os positivos do gabarito JA RECORTADOS (55% superior), para revisao
visual: o Pokemon de olho fechado sobreviveu ao recorte? Se NAO (o corte
removeu o bichinho dormindo), marque para REMOVER do gabarito — o modelo
recortado nao vera esse olho, entao manter como positivo seria ruido.

Marcar = candidato a REMOVER. Gera a lista de card_id a tirar do gabarito.
Saida: data/labels/revisao_positivos_recortados.html
"""
from pokemon.caminhos import RAW, IMAGES, LABELS
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

FRACAO = 0.55

def img_recortada_b64(card_id):
    """Recorta a carta e devolve como data-URI base64 (pra embutir no HTML,
    sem depender de arquivos soltos no disco)."""
    img = Image.open(IMAGES / f"{card_id}.png").convert("RGB")
    w, h = img.size
    img = img.crop((0, 0, w, int(h * FRACAO)))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def main():
    positivos = pd.read_csv(LABELS / "gabarito.csv")["card_id"].tolist()
    cat = pd.read_csv(RAW / "catalogo_completo.csv")[["card_id", "nome", "set_codigo", "rarity"]]
    info = cat.set_index("card_id")

    cards = ""
    faltando = 0
    for cid in positivos:
        arq = IMAGES / f"{cid}.png"
        if not arq.exists():
            faltando += 1
            continue
        b64 = img_recortada_b64(cid)
        nome = info.loc[cid, "nome"] if cid in info.index else "?"
        rar = info.loc[cid, "rarity"] if cid in info.index else "?"
        cards += f"""
        <label class="card">
          <input type="checkbox" value="{cid}">
          <img src="data:image/png;base64,{b64}"/>
          <div class="info"><b>{nome}</b> <span class="set">{rar}</span><br>
          <code>{cid}</code></div>
        </label>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      #bar {{ position:sticky; top:0; background:#1e1e1e; padding:12px 0; z-index:10; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
      .card {{ background:#2d2d2d; padding:8px; border-radius:8px; width:200px; cursor:pointer; display:block; }}
      .card img {{ width:100%; border-radius:4px; }}
      .card input {{ transform:scale(1.5); margin-bottom:6px; }}
      .card input:checked ~ img {{ outline:3px solid #ff5555; opacity:0.5; }}
      .info {{ font-size:12px; margin-top:6px; }} code {{ color:#4ec9b0; }}
      .set {{ color:#888; font-size:11px; }}
      textarea {{ width:100%; height:80px; background:#111; color:#ff5555; border:1px solid #444; font-family:monospace; }}
    </style>
    <div id="bar">
      <h2>REVISAO POS-RECORTE: o Pokemon de olho fechado SOBREVIVEU ao corte?
      Marque os que PERDERAM o bichinho (a remover do gabarito).</h2>
      <button onclick="gerar()">Gerar lista de REMOCAO</button>
      <span id="c">0 marcados para remover</span>
      <textarea id="saida" placeholder="card_id a remover do gabarito..."></textarea>
    </div>
    <div class="grid">{cards}</div>
    <script>
      const bs = () => [...document.querySelectorAll('input[type=checkbox]')];
      bs().forEach(b => b.addEventListener('change', () => {{
        document.getElementById('c').textContent = bs().filter(x=>x.checked).length + ' marcados para remover';
      }}));
      function gerar() {{
        document.getElementById('saida').value = bs().filter(x=>x.checked).map(x=>x.value).join('\\n');
      }}
    </script>"""

    saida = LABELS / "revisao_positivos_recortados.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Gerado: {saida} com {len(positivos)-faltando} positivos recortados.")
    if faltando:
        print(f"  ({faltando} sem imagem no disco, pulados)")

if __name__ == "__main__":
    main()