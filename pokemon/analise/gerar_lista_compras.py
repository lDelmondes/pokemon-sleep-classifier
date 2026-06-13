"""
Gera a LISTA DE COMPRAS: pontua todas as cartas com imagem usando o modelo de
PRODUCAO (exp16_adaptativo_1.pt — mediana das 3 rodadas do Exp.16), aplicando o
MESMO recorte do treino: RECORTE ADAPTATIVO POR RARIDADE (topo 8.5%, base por
raridade via recorte_rarity.py). Ordena por score. Marca gabarito vs candidata.

CRITICO: o recorte da inferencia tem que ser IDENTICO ao do treino do modelo.
Como o _1 treinou com janela por raridade, aqui tambem cortamos por raridade —
senao o modelo ve, na inferencia, faixas que nunca viu no treino (train-inference
skew), justo onde mora o texto que o recorte adaptativo removeu.

Saidas:
- data/lista_compras.csv  (TODAS as cartas: rank, score, flags)
- data/lista_compras.html (top N recortadas, gabarito em verde / candidata destacada)
"""
from pokemon.caminhos import RAW, IMAGES, LABELS, MODELOS
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import base64
from io import BytesIO

from pokemon.modelagem.modelo_siglip import construir_modelo
from pokemon.modelagem.recorte_rarity import janela_para_rarity, checar_cobertura

MODELO = "exp16_adaptativo_1.pt"
TOP_HTML = 500


def recortar_por_rarity(img, rarity):
    """Recorte IDENTICO ao treino do _1: topo fixo 0.085, base pela raridade."""
    topo, base = janela_para_rarity(rarity)
    w, h = img.size
    return img.crop((0, int(h * topo), w, int(h * base)))


class InferDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(IMAGES / f"{row['card_id']}.png").convert("RGB")
        img = recortar_por_rarity(img, row["rarity"])   # recorte por raridade da carta
        return self.preprocess(img), idx


def recorte_b64(card_id, rarity):
    img = Image.open(IMAGES / f"{card_id}.png").convert("RGB")
    img = recortar_por_rarity(img, rarity)
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, preprocess = construir_modelo(blocos_descongelados=2)
    modelo.load_state_dict(torch.load(MODELOS / MODELO))
    modelo.to(device).eval()
    print(f"Modelo de producao: {MODELO}")

    cat = pd.read_csv(RAW / "catalogo_completo.csv")     # 12k = universo de INFERENCIA (correto aqui)
    cat["rarity"] = cat["rarity"].fillna("nan").astype(str)
    cat = cat[cat["image_url"].notna()].reset_index(drop=True)
    cat = cat[cat["card_id"].apply(lambda c: (IMAGES / f"{c}.png").exists())].reset_index(drop=True)

    checar_cobertura(cat)   # GATE: toda raridade do universo tem janela definida
    print(f"Pontuando {len(cat)} cartas com imagem (recorte adaptativo por raridade)...")

    positivos = set(pd.read_csv(LABELS / "gabarito.csv")["card_id"])

    loader = DataLoader(InferDataset(cat, preprocess), batch_size=32, shuffle=False)
    probs = np.zeros(len(cat))
    with torch.no_grad():
        for imgs, idxs in loader:
            p = torch.sigmoid(modelo(imgs.to(device))).cpu().numpy()
            probs[idxs.numpy()] = p

    cat["score"] = probs
    cat["ja_gabarito"] = cat["card_id"].isin(positivos)
    cat = cat.sort_values("score", ascending=False).reset_index(drop=True)
    cat["rank"] = cat.index + 1

    # CSV completo
    cols = ["rank", "card_id", "nome", "set_codigo", "era", "rarity", "score", "ja_gabarito"]
    cat[cols].to_csv(RAW.parent / "lista_compras.csv", index=False, encoding="utf-8")
    print(f"CSV salvo: lista_compras.csv ({len(cat)} cartas)")

    # HTML: top N
    top = cat.head(TOP_HTML)
    n_gab = int(top["ja_gabarito"].sum())
    n_novas = len(top) - n_gab
    print(f"HTML top {TOP_HTML}: {n_gab} ja-gabarito, {n_novas} candidatas novas")

    cards = ""
    for _, r in top.iterrows():
        b64 = recorte_b64(r["card_id"], r["rarity"])
        classe = "gabarito" if r["ja_gabarito"] else "candidata"
        tag = "JA TENHO" if r["ja_gabarito"] else "CANDIDATA"
        cards += f"""
        <div class="card {classe}">
          <div class="tag">{tag}</div>
          <img src="data:image/png;base64,{b64}"/>
          <div class="info"><b>{r['nome']}</b> <span class="set">{r['set_codigo']} · {r['rarity']}</span><br>
          <code>{r['card_id']}</code><br>
          rank {int(r['rank'])} · score {r['score']:.3f}</div>
        </div>"""

    html = f"""<!doctype html><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#1e1e1e; color:#eee; padding:20px; }}
      h2 {{ position:sticky; top:0; background:#1e1e1e; padding:12px 0; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
      .card {{ background:#2d2d2d; padding:8px; border-radius:8px; width:200px; position:relative; }}
      .card img {{ width:100%; border-radius:4px; }}
      .card.gabarito {{ outline:3px solid #4ec9b0; }}
      .card.candidata {{ outline:3px solid #e0a030; }}
      .tag {{ font-size:10px; font-weight:bold; margin-bottom:4px; }}
      .gabarito .tag {{ color:#4ec9b0; }} .candidata .tag {{ color:#e0a030; }}
      .info {{ font-size:12px; margin-top:6px; }} code {{ color:#888; }}
      .set {{ color:#888; font-size:11px; }}
    </style>
    <h2>LISTA DE COMPRAS — top {TOP_HTML} por score.
    <span style="color:#4ec9b0">verde = ja tenho</span> ·
    <span style="color:#e0a030">laranja = candidata a investigar</span></h2>
    <div class="grid">{cards}</div>"""

    (RAW.parent / "lista_compras.html").write_text(html, encoding="utf-8")
    print(f"HTML salvo: lista_compras.html")


if __name__ == "__main__":
    main()