"""[Visuais 3/4 e 4/4 do README] Avaliacao do modelo de producao no conjunto de teste.
Gera dois PNGs em docs/img/:
  - curva_pr_producao.png   : precision-recall do _1 (producao) vs ancora 0.55 (a EQUIVALENCIA).
  - trabalho_economizado.png: recall vs cartas revisadas (argumento de PRODUTO, humano-no-loop).
Cada modelo pontua o MESMO teste (split seed 42) com o recorte que viu no treino:
  _1 -> adaptativo por raridade ; ancora _3 -> fixo 0.55. (sem train-inference skew)
Estilo: Storytelling com Dados (estilo_viz) — roxo focal, resto cinza, rotulagem direta.
Rodar: python -m pokemon.diagnosticos.viz_avaliacao_producao
"""
from pokemon.caminhos import IMAGES, MODELOS, RAW, ROOT
from pokemon.diagnosticos.estilo_viz import (ROXO, ROXO_CLARO, ROXO_TEXTO, PRETO,
                                             CINZA, CINZA_ESC, limpa, titulo, rodape)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pokemon.modelagem.split import carregar_dados_rotulados, dividir
from pokemon.modelagem.modelo_siglip import construir_modelo
from pokemon.modelagem.recorte_rarity import janela_para_rarity

TOPO_FIXO, BASE_FIXA = 0.085, 0.55   # recorte da ancora _3


class TesteDataset(Dataset):
    def __init__(self, df, preprocess, mapa_janela=None):
        # mapa_janela None -> recorte fixo (ancora); dict -> adaptativo (_1)
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.mapa_janela = mapa_janela

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cid = str(self.df.iloc[idx]["card_id"])
        img = Image.open(IMAGES / f"{cid}.png").convert("RGB")
        topo, base = (TOPO_FIXO, BASE_FIXA) if self.mapa_janela is None else self.mapa_janela[cid]
        w, h = img.size
        return self.preprocess(img.crop((0, int(h * topo), w, int(h * base)))), idx


def pontuar(modelo_path, df, preprocess, device, mapa_janela=None):
    modelo, _ = construir_modelo(blocos_descongelados=2)
    modelo.load_state_dict(torch.load(MODELOS / modelo_path))
    modelo.to(device).eval()
    loader = DataLoader(TesteDataset(df, preprocess, mapa_janela), batch_size=32, shuffle=False)
    probs = np.zeros(len(df))
    with torch.no_grad():
        for imgs, idxs in loader:
            probs[idxs.numpy()] = torch.sigmoid(modelo(imgs.to(device))).cpu().numpy()
    return probs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = construir_modelo(blocos_descongelados=2)   # so para o preprocess

    df = carregar_dados_rotulados()
    _, _, teste = dividir(df)
    y = teste["label"].values
    print(f"Teste: {len(teste)} cartas, {int(y.sum())} positivos")

    # mapa de janelas (recorte adaptativo) para o _1
    cat = pd.read_csv(RAW / "catalogo_desenvolvimento.csv")
    cat["rarity"] = cat["rarity"].fillna("nan").astype(str)
    mapa = {str(r.card_id): janela_para_rarity(r.rarity) for r in cat.itertuples()}

    probs_prod = pontuar("exp16_adaptativo_1.pt", teste, preprocess, device, mapa_janela=mapa)
    probs_anc  = pontuar("melhor_recorte_full_3.pt", teste, preprocess, device, mapa_janela=None)
    ap_prod, ap_anc = average_precision_score(y, probs_prod), average_precision_score(y, probs_anc)
    print(f"AP producao={ap_prod:.3f} | ancora={ap_anc:.3f}")

    img_dir = ROOT / "docs" / "img"; img_dir.mkdir(parents=True, exist_ok=True)

# ---------- Grafico 1: Curva PR (producao roxo = foco, ancora cinza = contexto) ----------
    fig, ax = plt.subplots(figsize=(7, 5.6)); limpa(ax, grid="y")
    prec_a, rec_a, _ = precision_recall_curve(y, probs_anc)
    ax.plot(rec_a, prec_a, lw=2, color=CINZA, zorder=2)
    prec_p, rec_p, _ = precision_recall_curve(y, probs_prod)
    ax.plot(rec_p, prec_p, lw=2.6, color=ROXO, zorder=3)
    ax.axhline(y.mean(), ls=":", lw=1.2, color=CINZA, zorder=1)
    ax.text(0.015, y.mean() + 0.02, "~aleatório", fontsize=8.5, color=CINZA)
    # rotulagem direta no espaco vazio (centro-baixo), longe das curvas
    ax.text(0.44, 0.27, f"Produção (adaptativo) · AP {ap_prod:.3f}", fontsize=9.5, color=ROXO, fontweight="bold")
    ax.text(0.44, 0.18, f"Âncora (0.55 fixo) · AP {ap_anc:.3f}", fontsize=9.5, color=CINZA)
    ax.set_xlabel("Recall", fontsize=10.5, color=CINZA_ESC)
    ax.set_ylabel("Precision", fontsize=10.5, color=CINZA_ESC)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.03)
    titulo(ax, "Produção e âncora têm desempenho equivalente",
           "Curva precision-recall no mesmo teste · a escolha foi de design, não de métrica")
    rodape(fig, f"AP quase idêntico ({ap_prod:.3f} vs {ap_anc:.3f}) — diferença dentro do ruído de "
                f"±0.02 entre rodadas. As curvas trocam de posição ao longo do recall: variação entre "
                f"instâncias, não vantagem real de uma config.")
    fig.tight_layout()
    fig.savefig(img_dir / "curva_pr_producao.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {img_dir / 'curva_pr_producao.png'}")

# ---------- Grafico 2: Trabalho economizado (recall@K do _1) ----------
    ordem = np.argsort(probs_prod)[::-1]
    y_ord = y[ordem]
    n = len(y)
    P = int(y.sum())
    
    ks = np.arange(1, n + 1)
    recall_acum = np.cumsum(y_ord) / P * 100
    diag = ks / n * 100
    
    # Encontra exatamente a posição onde chegamos no total de Positivos
    k100 = int(np.searchsorted(np.cumsum(y_ord), P)) + 1

    # Fatiamos os arrays até o ponto k100 e adicionamos o zero no início
    # Isso garante que a linha roxa e o sombreado parem no exato ponto de sucesso
    ks_plot = np.concatenate(([0], ks[:k100]))
    recall_plot = np.concatenate(([0], recall_acum[:k100]))
    diag_plot = np.concatenate(([0], diag[:k100]))

    fig, ax = plt.subplots(figsize=(7.7, 5.6))
    limpa(ax, grid="y")
    
    # Preenchimento e curva do modelo rodam APENAS nos arrays fatiados
    ax.fill_between(ks_plot, recall_plot, diag_plot, color=ROXO_CLARO, alpha=0.55, zorder=1)
    ax.plot(ks_plot, recall_plot, color=ROXO, lw=2.6, zorder=3)
    
    # A linha de revisão aleatória usa o array completo (até n) para dar contexto
    ax.plot([0, n], [0, 100], color=CINZA, lw=1.5, zorder=2)
    
    ax.scatter([k100], [100], s=55, color=ROXO, zorder=5)
    ax.text(n * 0.11, 52, "ranqueado\npelo modelo", fontsize=10, color=ROXO, fontweight="bold", ha="left")
    ax.text(n * 0.62, 46, "revisão aleatória", fontsize=9.5, color=CINZA, rotation=20)
    ax.text(n * 0.30, 73, "trabalho economizado", ha="center", fontsize=9.5, color=ROXO_TEXTO, style="italic")
    
    ax.annotate(f"revisar {k100} de {n} → todos os positivos", xy=(k100, 100),
                xytext=(k100 + n * 0.04, 103), fontsize=9, color=PRETO, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=CINZA_ESC, lw=1))
    
    ax.set_xlabel("Cartas revisadas (ranqueadas por score)", fontsize=10.5, color=CINZA_ESC)
    ax.set_ylabel("% dos olhos-fechados encontrados", fontsize=10.5, color=CINZA_ESC)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 108)
    
    titulo(ax, f"Revisar {k100} cartas (não {n}) acha todos os olhos-fechados",
           "Conjunto de teste · o modelo como triagem com humano-no-loop")
    rodape(fig, "Quanto mais rápido a curva roxa sobe acima da diagonal, mais trabalho o modelo "
                "economiza vs revisar tudo no olho.")
    
    fig.tight_layout()
    fig.savefig(img_dir / "trabalho_economizado.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {img_dir / 'trabalho_economizado.png'}")


if __name__ == "__main__":
    main()