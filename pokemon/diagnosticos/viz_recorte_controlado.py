"""[Visual 2/4] A descoberta central (Exp.13): recorte da arte, controlado.
Salva docs/img/recorte_controlado.png.
"""
from pokemon.caminhos import ROOT
from pokemon.diagnosticos.estilo_viz import ROXO, CINZA, CINZA_ESC, limpa, titulo, rodape
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE = [0.507, 0.566, 0.559]   # Exp.13 sem recorte
RECORTE  = [0.755, 0.759, 0.858]   # Exp.13 recorte 55%

def main():
    fig, ax = plt.subplots(figsize=(7, 5.6))
    limpa(ax, grid="y")
    rng = np.random.default_rng(3)
    for vals, cor, x in [(BASELINE, CINZA, 0), (RECORTE, ROXO, 1)]:
        xs = rng.normal(x, 0.028, len(vals))
        ax.plot([x, x], [min(vals), max(vals)], color=cor, lw=1, alpha=0.35, zorder=1)
        ax.scatter(xs, vals, s=120, color=cor, zorder=3, edgecolor="white", linewidth=1.5)
        m = float(np.mean(vals))
        ax.hlines(m, x-0.15, x+0.15, color=cor, lw=2.6, zorder=4)
        ax.text(x+0.2, m, f"média {m:.2f}", va="center", fontsize=10.5, color=cor, fontweight="bold")
    ax.text(0.5, 0.71, "+0.25 PR-AUC", ha="center", fontsize=12.5, color=ROXO, fontweight="bold")
    ax.text(0.5, 0.675, "só mudando o recorte", ha="center", fontsize=9, color=CINZA_ESC)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Sem recorte", "Com recorte (55%)"], fontsize=10.5, color=CINZA_ESC)
    ax.set_ylim(0.45, 0.90); ax.set_xlim(-0.45, 1.62); ax.set_ylabel("PR-AUC", fontsize=10.5, color=CINZA_ESC)
    titulo(ax, "Recortar a arte quase dobrou o PR-AUC",
           "Mesma base (4.022 cartas) · única variável = recorte · 3 rodadas cada")
    rodape(fig, "As faixas não se sobrepõem: a pior rodada com recorte (0.76) supera a melhor sem "
                "recorte (0.57). O ganho é real, não ruído de inicialização.")
    fig.tight_layout()
    saida = ROOT / "docs" / "img" / "recorte_controlado.png"
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {saida}")

if __name__ == "__main__":
    main()