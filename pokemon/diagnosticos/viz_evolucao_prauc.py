"""[Visual 1/4] Evolucao do PR-AUC ao longo da jornada (EXPERIMENTS.md).
Salva docs/img/evolucao_prauc.png. Regua mista (CV vs hold-out) sinalizada no grafico.
"""
from pokemon.caminhos import ROOT
from pokemon.diagnosticos.estilo_viz import ROXO, CINZA, CINZA_ESC, CINZA_CLARO, limpa, titulo, rodape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROTULOS = ["CLIP\n+ logística", "SigLIP\n+ logística", "Fine-tuning\nSigLIP", "Fine-tuning\n+ recorte"]
VALORES = [0.206, 0.273, 0.49, 0.68]   # Exp.2 / Exp.5 / Exp.11-12 / Exp.14

def main():
    cores = [CINZA, CINZA, CINZA, ROXO]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    limpa(ax, grid=None)
    ax.spines["left"].set_visible(False); ax.get_yaxis().set_visible(False)
    bars = ax.bar(range(4), VALORES, color=cores, width=0.6, zorder=3)
    for i, (b, v) in enumerate(zip(bars, VALORES)):
        ax.text(b.get_x()+b.get_width()/2, v+0.013, f"{v:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=ROXO if i == 3 else CINZA_ESC)
    ax.axvline(1.5, color=CINZA_CLARO, lw=1, ls=(0, (4, 4)), zorder=1, ymax=0.82)
    ax.text(0.5, 0.79, "validação cruzada · 1.257 cartas", ha="center", fontsize=8.5, color=CINZA, style="italic")
    ax.text(2.5, 0.79, "hold-out · 5.802 cartas", ha="center", fontsize=8.5, color=CINZA, style="italic")
    ax.text(3.48, 0.52, "+0.19\ncom o recorte", ha="left", va="center",
            fontsize=10, color=ROXO, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(ROTULOS, fontsize=10, color=CINZA_ESC)
    ax.set_ylim(0, 0.86); ax.set_xlim(-0.6, 4.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(ROTULOS, fontsize=10, color=CINZA_ESC)
    ax.set_ylim(0, 0.86); ax.set_xlim(-0.6, 3.85)
    titulo(ax, "O recorte da arte destravou o modelo — não foi mais dados",
           "PR-AUC ao longo da jornada · marcos de réguas distintas (leitura de ordem de magnitude)")
    rodape(fig, "Os dois primeiros via validação cruzada (1.257 cartas); os dois últimos via hold-out "
                "(5.802). Não é comparação pareada. Produção (recorte adaptativo) = 0.69.")
    fig.tight_layout()
    saida = ROOT / "docs" / "img" / "evolucao_prauc.png"
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {saida}")

if __name__ == "__main__":
    main()