"""Identidade visual compartilhada dos graficos (principios de 'Storytelling com
Dados', Knaflic): cor focal unica (roxo), resto em cinza, clutter minimo, titulo
como takeaway, rotulagem direta. Importado por todos os viz_*.py p/ consistencia.
"""
import matplotlib.pyplot as plt

ROXO        = "#6C2DC7"   # cor FOCAL — usar so no que e a mensagem
ROXO_CLARO  = "#DCC9F2"   # preenchimento suave (areas)
ROXO_TEXTO  = "#7A5BA6"   # texto sobre area roxa clara
PRETO       = "#262626"   # titulos
CINZA_ESC   = "#595959"   # texto secundario, rotulos de eixo
CINZA       = "#A6A6A6"   # series de CONTEXTO (ancora, baseline, referencia)
CINZA_CLARO = "#DBDBDB"   # grid, spines, linhas de referencia
BRANCO      = "#FFFFFF"

def limpa(ax, grid="y"):
    """Remove clutter: spines de cima/direita, grid suave, ticks sem tracinho."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(CINZA_CLARO)
    if grid:
        ax.grid(axis=grid, ls="-", lw=0.6, color=CINZA_CLARO, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=CINZA_ESC, length=0)

def titulo(ax, takeaway, sub=None):
    """Titulo como AFIRMACAO (takeaway), alinhado a esquerda. Subtitulo opcional."""
    ax.text(0, 1.12 if sub else 1.03, takeaway, transform=ax.transAxes, fontsize=13.5,
            fontweight="bold", color=PRETO, va="bottom", ha="left")
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9.5,
                color=CINZA_ESC, va="bottom", ha="left")

def rodape(fig, texto):
    fig.text(0.5, -0.02, texto, ha="center", va="top", fontsize=7.3, color=CINZA, wrap=True)