"""

Fonte unica de caminhos do projeto. Todos os modulos importam daqui em vez de recalcular Path(__file__).

"""
from pathlib import Path

# Este arquivo vive em pokemon/caminhos.py; parent.parent sobe ate a raiz
# do projeto (a pasta que contem 'pokemon/' e 'data/').
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"
IMAGES = DATA / "images"
CACHE = DATA / "cache_cards"
MODELOS = DATA / "modelos"

# Centraliza os mkdir que antes estavam espalhados pelos scripts (idempotente).
for _p in (RAW, LABELS, IMAGES, CACHE, MODELOS):
    _p.mkdir(parents=True, exist_ok=True)