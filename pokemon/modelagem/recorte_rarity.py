"""Recorte adaptativo por raridade (Exp. 16). Topo fixo 0.085; base por raridade."""
from __future__ import annotations
import pandas as pd

TOPO = 0.085  # herança Exp.15 (cortar topo é neutro)
FALLBACK = 0.55  # âncora de produção; só dispara se aparecer raridade nova

# CHAVES = string EXATA da coluna 'rarity' (TCGdex). "nan" cobre rarity ausente.
LIMITE_INFERIOR = {
    "Amazing Rare": 0.48,
    "Black White Rare": 0.60,
    "Classic Collection": 0.48,         
    "Common": 0.48,
    "Double rare": 0.48,
    "Holo Rare": 0.48,
    "Holo Rare V": 0.55,
    "Holo Rare VMAX": 0.55,
    "Holo Rare VSTAR": 0.48,
    "Hyper rare": 0.55,
    "Illustration rare": 0.55,
    "Mega Hyper Rare": 0.55,
    "Radiant Rare": 0.48,
    "Rare": 0.48,
    "Secret Rare": 0.55,
    "Shiny rare": 0.48,
    "Shiny rare V": 0.55,
    "Shiny rare VMAX": 0.55,
    "Shiny Ultra Rare": 0.55,
    "Special illustration rare": 0.55,
    "Ultra Rare": 0.55,
    "Uncommon": 0.48,
    "nan": 0.48,
}

def janela_para_rarity(rarity):
    chave = "nan" if rarity is None or (isinstance(rarity, float) and pd.isna(rarity)) else str(rarity)
    return (TOPO, LIMITE_INFERIOR.get(chave, FALLBACK))

def checar_cobertura(catalogo, col="rarity"):
    presentes = set(catalogo[col].fillna("nan").astype(str).unique())
    faltando = presentes - set(LIMITE_INFERIOR)
    if faltando:
        raise ValueError(
            f"Raridades no catálogo SEM entrada no dict: {sorted(faltando)}. "
            f"Corrija antes de treinar — senão caem no FALLBACK {FALLBACK} sem você mandar."
        )
    print(f"[recorte_rarity] OK: {len(presentes)} raridades cobertas.")

def recortar(img, rarity):
    topo, base = janela_para_rarity(rarity)
    w, h = img.size
    return img.crop((0, int(h * topo), w, int(h * base)))