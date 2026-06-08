"""

Funcoes compartilhadas para falar com a API da TCGdex.
Importado pelos scripts de ingestao -- uma fonte da verdade, sem duplicacao.

"""
import json, time
from pathlib import Path 
from pokemon.caminhos import CACHE
import requests

API = "https://api.tcgdex.net/v2/en"

def get(url):
    """GET simples com timeout (conexao, leitura) e erro explicito."""
    r = requests.get(url, timeout=(10, 20))
    r.raise_for_status()
    return r.json()

def listar_sets():
    """Todos os sets da TCGdex (resumo: id, name)."""
    return get(f"{API}/sets")

def cartas_do_set(set_id):
    """Lista resumida das cartas de um set."""
    return get(f"{API}/sets/{set_id}")["cards"]

def carta_completa(card_id):
    """Objeto completo da carta, com cache em disco."""
    f = CACHE / f"{card_id}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    dados = get(f"{API}/cards/{card_id}")
    f.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.1)
    return dados

def montar_url_imagem(full_card, qualidade="high", formato="png"):
    """Monta a URL da imagem a partir do objeto da carta. None se nao houver."""
    base = full_card.get("image")
    return f"{base}/{qualidade}.{formato}" if base else None

def baixar_imagem(url, destino: Path):
    """Baixa a imagem se ainda nao existe em disco. Retorna 'cache' ou 'baixada'."""
    if destino.exists():
        return "cache"
    r = requests.get(url, timeout=(10, 20))
    r.raise_for_status()
    destino.write_bytes(r.content)
    time.sleep(0.1)
    return "baixada"