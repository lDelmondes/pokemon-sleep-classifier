"""

Baixa as imagens das cartas com image_url para data/images/.

"""
import time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

def baixar_imagem(url: str, destino: Path):
    if destino.exists():
        return "cache"
    # timeout=(conexao, leitura): desiste rapido se o servidor nao responde
    r = requests.get(url, timeout=(10, 20))
    r.raise_for_status()
    destino.write_bytes(r.content)
    time.sleep(0.1)
    return "baixada"

def main():
    df = pd.read_csv(RAW / "catalogo_com_imagem.csv")
    total = len(df)
    print(f"{total} cartas com imagem para processar.\n")

    baixadas, cacheadas, erros = 0, 0, 0
    for i, row in enumerate(df.itertuples(), start=1):
        destino = IMAGES / f"{row.card_id}.png"
        try:
            resultado = baixar_imagem(row.image_url, destino)
            tag = "cache" if resultado == "cache" else "OK"
            if resultado == "baixada":
                baixadas += 1
            else:
                cacheadas += 1
        except Exception as e:
            erros += 1
            tag = f"ERRO ({type(e).__name__})"
        # imprime o progresso de CADA carta, com flush para aparecer na hora
        print(f"[{i}/{total}] {row.card_id:12} {tag}", flush=True)

    print(f"\nBaixadas: {baixadas} | Do cache: {cacheadas} | Erros: {erros}")
    print(f"Imagens em: {IMAGES}")

if __name__ == "__main__":
    main()