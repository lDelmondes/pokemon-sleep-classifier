"""
Benchmark de download de imagens da TCGdex: sequencial vs paralelo.
Baixa as mesmas N imagens das duas formas e cronometra, para decidir a
estrategia de ingestao do universo completo (~12k cartas) sem disparar
o download inteiro as cegas. NAO salva nada — so mede tempo.
"""
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# Set de teste: me04 (Chaos Rising / CRI) — era Mega, NAO ingerido ainda,
# entao mede download real de imagens que nao estao no cache local.
BASE = "https://assets.tcgdex.net/en/me/me04"
IDS = [f"{i:03d}" for i in range(1, 31)]   # 30 cartas de teste
URLS = [f"{BASE}/{i}/high.png" for i in IDS]

N_WORKERS = 20   # quantas baixas em paralelo

def baixar(url):
    try:
        r = requests.get(url, timeout=30)
        return len(r.content) if r.status_code == 200 else 0
    except Exception:
        return 0

def sequencial(urls):
    t = time.perf_counter()
    total = sum(baixar(u) for u in urls)
    dt = time.perf_counter() - t
    return dt, total

def paralelo(urls, workers):
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        resultados = list(ex.map(baixar, urls))
    dt = time.perf_counter() - t
    return dt, sum(resultados)

def main():
    print(f"Testando download de {len(URLS)} imagens (qualidade high)...\n")

    print("1. SEQUENCIAL (uma de cada vez)...")
    dt_seq, bytes_seq = sequencial(URLS)
    print(f"   {dt_seq:.1f}s para {len(URLS)} imagens ({dt_seq/len(URLS):.2f}s por imagem)")

    print(f"\n2. PARALELO ({N_WORKERS} simultaneas)...")
    dt_par, bytes_par = paralelo(URLS, N_WORKERS)
    print(f"   {dt_par:.1f}s para {len(URLS)} imagens ({dt_par/len(URLS):.2f}s por imagem)")

    print(f"\n--- RESULTADO ---")
    if dt_par > 0:
        print(f"Speedup: {dt_seq/dt_par:.1f}x mais rapido em paralelo")
    # extrapolacao para o universo completo
    n_total = 12217
    print(f"\nExtrapolando para {n_total} cartas:")
    print(f"  Sequencial: {dt_seq/len(URLS)*n_total/60:.0f} min")
    print(f"  Paralelo:   {dt_par/len(URLS)*n_total/60:.0f} min")

if __name__ == "__main__":
    main()