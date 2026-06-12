"""
Ingestao em MASSA dos sets faltantes do universo (le inventario_sets.csv).
Paraleliza metadados E imagens (ThreadPoolExecutor). Incremental (pula o que
ja existe). Retry em falhas. PRESERVA o catalogo existente (rarity + label) —
adiciona so as cartas novas, nao sobrescreve.
"""
from pokemon.caminhos import RAW, IMAGES, CACHE
from pokemon.dados.tcgdex_utils import cartas_do_set, montar_url_imagem
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.tcgdex.net/v2/en"
N_WORKERS = 20
EXCLUIR = {"sve", "mee"}        # Energy: sem Pokemon
MAX_RETRY = 3

def carta_completa_cache(card_id):
    """Metadados da carta, com cache. Retry em falha de rede."""
    f = CACHE / f"{card_id}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    for tentativa in range(MAX_RETRY):
        try:
            r = requests.get(f"{API}/cards/{card_id}", timeout=(10, 20))
            r.raise_for_status()
            dados = r.json()
            f.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
            return dados
        except Exception:
            if tentativa == MAX_RETRY - 1:
                return None
            time.sleep(1)

def baixar_imagem_retry(url, destino):
    """Baixa imagem se nao existe. Retry em falha. Retorna True se ok."""
    if destino.exists():
        return True
    if not url:
        return False
    for tentativa in range(MAX_RETRY):
        try:
            r = requests.get(url, timeout=(10, 30))
            r.raise_for_status()
            destino.write_bytes(r.content)
            return True
        except Exception:
            if tentativa == MAX_RETRY - 1:
                return False
            time.sleep(1)

def processar_carta(card_id, codigo, era):
    """Baixa metadados + imagem de UMA carta. Retorna a linha do catalogo ou None."""
    full = carta_completa_cache(card_id)
    if full is None or full.get("category") != "Pokemon":
        return None
    url_img = montar_url_imagem(full)
    destino = IMAGES / f"{full['id']}.png"
    ok = baixar_imagem_retry(url_img, destino)
    return {
        "card_id": full["id"],
        "set_codigo": codigo,
        "era": era,
        "numero": full.get("localId"),
        "nome": full["name"],
        "image_url": url_img if ok else None,
        "label": 0,
        "rarity": full.get("rarity"),
        "_img_ok": ok,
    }

def main():
    # 1. Catalogo existente — PRESERVAR
    cat_path = RAW / "catalogo_completo.csv"
    cat_atual = pd.read_csv(cat_path)
    ja_tenho_cartas = set(cat_atual["card_id"])
    print(f"Catalogo atual: {len(cat_atual)} cartas (preservado)")

    # 2. Sets a ingerir: faltantes do inventario, menos Energy
    inv = pd.read_csv(RAW / "inventario_sets.csv")
    falta = inv[~inv["ja_tenho"] & ~inv["set_id"].isin(EXCLUIR)]
    print(f"Sets a ingerir: {len(falta)} (excluindo Energy {EXCLUIR})\n")

    novas_linhas = []
    falhas_img = []

    for _, srow in falta.iterrows():
        sid, codigo, era = srow["set_id"], srow["set_id"].upper(), srow["serie"].upper()
        print(f"=== {sid} ({srow['set_nome']}) ===", flush=True)
        try:
            cartas = cartas_do_set(sid)
        except Exception as e:
            print(f"  ERRO ao listar set {sid}: {type(e).__name__} — pulando", flush=True)
            continue

        # so as cartas que ainda nao tenho no catalogo
        ids_novos = [c["id"] for c in cartas if c["id"] not in ja_tenho_cartas]
        if not ids_novos:
            print(f"  todas as {len(cartas)} ja no catalogo, pulando")
            continue

        # paraleliza o processamento das cartas deste set
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futuros = {ex.submit(processar_carta, cid, codigo, era): cid for cid in ids_novos}
            n_poke = 0
            for fut in as_completed(futuros):
                linha = fut.result()
                if linha is None:
                    continue
                if not linha.pop("_img_ok"):
                    falhas_img.append(linha["card_id"])
                novas_linhas.append(linha)
                n_poke += 1
        print(f"  {sid}: {n_poke} cartas de Pokemon novas", flush=True)

    # 3. Junta com o catalogo existente — NAO sobrescreve
    if novas_linhas:
        df_novas = pd.DataFrame(novas_linhas)
        # alinha colunas com o catalogo atual (caso ordem difira)
        df_final = pd.concat([cat_atual, df_novas], ignore_index=True)
        df_final = df_final.drop_duplicates(subset="card_id", keep="first")
        df_final.to_csv(cat_path, index=False, encoding="utf-8")
        print(f"\nCatalogo: {len(cat_atual)} -> {len(df_final)} cartas (+{len(df_final)-len(cat_atual)})")
    else:
        print("\nNenhuma carta nova adicionada.")

    if falhas_img:
        print(f"\n{len(falhas_img)} imagens FALHARAM (re-rode para tentar de novo):")
        print(falhas_img[:30])

if __name__ == "__main__":
    main()