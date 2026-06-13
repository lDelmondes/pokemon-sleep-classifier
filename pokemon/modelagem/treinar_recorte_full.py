"""
Treino com RECORTE ADAPTATIVO POR RARIDADE (Exp. 16) sobre o catalogo COMPLETO.
Topo fixo em 8.5%; o limite inferior (base) varia por raridade, via
pokemon.modelagem.recorte_rarity. Salva em exp16_adaptativo_<rodada>.pt
(NÃO sobrescreve a âncora melhor_recorte_full_*.pt).
"""
from pokemon.caminhos import IMAGES, MODELOS, RAW
import numpy as np
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import average_precision_score

from pokemon.modelagem.split import carregar_dados_rotulados, dividir   # << split COMPLETO
from pokemon.modelagem.modelo_siglip import construir_modelo
from pokemon.modelagem.recorte_rarity import janela_para_rarity, checar_cobertura

EPOCAS_MAX = 50
PACIENCIA = 7
LEARNING_RATE = 1e-5
BATCH_SIZE = 16


class CartasDataset(Dataset):
    def __init__(self, df, preprocess, mapa_janela, treino=False):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.mapa_janela = mapa_janela            # card_id -> (topo, base)
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if treino else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        card_id = str(row["card_id"])
        img = Image.open(IMAGES / f"{card_id}.png").convert("RGB")

        # Dimensões da imagem original
        w, h = img.size

        # RECORTE DINÂMICO: janela (topo, base) definida pela raridade da carta.
        # PIL.Image.crop -> (esquerda, topo, direita, base). Mesmo ponto do
        # pipeline onde antes ficava o crop fixo: crop -> aug -> preprocess.
        topo, base = self.mapa_janela[card_id]
        img = img.crop((0, int(h * topo), w, int(h * base)))

        if self.aug:
            img = self.aug(img)

        img = self.preprocess(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label


def rodar_epoca(modelo, loader, loss_fn, device, otimizador=None):
    treinando = otimizador is not None
    modelo.train() if treinando else modelo.eval()
    perda_total, n = 0.0, 0
    contexto = torch.enable_grad() if treinando else torch.no_grad()
    with contexto:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = modelo(imgs)
            perda = loss_fn(logits, labels)
            if treinando:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            perda_total += perda.item() * len(labels)
            n += len(labels)
    return perda_total / n


def avaliar_teste(modelo, loader, device):
    modelo.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = modelo(imgs.to(device))
            probs_all.extend(torch.sigmoid(logits).cpu().numpy())
            y_all.extend(labels.numpy())
    probs, y = np.array(probs_all), np.array(y_all)
    ordem = np.argsort(probs)[::-1]
    y_ord = y[ordem]
    Pt = int(y.sum())
    print(f"\n=== TESTE: {len(y)} cartas, {Pt} positivos ===")
    pr_auc = average_precision_score(y, probs)
    print(f"  PR-AUC (average precision): {pr_auc:.3f}")
    print(f"  K  | recall | precision")
    for K in [10, 20, 30, 50, 80]:
        vp = int(y_ord[:K].sum())
        print(f"  {K:3} |  {vp/Pt:4.0%}  |   {vp/K:4.0%}")
    ultimo = int(np.where(y_ord == 1)[0].max()) + 1
    print(f"  Para recall 100%: revisar {ultimo} de {len(y)}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Captura o número da rodada do terminal (ou usa "1" como padrão)
    rodada = sys.argv[1] if len(sys.argv) > 1 else "1"

    print(f"Treinando em: {device} | Rodada: {rodada} | RECORTE ADAPTATIVO POR RARIDADE")

    modelo, preprocess = construir_modelo(blocos_descongelados=2)
    modelo.to(device)

    df = carregar_dados_rotulados()
    treino_df, val_df, teste_df = dividir(df)

    # --- Exp.16: mapa card_id -> (topo, base) a partir do catálogo completo ---
    catalogo = pd.read_csv(RAW / "catalogo_completo.csv")
    catalogo["rarity"] = catalogo["rarity"].fillna("nan").astype(str)
    checar_cobertura(catalogo)                       # GATE: fail-loud se faltar raridade
    mapa_janela = {str(r.card_id): janela_para_rarity(r.rarity)
                   for r in catalogo.itertuples()}
    faltando = set(df["card_id"].astype(str)) - set(mapa_janela)
    if faltando:
        raise ValueError(
            f"{len(faltando)} cards rotulados fora do catálogo "
            f"(ex: {list(faltando)[:5]}). Recorte não definido para eles."
        )
    bases = sorted({b for _, b in mapa_janela.values()})
    print(f"[exp16] recorte dinâmico ATIVO | {len(mapa_janela)} cards | bases={bases}")
    # --------------------------------------------------------------------------

    dl_treino = DataLoader(CartasDataset(treino_df, preprocess, mapa_janela, treino=True),
                           batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(CartasDataset(val_df, preprocess, mapa_janela, treino=False),
                        batch_size=BATCH_SIZE, shuffle=False)
    dl_teste = DataLoader(CartasDataset(teste_df, preprocess, mapa_janela, treino=False),
                          batch_size=BATCH_SIZE, shuffle=False)

    n_pos = int(treino_df["label"].sum())
    n_neg = len(treino_df) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"Treino: {len(treino_df)} cartas, {n_pos} positivos | pos_weight={n_neg/n_pos:.1f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    params = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.Adam(params, lr=LEARNING_RATE)

    melhor_val = float("inf")
    sem_melhora = 0

    # Nome de saída separado da âncora (NÃO sobrescreve melhor_recorte_full_*.pt)
    caminho = MODELOS / f"exp16_adaptativo_{rodada}.pt"

    t_inicio = time.perf_counter()
    for epoca in range(1, EPOCAS_MAX + 1):
        t_epoca = time.perf_counter()
        pt = rodar_epoca(modelo, dl_treino, loss_fn, device, otimizador)
        pv = rodar_epoca(modelo, dl_val, loss_fn, device, otimizador=None)
        dt = time.perf_counter() - t_epoca
        print(f"Epoca {epoca:2} | treino {pt:.4f} | val {pv:.4f} | {dt:.1f}s", flush=True)
        if pv < melhor_val:
            melhor_val = pv
            sem_melhora = 0
            torch.save(modelo.state_dict(), caminho)
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                print(f"\nEarly stopping na epoca {epoca}.")
                break
    t_total = time.perf_counter() - t_inicio
    print(f"Tempo total de treino: {t_total:.1f}s ({t_total/60:.1f} min)")

    # Carrega o melhor checkpoint da rodada atual para o teste
    modelo.load_state_dict(torch.load(caminho))
    avaliar_teste(modelo, dl_teste, device)


if __name__ == "__main__":
    main()