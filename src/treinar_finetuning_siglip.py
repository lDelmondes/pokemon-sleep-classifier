"""

Fine-tuning do SigLIP2 para "olhos fechados". Reaproveita split.py e dataset.py, mas usa modelo_siglip.py e o preprocess proprio do SigLIP.
- split estratificado treino/val/teste
- 2 blocos + norm + attn_pool + cabeca descongelados
- BCEWithLogitsLoss com pos_weight, Adam lr baixo, early stopping

Saida: data/modelos/melhor_siglip.pt + metricas no teste.

"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from split import carregar_dados_rotulados, dividir
from modelo_siglip import construir_modelo

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"
MODELOS = ROOT / "data" / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

EPOCAS_MAX = 50
PACIENCIA = 7
LEARNING_RATE = 1e-5
BATCH_SIZE = 16

# Dataset que aplica o preprocess do SigLIP (passado de fora) + augmentation no treino
class CartasDataset(Dataset):
    def __init__(self, df, preprocess, treino=False):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        # augmentation so no treino, ANTES do preprocess do SigLIP
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if treino else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(IMAGES / f"{row['card_id']}.png").convert("RGB")
        if self.aug:
            img = self.aug(img)
        img = self.preprocess(img)              # preprocess oficial do SigLIP
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

def avaliar_teste(modelo, loader, device, P):
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
    print(f"  K  | recall | precision")
    for K in [10, 20, 30, 50, 80]:
        vp = int(y_ord[:K].sum())
        print(f"  {K:3} |  {vp/Pt:4.0%}  |   {vp/K:4.0%}")
    ultimo = int(np.where(y_ord == 1)[0].max()) + 1
    print(f"  Para recall 100%: revisar {ultimo} de {len(y)}")

def main():
    device = "cuda"
    print(f"Treinando em: {device}")

    modelo, preprocess = construir_modelo(blocos_descongelados=3)
    modelo.to(device)

    df = carregar_dados_rotulados()
    treino_df, val_df, teste_df = dividir(df)

    dl_treino = DataLoader(CartasDataset(treino_df, preprocess, treino=True),
                           batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(CartasDataset(val_df, preprocess, treino=False),
                        batch_size=BATCH_SIZE, shuffle=False)
    dl_teste = DataLoader(CartasDataset(teste_df, preprocess, treino=False),
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
    caminho = MODELOS / "melhor_siglip.pt"
    for epoca in range(1, EPOCAS_MAX + 1):
        pt = rodar_epoca(modelo, dl_treino, loss_fn, device, otimizador)
        pv = rodar_epoca(modelo, dl_val, loss_fn, device, otimizador=None)
        print(f"Epoca {epoca:2} | treino {pt:.4f} | val {pv:.4f}", flush=True)
        if pv < melhor_val:
            melhor_val = pv
            sem_melhora = 0
            torch.save(modelo.state_dict(), caminho)
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                print(f"\nEarly stopping na epoca {epoca}.")
                break

    modelo.load_state_dict(torch.load(caminho))
    avaliar_teste(modelo, dl_teste, device, int(df["label"].sum()))

if __name__ == "__main__":
    main()