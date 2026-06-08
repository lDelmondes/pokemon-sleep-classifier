"""

Loop de treino do fine-tuning do CLIP para "olhos fechados".
- split estratificado treino/val/teste
- modelo com blocos 10-11 + cabeca descongelados
- BCEWithLogitsLoss com pos_weight (desbalanceamento)
- Adam, learning rate baixo
- early stopping vigiando a perda de validacao

Roda na GPU. Saida: modelo salvo + metricas no teste.

"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from split import carregar_dados_rotulados, dividir
from dataset import CartasDataset, transform_treino, transform_avaliacao
from modelo import construir_modelo

ROOT = Path(__file__).resolve().parent.parent
MODELOS = ROOT / "data" / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

# --- Hiperparametros ---
EPOCAS_MAX = 50
PACIENCIA = 7          # epocas sem melhora na validacao antes de parar
LEARNING_RATE = 1e-5   # passo pequeno: fine-tuning afina, nao reconstroi
BATCH_SIZE = 16

def rodar_epoca(modelo, loader, loss_fn, device, otimizador=None):
    """Roda uma epoca. Se otimizador for dado -> modo treino (ajusta pesos).
    Se otimizador for None -> modo avaliacao (so mede)."""
    treinando = otimizador is not None
    modelo.train() if treinando else modelo.eval()

    perda_total, n = 0.0, 0
    # no_grad desliga o calculo de gradiente quando so estamos medindo
    contexto = torch.enable_grad() if treinando else torch.no_grad()
    with contexto:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = modelo(imgs)               # forward pass
            perda = loss_fn(logits, labels)     # mede a dor (com pos_weight)
            if treinando:
                otimizador.zero_grad()          # zera gradientes antigos
                perda.backward()                # backprop calcula gradientes
                otimizador.step()               # otimizador gira os botoes
            perda_total += perda.item() * len(labels)
            n += len(labels)
    return perda_total / n

def main():
    device = "cuda"   # confirmamos que a GPU esta disponivel
    print(f"Treinando em: {device}")

    # 1. Dados e split (Peca 2)
    df = carregar_dados_rotulados()
    treino_df, val_df, teste_df = dividir(df)

    ds_treino = CartasDataset(treino_df, transform_treino)      # COM augmentation
    ds_val = CartasDataset(val_df, transform_avaliacao)          # SEM
    ds_teste = CartasDataset(teste_df, transform_avaliacao)      # SEM

    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    dl_teste = DataLoader(ds_teste, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Modelo (Peca 3)
    modelo, _ = construir_modelo(blocos_descongelados=1)
    modelo.to(device)

    # 3. Funcao de perda com pos_weight (desbalanceamento)
    n_pos = int(treino_df["label"].sum())
    n_neg = len(treino_df) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"pos_weight = {n_neg}/{n_pos} = {n_neg/n_pos:.1f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Otimizador: SO os parametros descongelados (requires_grad=True)
    params = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.Adam(params, lr=LEARNING_RATE)

    # 5. Loop de epocas com early stopping
    melhor_perda_val = float("inf")
    epocas_sem_melhora = 0
    caminho_melhor = MODELOS / "melhor_modelo.pt"

    for epoca in range(1, EPOCAS_MAX + 1):
        perda_treino = rodar_epoca(modelo, dl_treino, loss_fn, device, otimizador)
        perda_val = rodar_epoca(modelo, dl_val, loss_fn, device, otimizador=None)
        print(f"Epoca {epoca:2} | treino {perda_treino:.4f} | val {perda_val:.4f}", flush=True)

        if perda_val < melhor_perda_val:
            melhor_perda_val = perda_val
            epocas_sem_melhora = 0
            torch.save(modelo.state_dict(), caminho_melhor)   # salva o melhor
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= PACIENCIA:
                print(f"\nEarly stopping na epoca {epoca} (sem melhora ha {PACIENCIA}).")
                break

    # 6. Restaura o melhor modelo e avalia no TESTE
    modelo.load_state_dict(torch.load(caminho_melhor))
    avaliar_teste(modelo, dl_teste, device)

def avaliar_teste(modelo, loader, device):
    """Mede recall/precision por top-K no conjunto de teste."""
    modelo.eval()
    todos_probs, todos_y = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = modelo(imgs.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()   # logit -> probabilidade
            todos_probs.extend(probs)
            todos_y.extend(labels.numpy())

    probs, y = np.array(todos_probs), np.array(todos_y)
    ordem = np.argsort(probs)[::-1]
    y_ord = y[ordem]
    P = int(y.sum())
    print(f"\n=== TESTE: {len(y)} cartas, {P} positivos ===")
    print(f"  K  | recall | precision")
    for K in [7, 10, 15, 20, 30]:
        vp = int(y_ord[:K].sum())
        print(f"  {K:3} |  {vp/P:4.0%}  |   {vp/K:4.0%}")
    ultimo = int(np.where(y_ord == 1)[0].max()) + 1
    print(f"\n  Para recall 100%: revisar as {ultimo} de {len(y)} cartas melhor rankeadas")

if __name__ == "__main__":
    main()