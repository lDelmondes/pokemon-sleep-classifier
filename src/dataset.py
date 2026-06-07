"""

Define o Dataset de cartas: le um PNG do disco, aplica transformacoes (pre-processamento + augmentation no treino) e entrega (tensor, rotulo).

Modulo importado pelo script de treino; nao roda sozinho.

"""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"

# Medias e desvios que o CLIP espera (padrao do pre-treino).
# Normalizar com esses valores poe a imagem na escala que a rede reconhece.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Transformacoes de TREINO: pre-processamento + augmentation.
transform_treino = transforms.Compose([
    transforms.Resize((224, 224)),                  # tamanho que o CLIP espera
    transforms.RandomHorizontalFlip(p=0.5),         # espelha 50% das vezes
    transforms.RandomRotation(degrees=15),          # gira ate +-15 graus
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # varia luz/contraste
    transforms.ToTensor(),                          # PNG -> tensor [0,1]
    transforms.Normalize(CLIP_MEAN, CLIP_STD),      # poe na escala do CLIP
])

# Transformacoes de AVALIACAO: SO pre-processamento, SEM augmentation.
transform_avaliacao = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

class CartasDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        # df precisa ter colunas 'card_id' e 'label'
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        # responde "quantas imagens tem?"
        return len(self.df)

    def __getitem__(self, idx):
        # responde "me da a imagem idx, pronta"
        row = self.df.iloc[idx]
        caminho = IMAGES / f"{row['card_id']}.png"
        img = Image.open(caminho).convert("RGB")   # garante 3 canais
        img = self.transform(img)                   # aplica as transformacoes
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label