"""

Monta o classificador de fine-tuning sobre o CLIP-ViT-B-32:
- congela todo o corpo
- descongela os 2 ultimos blocos (10 e 11)
- troca a "cabeca" por uma camada binaria nova (treinada do zero)

Modulo importado pelo script de treino.

"""
import torch
import torch.nn as nn
import open_clip

def construir_modelo(blocos_descongelados=2):
    # Carrega o CLIP pre-treinado (mesmo do scoring/embeddings)
    clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    # So nos interessa a torre de visao (o encoder de imagem)
    visual = clip.visual

    # 1. CONGELA tudo no corpo de visao
    for p in visual.parameters():
        p.requires_grad = False

    # 2. DESCONGELA os ultimos N blocos do transformer
    blocos = visual.transformer.resblocks      # lista dos 12 blocos
    for bloco in blocos[-blocos_descongelados:]:
        for p in bloco.parameters():
            p.requires_grad = True

    return ClassificadorCLIP(visual), preprocess

class ClassificadorCLIP(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual                    # corpo do CLIP (parte congelado)
        # cabeca nova: do tamanho da saida do CLIP (512) -> 1 numero (logit)
        self.cabeca = nn.Linear(512, 1)

    def forward(self, x):
        # x = batch de imagens (tensores). Passa pelo corpo -> representacao
        feats = self.visual(x)                  # (batch, 512)
        logit = self.cabeca(feats)              # (batch, 1)
        return logit.squeeze(1)                 # (batch,) - um numero por imagem