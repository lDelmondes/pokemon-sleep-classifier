"""

Classificador de fine-tuning sobre o SigLIP2 (ViT-B-16-SigLIP2):
- congela todo o corpo de visao
- descongela os N ultimos blocos do transformer (trunk.blocks)
- cabeca binaria nova (768 -> 1)

Estrutura confirmada via diagnostico: model.visual.trunk.blocks (12 blocos, dim 768).

"""
import torch
import torch.nn as nn
import open_clip

def construir_modelo(blocos_descongelados=2):
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:timm/ViT-B-16-SigLIP2"
    )
    visual = model.visual   # TimmModel (contem .trunk)

    # 1. Congela TUDO no corpo de visao
    for p in visual.parameters():
        p.requires_grad = False

    # 2. Descongela os ultimos N blocos do transformer.
    #    Caminho confirmado pelo diagnostico: visual.trunk.blocks
    blocos = visual.trunk.blocks            # Sequential com 12 Blocks
    for bloco in blocos[-blocos_descongelados:]:
        for p in bloco.parameters():
            p.requires_grad = True

    # 3. (Opcional) descongela tambem a norm final e o attn_pool, que ficam
    #    DEPOIS dos blocos e ajudam a adaptar a representacao agregada.
    for p in visual.trunk.norm.parameters():
        p.requires_grad = True
    for p in visual.trunk.attn_pool.parameters():
        p.requires_grad = True

    return ClassificadorSigLIP(visual), preprocess

class ClassificadorSigLIP(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.cabeca = nn.Linear(768, 1)   # SigLIP2-B = 768 dims (nao 512)

    def forward(self, x):
        feats = self.visual(x)            # (batch, 768)
        logit = self.cabeca(feats)        # (batch, 1)
        return logit.squeeze(1)