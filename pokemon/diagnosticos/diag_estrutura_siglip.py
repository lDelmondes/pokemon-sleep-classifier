"""

Inspeciona a estrutura interna do SigLIP2 para descobrir onde estao os
blocos do transformer (necessario para saber o que descongelar no fine-tuning).

"""
import open_clip

model, _ = open_clip.create_model_from_pretrained("hf-hub:timm/ViT-B-16-SigLIP2")

# Mostra a estrutura de alto nivel da torre de visao
print("=== Atributos de model.visual ===")
print([a for a in dir(model.visual) if not a.startswith("_")][:30])

print("\n=== Estrutura resumida de model.visual ===")
print(model.visual)