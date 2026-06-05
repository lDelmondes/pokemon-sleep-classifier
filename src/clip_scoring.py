"""

Roda CLIP zero-shot e pontua cada imagem contra varios pares de prompts (testes de hipotese). 
Saida: data/raw/scores_clip.csv, uma coluna de score por teste definido em PROMPTS_TESTES.

"""
from pathlib import Path
import pandas as pd
import torch
import open_clip
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"

# Cada par testa uma hipotese de prompt. indice 0 = positivo, indice 1 = negativo.
PROMPTS_TESTES = {
    "olhos_fechados": [
        "a pokemon card where the pokemon has its eyes closed",
        "a pokemon card where the pokemon has its eyes open",
    ],
    "dormindo": [
        "a pokemon card showing a sleeping pokemon",
        "a pokemon card showing an awake pokemon",
    ],
}

def main():
    device = "cpu"
    print("Carregando o modelo CLIP (primeira vez baixa os pesos, ~350MB)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    df = pd.read_csv(RAW / "catalogo_completo.csv")
    df = df[df["image_url"].notna()].reset_index(drop=True)
    total = len(df)

    # Pre-codifica os textos de cada teste uma vez so (nao mudam entre imagens)
    textos_cod = {}
    for nome_teste, frases in PROMPTS_TESTES.items():
        toks = tokenizer(frases).to(device)
        with torch.no_grad():
            tf = model.encode_text(toks)
            tf /= tf.norm(dim=-1, keepdim=True)
        textos_cod[nome_teste] = tf

    resultados = []
    for i, row in enumerate(df.itertuples(), start=1):
        caminho = IMAGES / f"{row.card_id}.png"
        linha = {"card_id": row.card_id, "set_codigo": row.set_codigo,
                 "numero": row.numero, "nome": row.nome}
        try:
            img = preprocess(Image.open(caminho).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                imgf = model.encode_image(img)
                imgf /= imgf.norm(dim=-1, keepdim=True)
                for nome_teste, tf in textos_cod.items():
                    probs = (100.0 * imgf @ tf.T).softmax(dim=-1)
                    linha[f"score_{nome_teste}"] = probs[0, 0].item()
        except Exception as e:
            for nome_teste in PROMPTS_TESTES:
                linha[f"score_{nome_teste}"] = None
            print(f"  ERRO em {row.card_id}: {e}")
        resultados.append(linha)
        if i % 50 == 0:
            print(f"[{i}/{total}] processadas", flush=True)

    out = pd.DataFrame(resultados)
    out.to_csv(RAW / "scores_clip.csv", index=False, encoding="utf-8")
    print(f"\nScores salvos em {RAW / 'scores_clip.csv'}")
    for nome_teste in PROMPTS_TESTES:
        print(f"\nTop 10 por '{nome_teste}':")
        top = out.sort_values(f"score_{nome_teste}", ascending=False).head(10)
        print(top[["set_codigo", "numero", "nome", f"score_{nome_teste}"]].to_string(index=False))

if __name__ == "__main__":
    main()