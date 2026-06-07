# Classificador de Pokémons dormindo

Um produto de dados para um problema real: minha namorada coleciona cartas de
Pokémon TCG onde o Pokémon aparece de **dormindo**. Triar +15.000 cartas
à mão é inviável — então construí um classificador de imagens para rankear
candidatos e montar a lista de compras da coleção.

> Cada abordagem testada, o porquê de cada falha, o diagnóstico e a mudança de
> rota estão documentados em **[EXPERIMENTS.md](EXPERIMENTS.md)**

## O problema, em termos de produto

- **Objetivo:** não perder nenhuma carta da coleção → métrica-norte é **recall**.
- **Humano no loop:** o modelo rankeia candidatos, a revisão humana confirma —
  então tolera-se precisão menor em troca de cobertura.
- **Restrições:** custo zero, treino local em GPU
  de consumo (RTX 2060, 6GB).

## A jornada, em uma frase

CLIP zero-shot → embeddings + regressão logística → fine-tuning (2 e 1 blocos)
→ diagnóstico de fome de dados → mudança de backbone (SigLIP2) + mais dados.
Cada passo com hipótese, métrica e decisão registradas em
[EXPERIMENTS.md](EXPERIMENTS.md).

## Stack técnica

- **Dados:** API gratuita da [TCGdex](https://tcgdex.dev) (sem chave)
- **Visão:** open_clip (CLIP ViT-B-32 → SigLIP2), PyTorch (GPU/CUDA)
- **ML clássico:** scikit-learn (regressão logística, validação cruzada)
- **Manipulação:** pandas, numpy

## Estrutura do repositório

| Caminho | Papel |
|---|---|
| `src/tcgdex_utils.py` | Acesso à API (módulo compartilhado) |
| `src/ingestao.py` | Baixa catálogo + imagens dos sets |
| `src/montar_gabarito.py` | Consolida os rótulos manuais |
| `src/extrair_embeddings*.py` | Extrai embeddings (CLIP / SigLIP2) |
| `src/treinar*.py` | Treino (logística / fine-tuning) |
| `src/avaliar.py` | Métricas (recall/precision por top-K) |
| `data/labels/positivos_ids.csv` | Os rótulos manuais (fonte do projeto) |

## Como rodar

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python src/ingestao.py          # popula catálogo e imagens

> Nota sobre GPU: o `requirements.txt` fixa a build CUDA do PyTorch. Para rodar
> em CPU, instale o torch com `--index-url https://download.pytorch.org/whl/cpu`.

## Status atual

🚧 **Em progresso.** Baseline (CLIP zero-shot e embeddings+logística) avaliado e
diagnosticado como insuficiente por limite de representação e fome de dados.
Em andamento: troca para SigLIP2 e expansão da base rotulada. Veja o
[EXPERIMENTS.md](EXPERIMENTS.md) para o estado detalhado.