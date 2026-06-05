# pokemon-sleep-classifier

Classificador de imagens para identificar cartas de Pokémon TCG onde o Pokémon
está de **olhos fechados**, com o objetivo de montar a lista de compras de uma
coleção temática. Produto de dados com humano no loop: o modelo gera candidatos,
a revisão humana confirma.

## Contexto do problema
Triar manualmente +15.000 cartas do TCG é inviável. Este projeto usa o embedding
do CLIP como extrator de features e um classificador treinado sobre os rótulos
para ranquear cartas por probabilidade de "olhos fechados", priorizando **recall**
(não perder nenhuma carta) sobre precision.

## Dados
- Fonte: API gratuita da [TCGdex](https://tcgdex.dev) (sem chave).
- 7 sets ingeridos (~1.285 cartas de Pokémon): WHT, SVP, MEW, SCR, TEF, PAL (era SV)
  e BST (era SWSH).
- Gabarito de positivos rotulado à mão em `data/labels/`.

## Estrutura
- `src/tcgdex_utils.py` — módulo compartilhado de acesso à API (sem execução direta).
- `src/ingestao.py` — baixa catálogo + imagens dos sets definidos no dicionário `SETS`.
- `src/clip_scoring.py` — baseline zero-shot: pontua cada imagem contra prompts.
- `src/montar_gabarito.py` — resolve os positivos (set, número) -> card_id.
- `src/avaliar.py` — mede recall/precision por top-K contra o gabarito.
- `src/folha_revisao.py` — gera HTML para revisão humana de rótulos.
- `src/diagnosticos/` — scripts exploratórios pontuais.

## Ordem de execução
1. `python src/ingestao.py` — popula `data/raw/catalogo_completo.csv` e `data/images/`.
2. `python src/montar_gabarito.py` — gera `data/labels/positivos.csv`.
3. `python src/clip_scoring.py` — gera `data/raw/scores_clip.csv`.
4. `python src/avaliar.py` — imprime as métricas.

## Setup
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Status
Baseline CLIP zero-shot avaliado (recall insuficiente em arte estilizada).
Em desenvolvimento: classificador sobre embeddings do CLIP com validação cruzada.