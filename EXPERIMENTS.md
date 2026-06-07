# Jornada de Experimentos — Classificador de Pokémon de Olhos Fechados

Este documento registra o caminho técnico do projeto: as hipóteses testadas,
os resultados medidos, os diagnósticos e as decisões de mudança de rota.
A intenção é documentar o **raciocínio**, não só o resultado final.

## Contexto do problema

Classificar cartas de Pokémon TCG onde o Pokémon está de **olhos fechados**,
para montar a lista de compras de uma coleção temática. Produto com humano no
loop: o modelo rankeia candidatos, a revisão humana confirma. Por isso a
métrica-norte é **recall** (não perder positivos), tolerando precisão menor.

**Dados iniciais:** ~1.257 cartas de 7 sets (TCGdex API, gratuita), com 44 positivos
rotulados à mão. Prevalência ~3,5% — problema fortemente desbalanceado.

**Restrições:** custo zero (só ferramentas gratuitas), treino local em
RTX 2060 (6GB).

---

## Experimento 1 — CLIP zero-shot

- **Hipótese:** o CLIP (modelo visão-linguagem pré-treinado) saberia comparar
  cada carta com as frases "olhos fechados" vs "olhos abertos" sem treino.
- **Setup:** CLIP ViT-B-32, scoring por similaridade imagem-texto. Dois pares
  de prompts testados ("olhos fechados" e "dormindo").
- **Resultado:** para capturar todos os positivos seria preciso revisar ~99%
  do catálogo. "Dormindo" superou levemente "olhos fechados", mas ambos fracos.
- **Diagnóstico:** o CLIP acerta os "dorminhocos famosos" (Snorlax, etc.) porque
  associa o personagem ao sono, não porque enxerga o olho. Em Pokémon sem essa
  associação cultural, erra. O sinal "olhos fechados" não é acessível por
  comparação direta com texto.
- **Decisão:** abandonar zero-shot puro; testar um classificador treinado
  sobre os embeddings do CLIP.

## Experimento 2 — Embeddings do CLIP + Regressão Logística

- **Hipótese:** a informação "olhos fechados" estaria no embedding (vetor de
  512 números), e uma logística a recuperaria, mesmo que o prompt não a captasse.
- **Setup:** embeddings do CLIP ViT-B-32 + regressão logística com
  `class_weight='balanced'`, validação stratified 5-fold.
- **Resultado:** PR-AUC 0,206 (chão aleatório ~0,035). "K para recall 100%" =
  1132 de 1257 (90% do catálogo). Melhora marginal sobre o zero-shot.
- **Diagnóstico:** inspeção visual dos positivos pior rankeados (Skitty,
  Wooper, Mienfoo — todos com olho claramente fechado e Pokémon centralizado)
  mostrou que o modelo erra até casos óbvios. Conclusão: o embedding do CLIP
  **não codifica "olhos fechados" de forma recuperável** — não é problema de
  cena cheia, é limite da representação. Fronteira linear não separa o que
  não está no espaço.
- **Decisão:** o teto da representação congelada foi atingido. Partir para
  fine-tuning (ajustar os pesos da rede, não só usar o embedding pronto).

## Experimento 3 — Fine-tuning do CLIP (2 blocos descongelados)

- **Hipótese:** descongelar as últimas camadas do CLIP permitiria à rede
  reorganizar a representação para destacar "olhos fechados".
- **Setup:** CLIP ViT-B-32, blocos 10-11 + cabeça binária descongelados,
  `BCEWithLogitsLoss` com pos_weight≈28, Adam lr=1e-5, early stopping,
  split único treino/val/teste (decisão consciente de prova de conceito).
- **Resultado:** perda de validação atingiu o fundo na **época 2** e explodiu
  (de 1.23 para 3.27). Overfitting clássico. "K para recall 100%" ≈ 143/189
  no teste (ruidoso, só 7 positivos no teste).
- **Diagnóstico:** overfitting precoce — a rede decorou os 30 positivos de
  treino em vez de generalizar. Causa candidata: capacidade excessiva (muitos
  pesos livres) OU fome de dados (poucos positivos). Hipóteses empatadas.
- **Decisão:** experimento barato para desempatar — reduzir capacidade
  (1 bloco) e ver se o overfitting alivia.

## Experimento 4 — Fine-tuning do CLIP (1 bloco descongelado)

- **Hipótese:** se a causa do overfitting fosse capacidade, descongelar só 1
  bloco migraria o fundo da validação para mais tarde.
- **Setup:** idêntico ao Exp. 3, mudando só `blocos_descongelados=2 → 1`.
  Rodado 2x (a segunda confirmou aleatoriedade não-fixada no treino).
- **Resultado:** fundo da validação continuou na **época 2-3** nas duas
  rodadas. Comportamento de overfitting idêntico ao de 2 blocos.
- **Diagnóstico:** reduzir capacidade NÃO aliviou — em 3 rodadas no total
  (Exp. 3 e 4), o overfitting precoce é robusto. Diagnóstico desempatado:
  o gargalo é **fome de dados**, não capacidade. 30 positivos de treino é
  pouco para a rede aprender o conceito em vez de memorizar.
- **Decisão:** atacar a raiz por duas frentes — (1) rotular mais dados,
  (2) trocar o backbone por um melhor em detalhes finos (SigLIP2), que pode
  ter no embedding o sinal que o CLIP descartava.

## Experimento 5 — SigLIP2 como backbone (em andamento)

- **Hipótese:** o SigLIP2 (treinado para "features densas" e localização)
  preserva o detalhe local "olhos fechados" que o CLIP ViT-B-32 descartava.
- **Setup:** reextrair embeddings com `ViT-B-16-SigLIP2` (gratuito, cabe na
  2060) e re-rodar a logística stratified 5-fold para comparar com o Exp. 2.
- **Resultado:** _(a preencher)_
- **Diagnóstico:** _(a preencher)_
- **Decisão:** _(a preencher)_

---

## Aprendizados transversais

- **Recall vs precision sob desbalanceamento:** com classe rara, acurácia é
  inútil e ROC-AUC é otimista; PR-AUC é a métrica honesta.
- **Validação só é honesta sobre dados não-treinados:** mesma raiz do data
  leakage, do early stopping e do R² dentro-da-amostra. Nunca medir sobre o
  que ajustou o modelo.
- **Diagnóstico por experimento barato:** antes de investir trabalho caro
  (rotular), isolar a causa com um teste de uma linha (1 vs 2 blocos).
- **A curva de perda treino vs validação é o detector de overfitting:** vê-se
  a decoreba acontecer no descolamento das duas curvas.