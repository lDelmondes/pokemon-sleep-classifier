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

## Experimento 5 — SigLIP2 como backbone

- **Hipótese:** o SigLIP2 (treinado para "features densas" e localização)
  preserva o detalhe local "olhos fechados" que o CLIP ViT-B-32 descartava.
- **Setup:** embeddings com `ViT-B-16-SigLIP2` (768 dims, vs 512 do CLIP) +
  mesma regressão logística stratified 5-fold do Exp. 2 (comparação justa).
- **Resultado:** PR-AUC subiu de 0,206 (CLIP) para **0,273** (SigLIP2).
  Precision no top-20 quase dobrou (30% → 55%). "K para recall 100%" caiu de
  1132/1257 (90%) para **730/1257 (58%)**.
- **Diagnóstico:** a troca de backbone melhorou de forma clara — confirma que
  parte do gargalo ERA a representação (o CLIP descartava o sinal). Mas o recall
  ainda é insuficiente (48% no top-100): a fome de dados continua sendo o outro
  gargalo. As duas frentes (modelo + dados) são complementares, não excludentes.
- **Decisão:** manter o SigLIP2 como backbone e atacar a segunda frente —
  rotular mais cartas para dar à logística/fine-tuning material suficiente.

## Experimento 6 — Mais dados (44 → 122 positivos) sobre SigLIP2

- **Hipótese:** o diagnóstico de falta de dados (Exp. 3-4) indicava que mais
  positivos destravariam o desempenho. Combinado com a melhor representação do
  SigLIP2 (Exp. 5), as duas frentes deveriam se somar.
- **Setup:** rotulagem manual de 7 sets novos (Scarlet & Violet, Twilight
  Masquerade, Surging Sparks, Evolving Skies, Brilliant Stars, Lost Origin,
  Crown Zenith), cobertura total por set. Base saltou de 44 para 122 positivos
  (2.611 cartas, prevalência ~4,7%). Distribuição de era equilibrada (64 SV /
  45 SWSH, contra 3 SWSH antes). Mesma logística stratified 5-fold sobre
  embeddings SigLIP2.
- **Resultado:** PR-AUC 0,273 → **0,291** (leve alta). Precision no topo subiu
  forte: top-20 de 55% → **70%**, top-44 de 32% → **48%**. Porém o recall na
  cauda continuou fraco (top-100 captura ~30% dos 122 positivos), e o "K para
  recall 100%" é de 1959/2611 (frágil a outliers — um único positivo difícil
  no fundo dispara a métrica).
- **Diagnóstico:** mais dados ajudou, mas MENOS que o esperado. O ganho
  concentrou-se na precisão do topo (o modelo confia mais e erra menos nos
  casos que já acertava), não na cauda de positivos difíceis. Hipótese: a
  logística sobre embeddings CONGELADOS está perto do teto — o sinal "olhos
  fechados" em arte muito estilizada pode não ser linearmente separável no
  embedding, e mais exemplos não ensinam o que a representação não capta.
  Importante: a comparação direta de recall/K com experimentos anteriores é
  enganosa, pois o número de positivos e o tamanho do catálogo mudaram.
- **Decisão:** testar fine-tuning sobre o SigLIP2. As duas condições que
  faltaram no fine-tuning anterior (Exp. 3-4, que overfittou) estão agora
  presentes: dados suficientes (98 positivos no treino vs. 30) e uma
  representação melhor. Se a logística sobre embeddings congelados está no
  teto, ajustar os pesos do backbone é o caminho para capturar a cauda difícil

## Experimento 7 — Fine-tuning sobre SigLIP2 (122 positivos)

- **Hipótese:** com as duas condições finalmente presentes (representação do
  SigLIP2 + 122 positivos), o fine-tuning teria material para capturar a cauda
  difícil que a logística não pegava.
- **Setup:** fine-tuning do `ViT-B-16-SigLIP2`, 2 blocos + norm + attn_pool +
  cabeça binária descongelados. BCEWithLogitsLoss com pos_weight≈20, Adam
  lr=1e-5, early stopping, split estratificado (86 pos treino / 18 val / 18 teste).
- **Resultado:** a curva treino/validação melhorou qualitativamente — o fundo
  da validação migrou da época 2 (fine-tunings anteriores) para a **época 4**,
  caindo de forma consistente antes de descolar. No teste: top-50 captura
  **78%** dos positivos, top-80 captura **89%**. Primeiro modelo genuinamente
  útil do projeto — uma triagem que corta ~80% do trabalho mantendo a maioria
  dos positivos.
- **Diagnóstico:** o fine-tuning "pegou" — confirma que as duas frentes (dados
  + representação) eram ambas necessárias. Ainda há overfitting residual (a
  validação dispara após a época 4), sinal de que mais dados levaria o ganho
  além. A cauda 100% continua cara (revisar 175/392 para pegar todos os 18).

### Iteração de qualidade de dados (inspeção da cauda)

A inspeção visual dos positivos pior rankeados pelo modelo virou uma auditoria
do gabarito e revelou três tipos de erro:
- **Rótulos errados (removidos):** Kirlia (olho fechado era de um humano, não
  do Pokémon) e Solrock (marcado pela pose, não tinha olho fechado).
- **Definição de alvo refinada:** os Hisuian Growlithe tinham o olho *coberto
  pelo pelo* (não-visível ≠ fechado). Nova sub-regra: "olho visivelmente
  fechado de um Pokémon" — olho coberto/suposto não conta. Removidos os 3.
- **Cauda irredutível (mantida):** Zeraora VMAX (quem dorme é um Pachirisu
  minúsculo em cima), Applin e Clefairy (múltiplos Pokémon, só alguns de olho
  fechado), Oranguru/Dracozolt rainbow (full-art estilizada). Positivos
  legítimos que definem o teto do que o modelo consegue.

Gabarito limpo de 122 → **117 positivos**. Aprendizado: menos dados corretos
valem mais que mais dados inconsistentes; a definição do alvo é viva, refinada
à luz da evidência. Limitação registrada: cartas com múltiplos Pokémon (só
alguns positivos) são ambíguas para um classificador de imagem única.

**Decisão:** 2 blocos adotado como config de referência (top-80 recall 89%,
fundo val época 4). Comparado contra 3 blocos no Exp. 8.

### Exp. 8 — Fine-tuning SigLIP2, 3 blocos, gabarito limpo (117 pos)
- **Hipótese:** mais capacidade descongelada melhora (se 2 blocos eram pouca
  capacidade) OU overfitta mais cedo (se o gargalo é dado).
- **Resultado:** fundo do val na época 2 (vs época 4 com 2 blocos); top-80 recall
  83%, top-10 28%/50%. Sem ganho acima do ruído (teste com só 18 positivos;
  diferenças de 1–2 positivos).
- **Conclusão:** capacidade extra não ajudou e antecipou o overfitting. Segundo
  desempate (após Exp. 3–4) apontando **fome de dados** como gargalo, não
  capacidade. 3 blocos descartado; 2 blocos mantido. Próxima alavanca: mais dados.

### Exp. 9 — Fine-tuning SigLIP2, 2 blocos, gabarito ampliado (212 pos)
- **Mudanca:** rotulados 16 sets novos (B&W ate SM, 5 eras), gabarito 117 -> 212.
- **Hipotese:** se o gargalo era fome de dados (Exp. 3-4, 7-8), ~2x positivos
  eleva o teto (PR-AUC).
- **Resultado:** PR-AUC TESTE = 0.615 (vs 0.291 da logistica/122 do Exp. 6 — 2x+).
  Teste: 32 pos em 680 cartas. Top-10 precision 80%, top-80 recall 78%.
  Curva: fundo val epoca 2, overfitting mais contido que Exp. 8 (val max 1.74 vs 2.5+).
- **Conclusao:** hipotese da fome de dados CONFIRMADA. Mais dados foi a alavanca,
  nao capacidade nem backbone. Curva ainda overfitta -> fome residual, teto nao
  saturado. Recall@K nao comparavel a rodadas anteriores (teste mudou de tamanho);
  PR-AUC e a metrica de comparacao valida daqui pra frente.

  ### Exp. 10 — Fine-tuning SigLIP2, 2 blocos, 212 pos + weight decay 0.01 (AdamW)
- **Hipotese:** L2 contem o overfitting do Exp. 9; se ajudar, PR-AUC sobe (>=0.66).
- **Resultado:** PR-AUC TESTE = 0.581 (vs 0.615 do Exp. 9). Fundo val epoca 3
  (vs 2), overfitting marginalmente mais tarde mas PR-AUC nao melhorou — piorou
  levemente. Top-K nao comparavel (ruido, 32 pos no teste).
- **Conclusao:** weight decay 0.01 NAO ajudou. Confirma que o teto residual e de
  DADOS, nao de regularizacao. Reverter para Adam sem decay (Exp. 9 e a config
  vigente). Proxima alavanca real: mais dados.
  
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