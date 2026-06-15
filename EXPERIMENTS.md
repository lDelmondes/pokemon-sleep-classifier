# Jornada de Experimentos — Classificador de Pokémon de Olhos Fechados

Registro técnico do projeto: as hipóteses testadas, os resultados medidos, os diagnósticos e as decisões de mudança de rota. A intenção é documentar o **raciocínio**, incluindo os desafios e as conclusões revistas, não só o resultado final.

---

## Resumo

O objetivo é triar um universo de ~12 mil cartas de Pokémon TCG atrás das que mostram o Pokémon **de olhos fechados** (um proxy operacional para "dormindo"), gerando uma lista de compras priorizada para revisão humana. O problema é fortemente desbalanceado (~5% de positivos) e de custo zero (ferramentas gratuitas, treino local em RTX 2060).

A jornada passou por três fases. Primeiro, embeddings congelados (CLIP, depois SigLIP2) sobre uma regressão logística — útil para diagnosticar, mas limitado pela representação. Segundo, fine-tuning do backbone combinado com mais dados rotulados à mão — que revelou uma lição contraintuitiva: **mais dados nem sempre ajuda** (um lote redundante chegou a degradar o modelo). Terceiro, a descoberta central: o gargalo nunca foi dados nem capacidade, e sim **resolução** — um olho fechado ocupa ~1-2% da carta e desaparecia ao redimensionar a imagem para 224×224. **Recortar a arte** (descartar moldura e texto) foi a maior alavanca do projeto, elevando o PR-AUC de ~0.49 para ~0.68. Afinações posteriores renderam ganho nulo, sinalizando o teto útil do problema.

O resultado é um MVP funcional: um classificador por transfer learning que generaliza para eras nunca vistas no treino e produz uma lista cujas candidatas de topo têm precisão de 100% (top 50) a ~94% (top 300), validada à mão.

---

## Introdução

### O problema e a decisão do proxy

A motivação é real: montar uma coleção temática de cartas de Pokémon TCG em que o Pokémon aparece **dormindo**. O universo relevante passa de 12 mil cartas, e a triagem manual é inviável.

"Dormindo", porém, é um alvo difícil para um classificador de imagem. É um conceito de **cena**: exigiria o modelo inferir um *estado* a partir de pose, ambiente, contexto e expressão combinados — e ensinar (ou validar) isso num MVP consumiria tempo e dados desproporcionais. A decisão de produto foi trocar o alvo conceitual por um **proxy visual, local e objetivamente rotulável**: **olhos fechados**. Olho fechado é um detalhe na arte que um classificador captura bem e correlaciona forte com "dormindo".

O proxy é imperfeito por construção — há Pokémon de olhos fechados acordados (rindo, piscando) e dorminhocos de olhos abertos —, mas troca um problema vago por um mensurável, e é uma **escolha permanente** deste produto, não um atalho temporário. Esta é a decisão de enquadramento mais fundamental do projeto, e tudo o que segue se apoia nela.

### Natureza do produto e métrica-norte

A solução é uma **ferramenta de curadoria, não um classificador autônomo**: o modelo rankeia as cartas por probabilidade de olhos fechados e uma pessoa confirma as candidatas do topo (**humano-no-loop**). Por isso a métrica que guia o projeto é o **recall** — não perder positivos pesa mais que um falso positivo ocasional, que o humano descarta na revisão. Para comparar configurações entre si, a métrica é o **PR-AUC** (average precision), honesta sob forte desbalanceamento.

### Ponto de partida

O projeto começou modesto: ~1.257 cartas de 7 sets (via a API gratuita da TCGdex), com 44 positivos rotulados à mão (prevalência ~3,5%). Ao longo da jornada, tanto o gabarito quanto o universo cresceram — até os 311 positivos e os dois universos de dados descritos na metodologia.

---

## Metodologia

### Fonte de dados

Todas as cartas e imagens vêm da **TCGdex** (`api.tcgdex.net`), API pública e gratuita. Os IDs de carta da TCGdex (ex.: `sm6-89`) são a chave única em todo o pipeline. Apenas cartas com `category == "Pokemon"` entram.

### Definição do alvo e regras de rotulagem

O alvo é "olho **visivelmente fechado** de um **Pokémon**". As regras, refinadas à luz da evidência ao longo do projeto:

- Olho canonicamente fechado conta; olho **coberto** por pelo ou sombra **não** conta (não-visível ≠ fechado).
- Pose de dormir **sem** olho fechado visível não conta; piscar (um olho aberto) não conta.
- Múltiplos Pokémon na arte: conta se **ao menos um** tem ambos os olhos fechados.
- Olho cerrado/sorrindo/apertado (`^^`, expressão de esforço) **conta** — a pálpebra está unida.
- Olho **estilizado permanente de espécie** (espirais do Spinda, costuras do Mimikyu) **não** conta: é design fixo, não expressão momentânea.

A distinção entre **expressão** (momentânea, conta) e **design fixo** (não conta) foi uma sub-regra firmada após auditorias do gabarito.

### Dois universos de dados

O projeto trabalha com dois conjuntos — confundi-los causou um bug real (ver Exp. 16):

- **Universo de desenvolvimento** — 5.836 cartas, 38 sets, 5 eras (Black & White até Scarlet & Violet), com **311 positivos** rotulados à mão. Onde o modelo treina, valida e é testado.
- **Universo de inferência** — 12.249 cartas, 6 eras (as 5 acima + Mega), onde o modelo é aplicado para gerar a lista de compras.

### Protocolo de avaliação

- **Split** estratificado 70/15/15 (treino/validação/teste) com `random_state=42` fixo, mantendo a proporção de positivos nos três grupos.
- O treino **não** é seedado, então rodadas idênticas variam (~±0.02 de PR-AUC). **Por isso toda comparação de configurações exige ≥3 rodadas** — um único número é ruído. Esta é a lição metodológica transversal do projeto.
- **Métrica de comparação:** PR-AUC (average precision). **Métrica-norte de produto:** recall.
- **Desbalanceamento** tratado via `pos_weight = n_neg/n_pos` na `BCEWithLogitsLoss`.

### Backbone e treino (configuração final)

`ViT-B-16-SigLIP2` (768 dims) via `open_clip`, com os **2 últimos blocos** + `norm` + `attn_pool` + uma cabeça `Linear(768, 1)` descongelados; o resto congelado. Otimizador Adam, `lr=1e-5`, early stopping pela perda de validação. Antes de entrar na rede, cada imagem passa pelo **recorte adaptativo da arte** (ver Fase 3).

### Ambiente

Python 3.12, PyTorch + CUDA, treino local em RTX 2060 (6GB). Uma rodada de treino com recorte no catálogo completo leva ~35-40 min.

---

# Experimentos

## Fase 1 — A representação congelada (Exp. 1–6)

Esta fase usa embeddings de modelos pré-treinados *sem* ajustar seus pesos, sobre uma regressão logística. O objetivo era diagnosticar onde estava o sinal — e descobrir que a representação congelada tinha um teto.

### Experimento 1 — CLIP zero-shot

- **Hipótese:** o CLIP (modelo visão-linguagem pré-treinado) saberia comparar cada carta com as frases "olhos fechados" vs "olhos abertos" sem treino.
- **Setup:** CLIP ViT-B-32, scoring por similaridade imagem-texto. Dois pares de prompts testados ("olhos fechados" e "dormindo").
- **Resultado:** para capturar todos os positivos seria preciso revisar ~99% do catálogo. "Dormindo" superou levemente "olhos fechados", mas ambos fracos.
- **Diagnóstico:** o CLIP acerta os "dorminhocos famosos" (Snorlax, etc.) provavelmente porque associa o personagem ao sono, não porque enxerga o olho. Em Pokémon sem essa associação cultural, erra. O sinal "olhos fechados" não é acessível por comparação direta com texto.
- **Decisão:** abandonar zero-shot puro; testar um classificador treinado sobre os embeddings do CLIP.

### Experimento 2 — Embeddings do CLIP + Regressão Logística

- **Hipótese:** a informação "olhos fechados" estaria no embedding (vetor de 512 números), e uma logística a recuperaria, mesmo que o prompt não a captasse.
- **Setup:** embeddings do CLIP ViT-B-32 + regressão logística com `class_weight='balanced'`, validação stratified 5-fold.
- **Resultado:** PR-AUC 0,206 (chão aleatório ~0,035). "K para recall 100%" = 1132 de 1257 (90% do catálogo). Melhora marginal sobre o zero-shot.
- **Diagnóstico:** inspeção visual dos positivos pior rankeados (Skitty, Wooper, Mienfoo — todos com olho claramente fechado e Pokémon centralizado) mostrou que o modelo erra até casos óbvios. Conclusão: o embedding do CLIP **não codifica "olhos fechados" de forma recuperável** — não é problema de cena cheia, é limite da representação. Fronteira linear não separa o que não está no espaço.
- **Decisão:** o teto da representação congelada foi atingido. Partir para fine-tuning (ajustar os pesos da rede, não só usar o embedding pronto).

### Experimento 3 — Fine-tuning do CLIP (2 blocos descongelados)

- **Hipótese:** descongelar as últimas camadas do CLIP permitiria à rede reorganizar a representação para destacar "olhos fechados".
- **Setup:** CLIP ViT-B-32, blocos 10-11 + cabeça binária descongelados, `BCEWithLogitsLoss` com pos_weight≈28, Adam lr=1e-5, early stopping, split único treino/val/teste (decisão consciente de prova de conceito).
- **Resultado:** perda de validação atingiu o fundo na **época 2** e explodiu (de 1.23 para 3.27). Overfitting clássico. "K para recall 100%" ≈ 143/189 no teste (ruidoso, só 7 positivos no teste).
- **Diagnóstico:** overfitting precoce — a rede decorou os 30 positivos de treino em vez de generalizar. Causa candidata: capacidade excessiva (muitos pesos livres) OU fome de dados (poucos positivos). Hipóteses empatadas.
- **Decisão:** experimento barato para desempatar — reduzir capacidade (1 bloco) e ver se o overfitting alivia.

### Experimento 4 — Fine-tuning do CLIP (1 bloco descongelado)

- **Hipótese:** se a causa do overfitting fosse capacidade, descongelar só 1 bloco migraria o fundo da validação para mais tarde.
- **Setup:** idêntico ao Exp. 3, mudando só `blocos_descongelados=2 → 1`. Rodado 2x (a segunda confirmou aleatoriedade não-fixada no treino).
- **Resultado:** fundo da validação continuou na **época 2-3** nas duas rodadas. Comportamento de overfitting idêntico ao de 2 blocos.
- **Diagnóstico:** reduzir capacidade NÃO aliviou — em 3 rodadas no total (Exp. 3 e 4), o overfitting precoce é robusto. Diagnóstico desempatado: o gargalo é **fome de dados**, não capacidade. 30 positivos de treino é pouco para a rede aprender o conceito em vez de memorizar.
- **Decisão:** atacar a raiz por duas frentes — (1) rotular mais dados, (2) trocar o backbone por um melhor em detalhes finos (SigLIP2), que pode ter no embedding o sinal que o CLIP descartava.

### Experimento 5 — SigLIP2 como backbone

- **Hipótese:** o SigLIP2 (treinado para "features densas" e localização) preserva o detalhe local "olhos fechados" que o CLIP ViT-B-32 descartava.
- **Setup:** embeddings com `ViT-B-16-SigLIP2` (768 dims, vs 512 do CLIP) + mesma regressão logística stratified 5-fold do Exp. 2 (comparação justa).
- **Resultado:** PR-AUC subiu de 0,206 (CLIP) para **0,273** (SigLIP2). Precision no top-20 quase dobrou (30% → 55%). "K para recall 100%" caiu de 1132/1257 (90%) para **730/1257 (58%)**.
- **Diagnóstico:** a troca de backbone melhorou de forma clara — confirma que parte do gargalo ERA a representação (o CLIP descartava o sinal). Mas o recall ainda é insuficiente (48% no top-100): a fome de dados continua sendo o outro gargalo. As duas frentes (modelo + dados) são complementares, não excludentes.
- **Decisão:** manter o SigLIP2 como backbone e atacar a segunda frente — rotular mais cartas para dar à logística/fine-tuning material suficiente.

### Experimento 6 — Mais dados (44 → 122 positivos) sobre SigLIP2

- **Hipótese:** o diagnóstico de falta de dados (Exp. 3-4) indicava que mais positivos destravariam o desempenho. Combinado com a melhor representação do SigLIP2 (Exp. 5), as duas frentes deveriam se somar.
- **Setup:** rotulagem manual de 7 sets novos (Scarlet & Violet Base Set, Twilight Masquerade, Surging Sparks, Evolving Skies, Brilliant Stars, Lost Origin, Crown Zenith), cobertura total por set. Base saltou de 44 para 122 positivos (2.611 cartas, prevalência ~4,7%). Distribuição de era equilibrada (64 SV / 45 SWSH, contra 3 SWSH antes). Mesma logística stratified 5-fold sobre embeddings SigLIP2.
- **Resultado:** PR-AUC 0,273 → **0,291** (leve alta). Precision no topo subiu forte: top-20 de 55% → **70%**, top-44 de 32% → **48%**. Porém o recall na cauda continuou fraco (top-100 captura ~30% dos 122 positivos), e o "K para recall 100%" é de 1959/2611 (frágil a outliers — um único positivo difícil no fundo dispara a métrica).
- **Diagnóstico:** mais dados ajudou, mas MENOS que o esperado. O ganho concentrou-se na precisão do topo (o modelo confia mais e erra menos nos casos que já acertava), não na cauda de positivos difíceis. Hipótese: a logística sobre embeddings CONGELADOS está perto do teto — o sinal "olhos fechados" em arte muito estilizada pode não ser linearmente separável no embedding, e mais exemplos não ensinam o que a representação não capta. Importante: a comparação direta de recall/K com experimentos anteriores é enganosa, pois o número de positivos e o tamanho do catálogo mudaram.
- **Decisão:** testar fine-tuning sobre o SigLIP2. As duas condições que faltaram no fine-tuning anterior (Exp. 3-4, que overfittou) estão agora presentes: dados suficientes (98 positivos no treino vs. 30) e uma representação melhor. Se a logística sobre embeddings congelados está no teto, ajustar os pesos do backbone é o caminho para capturar a cauda difícil.

## Fase 2 — Fine-tuning e a saga dos dados (Exp. 7–12)

Com uma representação melhor e mais rótulos, o fine-tuning finalmente "pegou". Esta fase rendeu o primeiro modelo útil — e a lição mais contraintuitiva do projeto: mais dados pode piorar.

### Experimento 7 — Fine-tuning sobre SigLIP2 (122 positivos)

- **Hipótese:** com as duas condições finalmente presentes (representação do SigLIP2 + 122 positivos), o fine-tuning teria material para capturar a cauda difícil que a logística não pegava.
- **Setup:** fine-tuning do `ViT-B-16-SigLIP2`, 2 blocos + norm + attn_pool + cabeça binária descongelados. BCEWithLogitsLoss com pos_weight≈20, Adam lr=1e-5, early stopping, split estratificado (86 pos treino / 18 val / 18 teste).
- **Resultado:** a curva treino/validação melhorou qualitativamente — o fundo da validação migrou da época 2 (fine-tunings anteriores) para a **época 4**, caindo de forma consistente antes de descolar. No teste: top-50 captura **78%** dos positivos, top-80 captura **89%**. Primeiro modelo genuinamente útil do projeto — uma triagem que corta ~80% do trabalho mantendo a maioria dos positivos.
- **Diagnóstico:** o fine-tuning "pegou" — confirma que as duas frentes (dados + representação) eram ambas necessárias. Ainda há overfitting residual (a validação dispara após a época 4), sinal de que mais dados levaria o ganho além. A cauda 100% continua cara (revisar 175/392 para pegar todos os 18).

**Iteração de qualidade de dados (inspeção da cauda).** A inspeção visual dos positivos pior rankeados virou uma auditoria do gabarito e revelou três tipos de erro:

- **Rótulos errados (removidos):** Kirlia (olho fechado era de um humano, não do Pokémon) e Solrock (marcado pela pose, não tinha olho fechado).
- **Definição de alvo refinada:** os Hisuian Growlithe tinham o olho *coberto pelo pelo* (não-visível ≠ fechado). Nova sub-regra: "olho visivelmente fechado de um Pokémon" — olho coberto/suposto não conta. Removidos os 3.
- **Cauda irredutível (mantida):** Zeraora VMAX (quem dorme é um Pachirisu minúsculo em cima), Applin e Clefairy (múltiplos Pokémon, só alguns de olho fechado), Oranguru/Dracozolt rainbow (full-art estilizada). Positivos legítimos que definem o teto do que o modelo consegue.

Gabarito limpo de 122 → **117 positivos**. Aprendizado: menos dados corretos valem mais que mais dados inconsistentes; a definição do alvo é viva, refinada à luz da evidência. Limitação registrada: cartas com múltiplos Pokémon (só alguns positivos) são ambíguas para um classificador de imagem única.

- **Decisão:** 2 blocos adotado como config de referência (top-80 recall 89%, fundo val época 4). Comparado contra 3 blocos no Exp. 8.

### Experimento 8 — Fine-tuning SigLIP2, 3 blocos, gabarito limpo (117 pos)

- **Hipótese:** mais capacidade descongelada melhora (se 2 blocos eram pouca capacidade) OU overfitta mais cedo (se o gargalo é dado).
- **Resultado:** fundo do val na época 2 (vs época 4 com 2 blocos); top-80 recall 83%, top-10 28%/50%. Sem ganho acima do ruído (teste com só 18 positivos; diferenças de 1–2 positivos).
- **Conclusão:** capacidade extra não ajudou e antecipou o overfitting. Segundo desempate (após Exp. 3–4) apontando **fome de dados** como gargalo, não capacidade. 3 blocos descartado; 2 blocos mantido. Próxima alavanca: mais dados.

### Experimento 9 — Fine-tuning SigLIP2, 2 blocos, gabarito ampliado (212 pos)

- **Mudança:** rotulados 16 sets novos (B&W até SM, 5 eras), gabarito 117 → 212.
- **Hipótese:** se o gargalo era fome de dados (Exp. 3-4, 7-8), ~2x positivos eleva o teto (PR-AUC).
- **Resultado:** PR-AUC TESTE = 0.615 (vs 0.291 da logística/122 do Exp. 6 — 2x+). Teste: 32 pos em 680 cartas. Top-10 precision 80%, top-80 recall 78%. Curva: fundo val época 2, overfitting mais contido que Exp. 8 (val max 1.74 vs 2.5+).
- **Conclusão:** hipótese da fome de dados CONFIRMADA. Mais dados foi a alavanca, não capacidade nem backbone. Curva ainda overfitta → fome residual, teto não saturado. Recall@K não comparável a rodadas anteriores (teste mudou de tamanho); PR-AUC é a métrica de comparação válida daqui pra frente.

> **Nota:** esta conclusão seria **revista** no Exp. 12 — o 0.615 era uma rodada única, e acabou se revelando o topo da variância, não a média.

### Experimento 10 — Fine-tuning SigLIP2, 2 blocos, 212 pos + weight decay 0.01 (AdamW)

- **Hipótese:** L2 contém o overfitting do Exp. 9; se ajudar, PR-AUC sobe (≥0.66).
- **Resultado:** PR-AUC TESTE = 0.581 (vs 0.615 do Exp. 9). Fundo val época 3 (vs 2), overfitting marginalmente mais tarde mas PR-AUC não melhorou — piorou levemente. Top-K não comparável (ruído, 32 pos no teste).
- **Conclusão:** weight decay 0.01 NÃO ajudou. Confirma que o teto residual é de DADOS, não de regularização. Reverter para Adam sem decay (Exp. 9 é a config vigente). Próxima alavanca real: mais dados.

### Experimento 11 — Fine-tuning SigLIP2, 2 blocos, gabarito ampliado (278 pos)

- **Mudança:** rotulados +8 sets (lote 3: FST, ASR, LOT, TEU, PRC, PHF, LTR, DRX), todos de eras JÁ presentes (SWSH/SM/XY/BW). Gabarito 212 → 278; catálogo 4.530 → 5.802 cartas.
- **Hipótese:** se ainda havia fome de dados (Exp. 9), +66 positivos sobe o PR-AUC.
- **Resultado:** PR-AUC TESTE = 0.469 / 0.487 / 0.516 em três rodadas (média ~0.49), ABAIXO do 0.615 do Exp. 9. Teste: 42 pos em 871 cartas.
- **Diagnóstico:** resultado contraintuitivo (mais dados, PR-AUC menor) disparou investigação. Pipeline verificado íntegro: gabarito 278 completo sobre o catálogo (sem falsos negativos), imagens presentes, teste bem distribuído (não enviesado para sets difíceis). A queda não era artefato de pipeline. Mas o 0.615 do Exp. 9 era rodada ÚNICA — suspeita de que fosse o topo da variância, não a média.
- **Decisão:** comparar 212 vs 278 com múltiplas rodadas controladas (Exp. 12) antes de concluir qualquer coisa. Lição de método: medir com rodada única é enganoso quando a curva de validação é instável.

### Experimento 12 — Comparação controlada 212 vs 278 (3 rodadas cada)

- **Motivação:** decidir se a queda do Exp. 11 era real ou se o 0.615 do Exp. 9 foi sorte. Treino sem seed fixo (só o split é fixo, rs=42), então cada config foi rodada 3x para estimar a distribuição do PR-AUC.
- **Método:** 212 reproduzido no catálogo filtrado de 4.564 (8 sets do lote 3 removidos via script não-destrutivo); 278 no catálogo completo de 5.802. Backups protegidos (`catalogo_5802.csv`, `positivos_ids_278.csv`), reversão e restauração por script auditável.
- **Resultado:**
  - 212 @ 4.564: 0.525 / 0.545 / 0.577 (média ~0.55). O 0.615 original do Exp. 9 era o TOPO da variância, fora da faixa das três rodadas controladas.
  - 278 @ 5.802: 0.469 / 0.487 / 0.516 (média ~0.49).
  - As faixas NÃO se sobrepõem (pior do 212 = 0.525 > melhor do 278 = 0.516).
- **Conclusão:** 212 (~0.55) é melhor que 278 (~0.49) de forma reproduzível. Adicionar o lote 3 DEGRADOU o modelo, não saturou. **Isto revisa a conclusão do Exp. 9:** "mais dados sempre ajuda" é FALSO para este problema. O lote 1 (eras NOVAS, variação genuína) ajudou; o lote 3 (eras JÁ vistas) prejudicou — pouco sinal positivo novo + ~1.200 negativos novos diluindo. Existe um ponto ótimo de dados antes de 278.
- **Lição de método:** rodada única é enganosa com curva de validação instável; comparar configs exige 3+ rodadas e olhar a distribuição, nunca um ponto.
- **Tensão em aberto:** o modelo é melhor com 212, mas o produto precisa rankear os 5.802 (a coleção quer todas as eras). Resolver no próximo passo.

## Fase 3 — A descoberta do recorte (Exp. 13–16)

A virada do projeto. Uma auditoria mostrou o modelo errando olhos fechados óbvios — e o diagnóstico mudou de "dados/capacidade" para **resolução**. Recortar a arte foi a maior alavanca de todas.

### Experimento 13 — Recorte da arte (resolução como gargalo) — subconjunto Common/Uncommon/Rare

- **Motivação:** auditoria (Exp. anteriores) mostrou que o modelo erra olhos fechados ÓBVIOS (Klefki sm6-89 com prob 0.008). Hipótese: o gargalo não é dado nem capacidade, é RESOLUÇÃO — o olho (~1-2% da carta) some quando a carta inteira é espremida em 224×224.
- **Método:** isolar a variável "recorte". Filtrou-se o catálogo para raridades de layout NORMAL (Common/Uncommon/Rare, onde a arte fica sempre na metade superior) → 4.022 cartas, 237 positivos. Mediu-se baseline (sem recorte) e depois recorte conservador (manter 55% superiores da carta), 3 rodadas cada.
- **Resultado:**
  - Baseline (sem recorte): PR-AUC 0.507 / 0.566 / 0.559 → média ~0.544.
  - Com recorte (55% superior): PR-AUC 0.755 / 0.759 / 0.858 → média ~0.79.
  - Distribuições não se sobrepõem (pior recorte 0.755 >> melhor baseline 0.566).
  - Produto: top-10 precision 90-100%, recall@80 83-92%.
- **Conclusão:** HIPÓTESE CONFIRMADA de forma contundente. O gargalo era resolução do detalhe, não dados nem capacidade. Recortar a arte (jogar fora moldura+texto) dobra a fração útil da imagem e o olho fica visível ao SigLIP. Maior ganho do projeto (+0.25 PR-AUC) veio de uma mudança de **pré-processamento** da imagem, não de mais dados ou hiperparâmetros.
- **Escopo/limite:** testado só em layout normal (Common/Uncommon/Rare). Full-art e special-art (Illustration/Secret/Ultra) ficaram fora; recorte fixo em 55% pode decepar Pokémon nessas. Tratamento delas = próxima fase.
- **Custo:** ~30 min por rodada de treino (crop adiciona processamento de imagem).

### Experimento 14 — Recorte da arte (55%) no catálogo COMPLETO (todas as raridades)

- **Motivação:** Exp. 13 provou o recorte no subconjunto fácil (Common/Uncommon/Rare, ~0.79). Faltava validar no escopo de PRODUTO: catálogo completo, incluindo full-art e special-art (Illustration/Secret/Ultra), que são mais difíceis.
- **Premissa validada antes:** inspeção visual mostrou que o "centro de interesse" da arte (rosto/olho do Pokémon) fica na metade superior mesmo em full-arts. Recorte global simples (55% superior) deve servir para quase toda a base.
- **Gabarito ajustado ao recorte:** revisão visual dos 312 positivos JÁ recortados removeu os que perderam o Pokémon no corte. Só 1 caiu (`swsh12.5-GG22`, full-art com o Pokémon na metade inferior). Gabarito 312 → 311. Confirma a premissa: 99,7% dos positivos têm o olho na metade superior.
- **Método:** treino com recorte de 55% sobre o catálogo completo (4.060 cartas treino, 217 positivos), split estratificado fixo, 3 rodadas.
- **Resultado:**
  - Sem recorte (Exp. 11/auditoria): PR-AUC ~0.49.
  - Com recorte (catálogo completo): 0.690 / 0.666 / 0.678 → média ~0.678.
  - Distribuições não se sobrepõem. Ganho de +0.19, estável (amplitude 0.024).
  - Produto: top-10 precision 80-90%, recall@80 83-89%, recall 100% revisando ~226-285 de 871 (corta ~70% do trabalho com cobertura total).
- **Conclusão:** o recorte resolve o problema no escopo COMPLETO, não só no subconjunto fácil. Bate a meta de 0.70 (melhor rodada 0.690, média 0.678 a um passo). O modelo passou de "investigação travada" para "pronto para produto".
- **Custo:** ~35-40 min por rodada (catálogo completo + crop).

### Experimento 15 — Recorte com corte do topo (remover faixa de nome/HP) — RESULTADO NEGATIVO

- **Hipótese:** a faixa superior da carta (nome do Pokémon, HP, tipo) seria um atalho de SHORTCUT LEARNING — o modelo poderia aprender a "ler o nome" em vez de olhar o olho. Remover essa faixa forçaria o foco na arte e subiria o PR-AUC.
- **Método:** recorte combinado — além do limite inferior em 55% (Exp. 14), cortar também 8.5% do topo (faixa de nome/HP). Fração 8.5% escolhida por inspeção visual (remove o texto sem decepar o rosto, que fica logo abaixo). Catálogo completo, 3 rodadas.
- **Resultado:**
  - Recorte 55% só (Exp. 14): 0.690 / 0.666 / 0.678 → média ~0.678.
  - Recorte 8.5%-55% (corte topo): 0.675 / 0.670 / 0.689 → média ~0.678.
  - IDÊNTICO. Faixas totalmente sobrepostas.
- **Conclusão:** hipótese REFUTADA. Cortar o topo não mudou nada — o texto de nome/HP NÃO era um atalho relevante. O modelo já estava olhando a arte, não lendo o nome. O recorte de 55% (Exp. 14) sozinho já capturava todo o ganho desta alavanca. Resultado negativo valioso: confirma que o sinal aprendido é a arte, e descarta a suspeita de shortcut learning pelo texto.
- **Leitura estratégica:** após o salto grande do recorte (0.49 → 0.68), as afinações finas rendem ~zero. Sinal de proximidade do teto do problema. A questão deixa de ser "qual próxima alavanca" e passa a ser "o modelo já serve ao produto?" (revisar ~180/871 = recall 100%, top-30 precision ~80%).

### Experimento 16 — Recorte adaptativo por raridade

- **Hipótese:** uma janela de recorte por raridade (em vez de 0–55% fixo) melhora o PR-AUC, combinando (a) apertar a base nos layouts normais para remover a caixa "Sleeping Pokémon" e (b) alargar a janela em full-arts, onde a arte sangra e o Pokémon fica mais baixo.
- **Ressalva de desenho (registrada ANTES de rodar):** o dicionário implementado virou ~mecanismo (a) puro. Full-arts (Illustration/Special/Ultra/Hyper/Secret) ficaram em 0.55 — idênticos à produção. Só `Black White Rare` foi a 0.60. A parte (b), que era a promissora, não foi de fato testada; o experimento mede essencialmente "apertar os normais de 0.55→0.48 ajuda?".
- **Setup:** dev set idêntico à âncora — 4060 treino / 871 val / 871 teste, 311 positivos, prevalência ~5.3%, `pos_weight=17.7`, split estratificado seed 42. Única variável vs âncora = janela de recorte por carta (topo fixo 0.085). 3 rodadas.
- **Armadilha de infra encontrada no setup (lição transversal):** a ingestão do universo de inferência havia sobrescrito `catalogo_completo.csv` (5.836 → 12.249), e o `split.py` lia esse arquivo sem distinguir desenvolvimento de inferência. A primeira tentativa treinou com 8324 cartas / `pos_weight=37.4` — ~4.200 negativos de inferência vazados para o treino. Corrigido congelando o universo de desenvolvimento em `catalogo_desenvolvimento.csv` e travando o split nele. Lição: *fonte de dados implícita é dívida — o split deve declarar seu universo, não herdar o arquivo do dia.*
- **Resultados (PR-AUC no dev test):**

  | | R1 | R2 | R3 | Média | Faixa |
  |---|---|---|---|---|---|
  | Âncora (0.55 fixo) | 0.675 | 0.670 | 0.689 | 0.678 | [0.670, 0.689] |
  | Exp.16 (adaptativo) | 0.694 | 0.709 | 0.675 | 0.693 | [0.675, 0.709] |

- **Leitura:** diferença de médias +0.015, mas as faixas se sobrepõem por completo (a âncora chega a 0.689; o adaptativo desce a 0.675). Dentro do ruído de ±0.02 de seed, indistinguíveis. **Resultado: neutro** — nenhuma config é estatisticamente superior à outra.
- **Convergência com Exp.15:** confirma, por outro caminho, que o modelo não se apoia no texto da carta — já olha a arte. Remover mais texto da base (mec. (a)) não move o ponteiro porque não era texto que estava sendo lido. Duas evidências independentes, mesma conclusão.
- **Em aberto:** a hipótese (b) — full-arts com janela > 0.55 — não foi testada e segue válida, não refutada. Candidata natural a experimento futuro.
- **Decisão — adoção sob equivalência (NÃO por métrica):** adotado o recorte adaptativo (na forma testada: normais a 0.48, full-arts em 0.55) como configuração de produção. A justificativa não é ganho de PR-AUC — não há, o Exp.16 é neutro. É decisão de *design* sob equivalência estatística: como o recorte mais agressivo nos normais custa zero de performance, escolho a entrada mais alinhada ao objetivo do produto — máximo de arte, mínimo de moldura/texto. Se o modelo deve decidir pela ARTE, mostrar a ele só a arte é a entrada fiel ao objetivo, esteja o texto sendo usado hoje ou não. Benefício colateral: menor superfície para shortcut learning em distribuições futuras (eras e layouts novos). Explicitamente *não* é correção de um atalho existente — Exp.15 e 16 mostraram que não há.
- **Modelo de produção:** `exp16_adaptativo_1.pt` — a MEDIANA das 3 rodadas (0.694), não o pico (0.709, R2). Escolher o máximo de N rodadas é estimador enviesado para cima e regride à média fora da amostra; adotar a mediana entre instâncias equivalentes evita ancorar a produção no extremo da variância de seed. Coerente com a tese transversal do projeto: 1 número é ruído — por isso 3 rodadas, e por isso não se escolhe o melhor de 3.
- **Validação em produção:** lista de compras regenerada com o `_1`, recorte adaptativo na inferência idêntico ao do treino (mesma fonte de janela, `recorte_rarity.py` — sem train-inference skew, confirmado por gate de cobertura). Precisão das candidatas por contagem manual — top 50: 100% (50/50); top 100: 98% (98/100); top 300: 94% (282/300). Equivalente à âncora (~97% no top 300, `_3`), dentro do ruído já aceito. Ressalva honesta: a contagem da âncora foi estimativa de olho (~4 FP); esta é contagem dura — parte do gap é medição mais rigorosa, não regressão real. Qualidade decai como ladeira suave: os erros se concentram na cauda (score já baixo), onde a revisão humana opera em modo cético — o lugar certo para errar numa ferramenta com humano no loop.

---

## Conclusão

### A pergunta foi respondida

Era possível, com transfer learning sobre um backbone de visão e um conjunto pequeno de rótulos manuais, construir um rankeador que triasse ~12 mil cartas com recall alto o suficiente para servir como ferramenta de curadoria com humano no loop? **Sim.** O modelo final generaliza para 6 eras (uma delas, Mega, nunca vista no treino) e produz uma lista cujas candidatas de topo têm 100% de precisão (top 50) e ~94% (top 300), com a qualidade decaindo suavemente, o comportamento ideal para revisão humana.

### A lição central

O maior aprendizado não é sobre o modelo, é sobre **onde procurar o gargalo**. Por mais de dez experimentos, os suspeitos foram dados (rotular mais) e capacidade (mais/menos camadas). O destravamento veio de um terceiro lugar quase sempre ignorado: o **pré-processamento** — *como* a imagem chega à rede. Um olho fechado de ~1-2% da carta desaparecia no redimensionamento; recortar a arte para concentrar os pixels valeu mais (+0.25 PR-AUC no subconjunto controlado) que qualquer ganho de dados ou arquitetura. **Antes de assumir "preciso de mais dados ou um modelo maior", vale questionar se o sinal está chegando intacto à rede.**

### Aprendizados transversais

- **PR-AUC é a métrica honesta sob desbalanceamento.** Com classe rara, acurácia é inútil e ROC-AUC é otimista.
- **Validação só vale sobre dados não-treinados.** Mesma raiz do data leakage, do early stopping e do R² dentro-da-amostra.
- **Diagnóstico por experimento barato.** Antes de investir trabalho caro (rotular), isolar a causa com um teste mínimo (1 vs 2 blocos).
- **A curva treino vs validação é o detector de overfitting.** A decoreba se vê no descolamento das duas curvas.
- **Variância de inicialização mascara sinal.** Sem seed no treino, uma rodada única pode estar no topo ou no fundo da distribuição; diferenças menores que essa variância não são conclusivas. Comparar configs exige múltiplas rodadas (média ± faixa), nunca um ponto — lição que custou a revisão de uma conclusão (Exp. 9 → 12).
- **"Mais dados" não é monotônico.** Dados com variação genuína (eras novas) ajudam; dados redundantes podem prejudicar, diluindo o sinal com negativos. O valor de um lote depende do que ele adiciona, não do volume.
- **O gargalo pode estar no pré-processamento.** A maior alavanca do projeto foi controlar o que entra na imagem, não o modelo nem os dados.
- **Decisão sob equivalência é decisão legítima.** Quando duas opções empatam na métrica (Exp. 16), escolher pela que melhor serve ao objetivo de design — e não fingir que uma é superior — é mais honesto e mais robusto.

### Estado atual e próximos passos

O projeto é um **MVP funcional**: pipeline completo, modelo de produção definido (a rodada mediana, não o pico), lista de compras gerada e validada à mão. Os caminhos à frente, em ordem de retorno provável: **recorte guiado por detecção** (detectar o rosto/Pokémon na arte e recortar com precisão em volta dele — refinamento da maior alavanca do projeto); **active learning** (candidatas confirmadas viram novos positivos, retroalimentando o modelo); **expandir a cobertura** às eras fundadoras do TCG; e **janela maior para full-arts**, a hipótese (b) que o Exp. 16 deixou testada só pela metade.