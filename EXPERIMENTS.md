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

## Exp. 8 — Fine-tuning SigLIP2, 3 blocos, gabarito limpo (117 pos)
- **Hipótese:** mais capacidade descongelada melhora (se 2 blocos eram pouca
  capacidade) OU overfitta mais cedo (se o gargalo é dado).
- **Resultado:** fundo do val na época 2 (vs época 4 com 2 blocos); top-80 recall
  83%, top-10 28%/50%. Sem ganho acima do ruído (teste com só 18 positivos;
  diferenças de 1–2 positivos).
- **Conclusão:** capacidade extra não ajudou e antecipou o overfitting. Segundo
  desempate (após Exp. 3–4) apontando **fome de dados** como gargalo, não
  capacidade. 3 blocos descartado; 2 blocos mantido. Próxima alavanca: mais dados.

## Exp. 9 — Fine-tuning SigLIP2, 2 blocos, gabarito ampliado (212 pos)
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

## Exp. 10 — Fine-tuning SigLIP2, 2 blocos, 212 pos + weight decay 0.01 (AdamW)
- **Hipotese:** L2 contem o overfitting do Exp. 9; se ajudar, PR-AUC sobe (>=0.66).
- **Resultado:** PR-AUC TESTE = 0.581 (vs 0.615 do Exp. 9). Fundo val epoca 3
  (vs 2), overfitting marginalmente mais tarde mas PR-AUC nao melhorou — piorou
  levemente. Top-K nao comparavel (ruido, 32 pos no teste).
- **Conclusao:** weight decay 0.01 NAO ajudou. Confirma que o teto residual e de
  DADOS, nao de regularizacao. Reverter para Adam sem decay (Exp. 9 e a config
  vigente). Proxima alavanca real: mais dados.

## Exp. 11 — Fine-tuning SigLIP2, 2 blocos, gabarito ampliado (278 pos)
- **Mudanca:** rotulados +8 sets (lote 3: FST, ASR, LOT, TEU, PRC, PHF, LTR, DRX),
  todos de eras JA presentes (SWSH/SM/XY/BW). Gabarito 212 -> 278; catalogo
  4.530 -> 5.802 cartas.
- **Hipotese:** se ainda havia fome de dados (Exp. 9), +66 positivos sobe o PR-AUC.
- **Resultado:** PR-AUC TESTE = 0.469 / 0.487 / 0.516 em tres rodadas (media ~0.49),
  ABAIXO do 0.615 do Exp. 9. Teste: 42 pos em 871 cartas.
- **Diagnostico:** resultado contraintuitivo (mais dados, PR-AUC menor) disparou
  investigacao. Pipeline verificado integro: gabarito 278 completo sobre o catalogo
  (sem falsos negativos), imagens presentes, teste bem distribuido (nao enviesado
  para sets dificeis). A queda nao era artefato de pipeline. Mas o 0.615 do Exp. 9
  era rodada UNICA — suspeita de que fosse o topo da variancia, nao a media.
- **Decisao:** comparar 212 vs 278 com multiplas rodadas controladas (Exp. 12)
  antes de concluir qualquer coisa. Licao de metodo: medir com rodada unica e
  enganoso quando a curva de validacao e instavel.

## Exp. 12 — Comparacao controlada 212 vs 278 (3 rodadas cada)
- **Motivacao:** decidir se a queda do Exp. 11 era real ou se o 0.615 do Exp. 9
  foi sorte. Treino sem seed fixo (so o split e fixo, rs=42), entao cada config
  foi rodada 3x para estimar a distribuicao do PR-AUC.
- **Metodo:** 212 reproduzido no catalogo filtrado de 4.564 (8 sets do lote 3
  removidos via script nao-destrutivo); 278 no catalogo completo de 5.802.
  Backups protegidos (catalogo_5802.csv, positivos_ids_278.csv), reversao e
  restauracao por script auditavel.
- **Resultado:**
  - 212 @ 4.564: 0.525 / 0.545 / 0.577 (media ~0.55). O 0.615 original do Exp. 9
    era o TOPO da variancia, fora da faixa das tres rodadas controladas.
  - 278 @ 5.802: 0.469 / 0.487 / 0.516 (media ~0.49).
  - As faixas NAO se sobrepoem (pior do 212 = 0.525 > melhor do 278 = 0.516).
- **Conclusao:** 212 (~0.55) e melhor que 278 (~0.49) de forma reproduzivel.
  Adicionar o lote 3 DEGRADOU o modelo, nao saturou. **Isto revisa a conclusao
  do Exp. 9:** "mais dados sempre ajuda" e FALSO para este problema. O lote 1
  (eras NOVAS, variacao genuina) ajudou; o lote 3 (eras JA vistas) prejudicou —
  pouco sinal positivo novo + ~1.200 negativos novos diluindo. Existe um ponto
  otimo de dados antes de 278.
- **Licao de metodo:** rodada unica e enganosa com curva de validacao instavel;
  comparar configs exige 3+ rodadas e olhar a distribuicao, nunca um ponto.
- **Tensao em aberto:** o modelo e melhor com 212, mas o produto precisa rankear
  os 5.802 (a colecao quer todas as eras). Resolver no proximo passo.

## Exp. 13 — Recorte da arte (resolucao como gargalo) — subconjunto Common/Uncommon/Rare
- **Motivacao:** auditoria (Exp. anteriores) mostrou que o modelo erra olhos
  fechados OBVIOS (Klefki ^^ com prob 0.008). Hipotese: o gargalo nao e dado nem
  capacidade, e RESOLUCAO — o olho (~1-2% da carta) some quando a carta inteira
  e espremida em 224x224.
- **Metodo:** isolar a variavel "recorte". Filtrou-se o catalogo para raridades
  de layout NORMAL (Common/Uncommon/Rare, onde a arte fica sempre na metade
  superior) -> 4.022 cartas, 237 positivos. Mediu-se baseline (sem recorte) e
  depois recorte conservador (manter 55% superiores da carta), 3 rodadas cada.
- **Resultado:**
  - Baseline (sem recorte): PR-AUC 0.507 / 0.566 / 0.559 -> media ~0.544.
  - Com recorte (55% superior): PR-AUC 0.755 / 0.759 / 0.858 -> media ~0.79.
  - Distribuicoes nao se sobrepoem (pior recorte 0.755 >> melhor baseline 0.566).
  - Produto: top-10 precision 90-100%, recall@80 83-92%.
- **Conclusao:** HIPOTESE CONFIRMADA de forma contundente. O gargalo era
  resolucao do detalhe, nao dados nem capacidade. Recortar a arte (jogar fora
  moldura+texto) dobra a fracao util da imagem e o olho fica visivel ao SigLIP.
  Maior ganho do projeto (+0.25 PR-AUC) veio de uma mudanca de pre-processamento
  da imagem, nao de mais dados ou hiperparametros.
- **Escopo/limite:** testado so em layout normal (Common/Uncommon/Rare). Full-art
  e special-art (Illustration/Secret/Ultra) ficaram fora; recorte fixo em 55%
  pode decepar Pokemon nessas. Tratamento delas = proxima fase.
- **Custo:** ~30 min por rodada de treino (crop adiciona processamento de imagem).

## Exp. 14 — Recorte da arte (55%) no catalogo COMPLETO (todas as raridades)
- **Motivacao:** Exp. 13 provou o recorte no subconjunto facil (Common/Uncommon/
  Rare, ~0.79). Faltava validar no escopo de PRODUTO: catalogo completo, incluindo
  full-art e special-art (Illustration/Secret/Ultra), que sao mais dificeis.
- **Premissa validada antes:** inspecao visual mostrou que o "centro de interesse"
  da arte (rosto/olho do Pokemon) fica na metade superior mesmo em full-arts.
  Recorte global simples (55% superior) deve servir para quase toda a base.
- **Gabarito ajustado ao recorte:** revisao visual dos 312 positivos JA recortados
  removeu os que perderam o Pokemon no corte. So 1 caiu (swsh12.5-GG22, full-art
  com o bichinho na metade inferior). Gabarito 312 -> 311. Confirma a premissa:
  99.7% dos positivos tem o olho na metade superior.
- **Metodo:** treino com recorte de 55% sobre o catalogo completo (4.060 cartas
  treino, 217 positivos), split estratificado fixo, 3 rodadas.
- **Resultado:**
  - Sem recorte (Exp. 11/auditoria): PR-AUC ~0.49.
  - Com recorte (catalogo completo): 0.690 / 0.666 / 0.678 -> media ~0.678.
  - Distribuicoes nao se sobrepoem. Ganho de +0.19, estavel (amplitude 0.024).
  - Produto: top-10 precision 80-90%, recall@80 83-89%, recall 100% revisando
    ~226-285 de 871 (corta ~70% do trabalho com cobertura total).
- **Conclusao:** o recorte resolve o problema no escopo COMPLETO, nao so no
  subconjunto facil. Bate a meta de 0.70 (melhor rodada 0.690, media 0.678 a um
  passo). O modelo passou de "investigacao travada" para "pronto para produto".
- **Custo:** ~35-40 min por rodada (catalogo completo + crop).

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
- **Variancia de inicializacao mascara sinal:** sem seed no treino, uma rodada
  unica pode estar no topo ou no fundo da distribuicao. Diferencas de PR-AUC
  menores que a variancia entre rodadas nao sao conclusivas. Comparar configs
  exige multiplas rodadas (media ± faixa), nao um ponto.
- **"Mais dados" nao e monotonico:** dados que adicionam variacao genuina (eras
  novas) ajudam; dados redundantes (mais do mesmo) podem prejudicar ao diluir o
  sinal com negativos. O valor de um lote depende do que ele adiciona, nao do
  volume.
- **O gargalo pode estar no pre-processamento, nao no modelo nem nos dados:**
  perseguimos dados (lotes) e capacidade (blocos) por muitos experimentos; o
  destravamento veio de COMO a imagem e apresentada ao modelo (recorte). Antes de
  assumir "preciso de mais dados/capacidade", questionar se o sinal esta chegando
  intacto a rede.
- **Shortcut learning e o pre-processamento como alavanca:** texto/moldura na
  imagem sao atalhos que o modelo pode aprender em vez do sinal real. Controlar o
  que entra na imagem (recorte) foi a maior alavanca do projeto — maior que dados
  ou capacidade.