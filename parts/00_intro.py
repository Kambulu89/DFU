# %% [markdown]
# # Segmentação de tecidos em úlceras do pé diabético (DFUTissue) — pipeline alternativo
#
# **Objetivo científico.** Segmentar *Fibrin*, *Granulation* e *Callus* (mais fundo) no dataset **DFUTissue**
# (110 imagens anotadas: 78 treino / 16 validação / 16 teste; 600 imagens sem anotação) e comparar, **sem fuga de
# informação**, com o artigo *Wound Tissue Segmentation in Diabetic Foot Ulcer Images Using Deep Learning: A Pilot Study*
# (Dhar et al., [arXiv:2406.16012](https://arxiv.org/abs/2406.16012); código/dados em
# [uwm-bigdata/DFUTissueSegNet](https://github.com/uwm-bigdata/DFUTissueSegNet)).
#
# **Estado desta entrega (leia primeiro).**
# * O código foi escrito e **testado de ponta a ponta em CPU, em modo `smoke`, com os dados reais**, fora do Kaggle.
#   **Não foi executado em GPU Kaggle** e **nenhum resultado de desempenho é fornecido aqui**: todas as tabelas,
#   gráficos e conclusões desta página são gerados quando *você* executa o notebook.
# * Os resultados do artigo (tabela abaixo) são constantes copiadas das Tabelas 3–5 do artigo, rotuladas como tal.
# * O notebook só declara que ultrapassou o artigo se as métricas calculadas na sessão o mostrarem, e mostra sempre o
#   intervalo de confiança *bootstrap* sobre as 16 imagens de teste (um teste tão pequeno é muito ruidoso).
#
# ## 1. O que o artigo reporta e como as métricas são calculadas
#
# | Métrica (artigo) | SL | SSL |
# |---|---|---|
# | Dice global | 84,89 | **87,64** |
# | IoU / Precision / Recall (SSL) | 73,75 / 87,69 / 82,27 (SL) | 77,99 / 88,14 / 87,14 |
# | Dice Fibrin / Granulation / Callus (SSL) | — | 69,01 / 94,11 / 78,27 |
#
# **Achado crítico para a comparação.** A média aritmética dos três Dice por classe do artigo (SSL) é 80,46, não 87,64, e a
# *precision* global (88,14) é superior à média das três *precision* por classe (81,75). Logo o "global" **não** é um macro-Dice
# dos tecidos. Lendo o código dos autores (`Codes/supervised_training.ipynb`, célula de inferência) verifica-se que:
#
# 1. as máscaras são índices `0=fundo, 1=fibrina, 2=granulação, 3=calo` e o fundo é **excluído** das métricas;
# 2. as contagens `tp/fp/fn` são acumuladas por imagem **sem serem repostas entre classes** e somadas ao total global
#    (`stp += tp` dentro do ciclo de classes), o que dá mais peso às classes processadas depois e **não é um micro-Dice padrão**;
# 3. as métricas por classe usam também esses acumuladores, e classes ausentes em ambos (GT e predição) são ignoradas.
#
# Por isso este notebook calcula, sobre a mesma matriz de confusão e **com definições explícitas**:
# *(a)* Dice por classe agregado sobre o dataset; *(b)* macro-Dice dos 3 tecidos; *(c)* micro-Dice dos tecidos;
# *(d)* macro com fundo; *(e)* a média por imagem; e *(f)* uma **réplica fiel do protocolo do código dos autores**
# (validada contra uma porta literal do ciclo original). A *(f)* é a única comparável ao 87,64 do artigo, e mesmo assim
# a equivalência é uma **hipótese** (não pude confirmar que a tabela do artigo foi gerada exatamente com esse código).
# A métrica de seleção de modelos usa *(b)*, que é a mais exigente para a fibrina.
#
# ## 2. Análise dos problemas e estratégia escolhida
#
# Problemas identificados (artigo + auditoria dos dados reais): poucos dados anotados (78 treino); fibrina = 0,9 % dos
# píxeis (≈12 % do primeiro plano); fronteiras ambíguas fibrina/granulação e calo/pele; variação de cor e iluminação;
# e — descoberto na auditoria — **as imagens "Padded" são as originais (67–266 px) coladas no canto de uma tela preta 256×256**,
# pelo que numa imagem pequena a ferida ocupa uma fração minúscula do que a rede vê e a resolução efetiva é baixa.
#
# | Decisão | Justificação | Custo / risco |
# |---|---|---|
# | **ROI + *letterbox***: recortar o conteúdo, ampliá-lo para preencher a tela (256 ou 384) e reverter na avaliação | aumenta a resolução efetiva sem inventar detalhe; mais *tokens* úteis para encoders com *stride* 32; avaliado também no frame original | ampliar não recupera detalhe; testado contra o modo `padded` (réplica do artigo) na validação |
# | **Modelos complementares**: U-Net++ + EfficientNet-B4 (CNN), UPerNet + ConvNeXt-S IN-22k (CNN moderna, contexto multiescala), SegFormer MiT-B3 (Transformer) | diversidade de vieses indutivos para *ensemble*; todos com pesos pré-treinados | MiT-B3 é o encoder do artigo, mas aqui com decoder diferente (não o P-scSE) e como membro de ensemble |
# | **Transfer learning**: *freeze* inicial do encoder, *fine-tuning* com LR diferencial (encoder ≪ decoder), AdamW, *warmup*+cosseno, EMA | poucos dados → regularizar a adaptação | LR/épocas podem sobreajustar a validação de 16 imagens (mitigado por sementes múltiplas) |
# | **Augmentation clinicamente plausível**, implementada em OpenCV/NumPy (máscaras sempre com *nearest*): flips, afim moderada, fotométrica suave, ruído/desfoque; sem *elastic* nem *copy-paste* | evitar tecidos anatomicamente implausíveis; sem dependência de versão do Albumentations | menos diversidade que métodos agressivos |
# | **Perda**: CE (pesos raiz-inversa limitados) + Dice nos tecidos; alternativas (Focal+Dice, Focal-Tversky, +Lovász) comparadas na validação; *class-aware sampling* para fibrina | desequilíbrio sem sobreponderar (falsos positivos) | — |
# | **SSL com professor *ensemble* limpo por fold** (`ssle`): *pseudo-labels* offline do ensemble (com TTA) de todas as arquiteturas do mesmo fold — que nunca viram as imagens retidas, logo o OOF não tem fuga —, limiares por classe, aluno com vista forte + CutMix; gerações configuráveis (`ssl_ens_gens`): na g>1 o professor passa a ser o ensemble {supervisionados + alunos das gerações anteriores} do mesmo fold e o aluno parte sempre do supervisionado (a g=2 não trouxe ganho mensurável na 2.ª corrida, por isso o `full` usa 1) | um ensemble gera *pseudo-labels* mais exatas do que um modelo isolado, e uma 2.ª geração refina-as | custo extra de inferência/treino; medido por bootstrap pareado contra o supervisionado e a geração anterior |
# | **SSL**: *EMA teacher–student* online, vistas fraca→forte (fotométrica + CutMix), limiares de confiança **por classe** adaptativos, ignorar píxeis incertos, ramp-up do peso, rondas iterativas, exclusão de quase-duplicados de val/teste; aplicado a **todos os folds** das melhores arquiteturas e avaliado **sem seleção de época** (`ssl_select="last"`) | substitui a escolha aleatória de *pseudo-labels* do artigo; proteção contra *confirmation bias* | só ajuda se o modelo supervisionado já for razoável; medido por bootstrap pareado, não assumido |
# | **Validação cruzada estratificada de 5 folds** nas 94 imagens de desenvolvimento (modo `full`) | cada imagem tem uma previsão **fora de amostra** (OOF): decide-se com 94 imagens em vez de 16; o *ensemble* final usa os modelos de todos os folds, ou seja, quase todos os dados | 5× o custo de treino; épocas fixas e EMA final (sem seleção) para o OOF não ter viés |
# | **Inferência**: TTA (flips ± multiescala), *ensemble* uniforme de probabilidades, **escala de probabilidade por classe** e remoção de componentes pequenas (área mínima global e da fibrina) e de componentes de fibrina de baixa confiança — tudo ajustado **só no OOF/validação**, com histerese e medição do ganho "honesto" (ajuste numa metade, avaliação na outra) | ataca os falsos positivos de fibrina observados | ajuste em poucos parâmetros; ganho in-sample é otimista |
# | **Re-treino em 100 % do desenvolvimento** (`refit_all`): depois de a CV fixar a receita (épocas fixas, EMA final), treinam-se modelos nas 94 imagens, que se juntam ao *ensemble* aplicado ao teste | +25 % de dados por modelo; prática padrão depois da seleção | não têm previsões fora de amostra: não entram nas decisões do desenvolvimento |
# | **Objetivo das decisões** (`objective="combo"`): TTA, candidato e pós-processamento maximizam a média do macro-Dice e do Dice global do artigo | alinha as escolhas com as métricas que o artigo reporta | fixado antes de ver o teste |
# | **Estatística**: *bootstrap* pareado (mesmas imagens) para decidir se SSL, *ensemble*, TTA ou pós-processamento melhoram de facto | com 94 imagens tem poder real; com 16 do teste só indica tendência | — |
#
# **Não implementado deliberadamente** (para não apresentar o que não foi feito): *self-supervised pre-training* (600 imagens de 256 px não
# justificam o custo face a ImageNet/IN-22k), DINOv2 (depende de pesos externos/Internet), Optuna (validação de 16 imagens
# sobreajusta com HPO) e a tarefa binária *Chronic Wound* (experiência distinta, não misturável com a multiclasse).
#
# **Disciplina anti-fuga.** O teste só é lido na fase final, através de um *ledger* (`TEST_LEDGER`) que regista cada
# avaliação; não participa em seleção, *pseudo-labels*, limiares nem pós-processamento. Os 600 não anotados são filtrados
# por *perceptual hash* contra validação e teste.
#
# **Como usar no Kaggle.** (1) Crie um *Dataset* Kaggle com a pasta `DFUTissue` do repositório (contém `Labeled/` e
# `Unlabeled/`) e adicione-o ao notebook; (2) ative a GPU; (3) escolha `MODE` na célula de configuração
# (`quick` ≈ primeira corrida curta com divisão oficial; `full` = experiência completa com validação cruzada, ~horas); (4) com Internet desligada, adicione também um dataset com os
# *wheels* de `segmentation_models_pytorch` e os pesos pré-treinados (ver célula de ambiente e de pesos).
