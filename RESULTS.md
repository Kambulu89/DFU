# Resultados — DFUTissue (teste oficial, 16 imagens)

Resumo das corridas **`full` válidas** (validação cruzada de 5 folds nas 94 imagens de desenvolvimento, *ensemble*, SSL com professor *ensemble*; teste lido uma única vez por corrida).
Não se escolheu nenhuma corrida pelo resultado no teste: reporta-se **a média e o desvio entre corridas independentes**, porque escolher a melhor no teste seria otimista (fuga de informação).

## Dice por corrida (%)

| Corrida `full` | Dice global (réplica do protocolo) | Fibrin | Granulation | Callus |
|---|---|---|---|---|
| CV, 3 arquiteturas (ConvNeXt-S, SegFormer-B3, EfficientNet-B4) | 86,85 | 68,75 | 94,40 | 78,69 |
| CV, 4 arquiteturas + SSL com professor *ensemble* | 87,24 | 69,74 | 94,58 | 79,04 |
| CV, 2 arquiteturas, SSL em 2 gerações | 86,99 | 69,16 | 94,53 | 78,29 |
| CV, 2 arquiteturas, 2 sementes, SSL em 1 geração | 86,72 | 67,27 | 94,38 | 78,23 |
| **Média ± desvio-padrão (4 corridas)** | **86,95 ± 0,22** | **68,73 ± 1,05** | **94,47 ± 0,10** | **78,56 ± 0,38** |
| Artigo (SSL, reportado) | 87,64 | 69,01 | 94,11 | 78,27 |
| Diferença média | −0,69 | −0,28 | +0,36 | +0,29 |

Excluídas: a primeira corrida `full` (divisão oficial, sem CV: 85,80), uma corrida em que o orçamento de tempo cortou a fase final (85,99) e as corridas `quick`/`smoke` (diagnóstico).

## Leitura

* **Por tecido estamos ao nível do artigo:** granulação e calo ligeiramente acima, fibrina ligeiramente abaixo, sempre dentro de ~0,4 pp em média.
* **Dice global (réplica): ~0,7 pp abaixo.** O IC 95 % *bootstrap* de cada corrida no teste tem ~±3 pp e contém sempre 87,64: a diferença não é estatisticamente distinguível.
* **Precision vs recall:** o nosso *recall* global (~87,5 %) supera o do artigo (87,14 %), mas a *precision* (~86 %) fica abaixo (88,14 %). A causa principal é a fibrina: *precision* ~57 % contra 71,5 % no artigo, mas *recall* ~86 % contra 66,7 %. Os nossos modelos marcam fibrina a mais.
* **O teste oficial é atípico:** tem fibrina em 6 de 16 imagens (~4 % do primeiro plano), contra 72 % das imagens (12–17 % do primeiro plano) no desenvolvimento. Ajustar o modelo a essa diferença exigiria olhar para o teste, o que não se faz.
* **O que a validação cruzada mostrou de forma robusta (bootstrap pareado, 94 imagens):** *ensemble* + SSL face ao melhor supervisionado (+1,7 a +2,6 pp), SSL com professor *ensemble* (+0,9 a +1,6 pp no calo e no SegFormer), TTA (+0,3 a +0,8 pp quando escolhido). A ablação (validação oficial, 1 semente) mostrou que o pré-treino vale ~+21 pp. Não ajudaram: encoders maiores (ConvNeXt-B, MiT-B5), 2.ª geração do SSL, pós-processamento (rejeitado sempre pela porta do ganho honesto).

## Melhorias para a próxima corrida

* **Re-treino em 100 % do desenvolvimento** (`refit_all`): depois de a CV fixar a receita, treinam-se modelos nas 94 imagens (+25 % de dados por modelo) que se juntam ao *ensemble* aplicado ao teste.
* **Objetivo das decisões no desenvolvimento** (`objective="combo"`): TTA, candidato e pós-processamento passam a maximizar a média do macro-Dice e do Dice global do artigo, em vez de só o macro-Dice.

Quando houver resultados, a tabela acima é atualizada com a nova corrida (sem substituir as anteriores).
