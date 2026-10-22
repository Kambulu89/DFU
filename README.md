# DFUTissue — notebook Kaggle de segmentação de tecidos (fibrina / granulação / calo)

Pipeline alternativo e **auditável** para o dataset DFUTissue (arXiv:2406.16012), com comparação sem fuga de informação com o artigo.
Este diretório é de investigação e é independente da plataforma de gestão (`backend/`, `src/`).

| Ficheiro | Função |
|---|---|
| `dfu_tissue_segmentation.ipynb` | **Notebook a carregar no Kaggle** (53 células; sem saídas — são geradas ao executar) |
| `parts/*.py` | Fonte do notebook (formato percent, uma secção por ficheiro). Edite aqui e regenere |
| `build_notebook.py` | `python build_notebook.py` regenera o `.ipynb` (precisa de `nbformat`); `--flat x.py` gera um script plano para testes |

## Estado de verificação (honesto)

* Testado **em CPU, fora do Kaggle**, com os dados reais clonados do repositório dos autores: modo `smoke` de ponta a ponta (auditoria, métricas, treino,
  sondas, ablação, SSL, ensemble/TTA, avaliação no teste, exportação), as 3 arquiteturas reais (treino de 1 época + SSL), carregamento de pesos locais,
  TTA multiescala a 256 e 384.
* **Não foi executado em GPU Kaggle** e **não existem resultados de desempenho** neste repositório. Os números do modo `smoke` (modelo minúsculo, 1–3 épocas)
  não têm valor científico. Não se afirma que o artigo foi ultrapassado.
* Sem Internet no ambiente de desenvolvimento, os pesos pré-treinados de `timm` (EfficientNet-B4, ConvNeXt-S) **não puderam ser descarregados**:
  o caminho de descarga online não foi exercitado para esses encoders (só o fallback local, com um ficheiro gerado localmente). Os pesos MiT-B3 de `smp` descarregam-se do GitHub.

## Como correr no Kaggle

1. Crie um *Dataset* com a pasta `DFUTissue` de <https://github.com/uwm-bigdata/DFUTissueSegNet> (contém `Labeled/` e `Unlabeled/`) e adicione-o ao notebook.
2. Ative a GPU (T4/P100). Com **Internet ligada**, o notebook instala `segmentation-models-pytorch` e descarrega pesos pré-treinados. Com **Internet desligada**, adicione
   datasets com o *wheel* de `segmentation_models_pytorch` e com os pesos (ficheiros `.pth/.pt/.bin/.safetensors` cujo nome/pasta contenha `efficientnet_b4`,
   `convnext_small` ou `mit_b3`). Sem pesos, treina-se com inicialização aleatória e isso fica registado (`weights_provenance.json`) — não é *transfer learning*.
3. Na célula de configuração escolha `MODE`: `quick` (primeira corrida, 2 arquiteturas × 1 semente) ou `full` (3 arquiteturas × 3 sementes, sondas, ablação, SSL em 2 rondas).
   Variáveis opcionais: `DFU_MODE`, `DFU_DATA_ROOT`, `DFU_ALLOW_DOWNLOAD=1` (clona o repositório se houver Internet), `DFU_CFG_OVERRIDE` (JSON).
4. Execute tudo. As execuções são idempotentes (`runs/*/result.json`): se a sessão cair, volte a correr e o que já terminou é reutilizado (treinos longos retomam de `last.pt`).

## Decisões metodológicas que afetam a comparação com o artigo

* **Divisão oficial** 78/16/16 (`Labeled/*_names.txt`). O teste só é lido na fase final, com registo (`test_ledger.json`).
* **Definição do Dice "global" do artigo**: ver secção 1 do notebook. O código dos autores exclui o fundo e usa acumuladores não repostos entre classes; o notebook
  replica esse protocolo (validado contra uma porta literal do ciclo original) **e** reporta métricas padrão. A equivalência com o 87,64 do artigo é uma hipótese.
* **Frame**: as imagens "Padded" são as originais (67–266 px) coladas num canto de uma tela preta 256×256 (mediana: 27 % da tela). O pipeline recorta e amplia o conteúdo
  (`roi`) e reverte para o tamanho original na avaliação; o modo `padded` replica a tela do artigo. Em modo `full` a escolha é feita na validação.
* **Observação dos dados**: a fibrina é 12–17 % do primeiro plano em treino/validação mas apenas ~4 % no teste oficial (6/16 imagens). A métrica de fibrina no teste é, por isso, muito ruidosa.

## Pesos e licenças

A origem de cada execução fica em `weights_provenance.json`. Os pesos MiT-B3 (NVlabs SegFormer) têm licença da NVIDIA de uso não comercial; confirme nos *model cards* a licença dos
restantes pesos antes de reutilizar. O dataset DFUTissue tem os seus próprios termos (ver `Citation.txt` no repositório dos autores); cite o artigo.
