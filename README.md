# Segmentação de Imagens DRIVE utilizando U-Net

Este projeto implementa um modelo de segmentação de imagens utilizando a arquitetura U-Net, aplicado ao conjunto de dados DRIVE (Digital Retinal Images for Vessel Extraction). O objetivo é segmentar vasos sanguíneos em imagens de retina para auxiliar no diagnóstico de doenças oculares.

## Descrição do Projeto

O projeto foi criado utilizando a biblioteca `segmentation_models`, que é construída sobre o Keras e TensorFlow. Essa biblioteca possui alguns facilitadores para a criação de redes para a segmentação de imagens, tendo modelos prontos que podem serem importados fácilmente e, também, backbones pré-definidos, o que facilita a realização dos experimentos.

## Requisitos

- Python 3.11.10
- Bibliotecas listadas em `requirements.txt`

### Instalação das Dependências

Para instalar as dependências necessárias, execute:

```bash
pip install -r requirements.txt
```

## Dados

### Disponibilidade dos Dados

Os dados do conjunto DRIVE estão incluídos no repositório na pasta `data/raw`.

## Enunciado

O enunciado do trabalho está disponível no arquivo `enunciado.pdf`, contendo os objetivos, metodologia e requisitos do projeto.

## Execução dos Experimentos

A criação dos modelos, realização dos experimentos e análise dos resultados são realizadas no notebook `generate_results.ipynb`.

### Passos para Execução:

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/seu_usuario/seu_repositorio.git
   ```

2. **Navegue até o diretório do projeto:**

   ```bash
   cd seu_repositorio
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o notebook:**

   Abra o `generate_results.ipynb` em um ambiente Jupyter Notebook ou JupyterLab e execute as células sequencialmente.


## Estrutura do Repositório

- [data/raw/](data/raw/): Dados brutos do DRIVE.
- [results/history/](results/history/): Histórico de treinamento dos modelos.
- [generate_results.ipynb](generate_results.ipynb): Notebook para criação dos modelos, experimentos e análise.
- [requirements.txt](requirements.txt): Lista de dependências necessárias.


## Resultados

A tabela abaixo mostra os resultados obtidos com diferentes configurações de *backbone*, funções de ativação, uso de *data augmentation* e métricas de desempenho:

| model_name                              | backbone      | activation | augmentation | loss    | IOU Score | precision | recall   | accuracy |
|-----------------------------------------|---------------|------------|--------------|---------|-----------|-----------|----------|----------|
| model_efficientnetb0_softmax_aug=True    | efficientnetb0| softmax    | True         | 0.838884| 0.087679  | 0.087679  | 1.000000 | 0.087679 |
| model_efficientnetb0_softmax_aug=False   | efficientnetb0| softmax    | False        | 0.838884| 0.087679  | 0.087679  | 1.000000 | 0.087679 |
| model_None_softmax_aug=True              | None          | softmax    | True         | 0.838884| 0.087679  | 0.087679  | 1.000000 | 0.087679 |
| model_None_softmax_aug=False             | None          | softmax    | False        | 0.838884| 0.087679  | 0.087679  | 1.000000 | 0.087679 |
| model_None_sigmoid_aug=True              | None          | sigmoid    | True         | 0.285999| 0.556048  | 0.680971  | 0.784100 | 0.948858 |
| model_efficientnetb0_sigmoid_aug_True    | efficientnetb0| sigmoid    | True         | 0.273582| 0.590767  | 0.766407  | 0.748621 | 0.975954 |
| model_None_sigmoid_aug=False             | None          | sigmoid    | False        | 0.265071| 0.581868  | 0.782802  | 0.723317 | 0.958144 |
| model_efficientnetb0_sigmoid_aug=False   | efficientnetb0| sigmoid    | False        | 0.251958| 0.598322  | 0.831838  | 0.688824 | 0.960507 |

A melhor configuração foi a combinação de *sigmoid* como função de ativação, *efficientnetb0* como *backbone*, e sem *data augmentation*, que alcançou uma *accuracy* de 96.05% e um *IOU Score* de 0.598.