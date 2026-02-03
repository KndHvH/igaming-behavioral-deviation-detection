# Identificação de desvios comportamentais no iGaming por meio de aprendizado não supervisionado

**English version**: [`README.en.md`](README.en.md)

Este repositório acompanha a monografia de especialização em Inteligência Artificial (PECE/Poli-USP) de **Matias Cornelsen Herklotz** (2025), orientada pela **Profa. Dra. Larissa Driemeier**.

O objetivo do trabalho é **identificar desvios comportamentais** no contexto de iGaming a partir de dados históricos de apostas, usando **técnicas não supervisionadas**. A proposta combina:

- **Clusterização de usuários** a partir de métricas comportamentais, para encontrar um **grupo de referência** considerado mais estável.
- **Autoencoders** treinados apenas nesse grupo de referência, usando **erro de reconstrução** como medida contínua de “distância” do comportamento esperado e permitindo **análise diária** de desvios.

O texto completo da monografia está em `monografia.pdf`.

## Resumo

O trabalho investiga a identificação de desvios comportamentais no iGaming por meio de aprendizado não supervisionado, analisando o comportamento de apostadores ao longo do tempo para identificar padrões anômalos. Inicialmente, os apostadores são agrupados por características comportamentais, possibilitando a seleção de um grupo de referência mais estável. A partir desse grupo, treinam-se autoencoders para modelar o comportamento esperado; o erro de reconstrução é usado como medida de distância em relação ao padrão observado. Os resultados indicam que dias com maior erro tendem a apresentar comportamentos mais anômalos, e um limiar pode ser empregado para classificar dias de aposta. Apesar de promissora, a metodologia requer validação adicional, especialmente quanto à relação entre anomalias detectadas e comportamentos problemáticos e quanto à definição/uso de limiares.

**Palavras-chave**: Apostas online; iGaming; Adicção comportamental; Clusterização; Autoencoders.

## Estrutura do repositório

O repositório é centrado em dois notebooks (clusterização e autoencoder) e nos artefatos de dados/figuras gerados durante o desenvolvimento:

```text
.
├─ monografia.pdf
├─ model/
│  ├─ clustering.ipynb          # engenharia de features + clusterização de usuários
│  ├─ autoencoder.ipynb         # ensemble de autoencoders + erro de reconstrução por dia
│  ├─ helper/
│  │  └─ model_plots.py         # funções auxiliares de visualização
│  ├─ data/
│  │  ├─ users.parquet          # tabela agregada por usuário (insumo para clusterização)
│  │  ├─ labeled_users.parquet  # usuários rotulados com cluster (saída da clusterização)
│  │  ├─ daily.parquet          # tabela agregada por usuário-dia (insumo para autoencoder)
│  │  └─ recon_df.parquet       # resultados do autoencoder (erro de reconstrução, etc.)
│  └─ figs/
│     ├─ elbow.pdf              # apoio à escolha de k/diagnósticos de clusterização
│     ├─ autoencoder.pdf        # arquitetura/diagnósticos do autoencoder
│     ├─ p*.pdf                 # figuras intermediárias do desenvolvimento
│     └─ user_*.pdf             # exemplos de usuários analisados (séries/erros)
├─ pyproject.toml               # dependências Python (ambiente)
└─ uv.lock                      # lockfile para reprodução com uv
```

### Sobre os notebooks

- **`model/clustering.ipynb`**: notebook exploratório que implementa o pipeline de pré-processamento/engenharia de features e **agrupa usuários** (abordagem final na monografia: **K-Means com 6 clusters** após redução de dimensionalidade). O cluster mais “estável” é usado como referência para a etapa seguinte.
- **`model/autoencoder.ipynb`**: usa os usuários do cluster de referência, treina um **ensemble de autoencoders** (arquitetura simples e simétrica) e computa **erro de reconstrução** por dia como medida de desvio. O notebook também explora a ideia de **limiar** (ex.: percentil 95 do grupo de referência) para classificar dias como atípicos.

## Dados

### Fonte

O conjunto de dados utilizado é público (Kaggle): **bc.game Crash Dataset [Historical]** (apostas do jogo *Crash* no site bc.game, em criptomoedas, com campos convertíveis para USD quando aplicável).

Link: [Kaggle — bc.game Crash Dataset [Historical]](https://www.kaggle.com/datasets/ccanb23/bcgame-crash-dataset)

### O que está versionado aqui

Por padrão, este repositório inclui tabelas **processadas/agregadas** em `model/data/` (arquivos `.parquet`) e figuras em `model/figs/`, que foram usadas para análise e discussão.

Arquivos brutos grandes (por exemplo, `bets.csv`, `games.csv` ou artefatos intermediários como `bets.parquet`) estão listados no `.gitignore` e **não devem ser commitados**.

### Observações importantes

- **IDs**: os usuários são representados por identificadores numéricos (sem dados pessoais explícitos).
- **Escopo**: o trabalho **não** faz diagnóstico clínico; ele mede **desvios de padrão** (anomalias) no comportamento observado.

## Como reproduzir

Os notebooks refletem um fluxo real de desenvolvimento (exploratório/iterativo). Ainda assim, a reprodução costuma funcionar bem seguindo as orientações abaixo.

### Requisitos

- **Python**: >= 3.10 (ver `pyproject.toml`)
- **Gerenciador de ambiente**: recomendado `uv` (há `uv.lock`)
- **Para o autoencoder**: é necessário **PyTorch** (`torch`). A instalação pode variar conforme CPU/GPU; siga as instruções oficiais do projeto para escolher o wheel/versão adequados ao seu hardware: [PyTorch — Get Started (Local)](https://pytorch.org/get-started/locally/)

### Passo a passo (com `uv`)

1. Sincronize o ambiente:

```bash
uv sync
```

2. Abra e execute os notebooks em seu editor (ex.: VS Code/Cursor) usando o kernel do ambiente criado pelo `uv`.

3. Ordem sugerida:
   - Execute `model/clustering.ipynb` para (re)gerar `users.parquet` e `labeled_users.parquet`.
   - Execute `model/autoencoder.ipynb` para (re)gerar `daily.parquet`/`recon_df.parquet` e as análises baseadas no erro de reconstrução.

### Reprodução “do zero” (a partir do dado bruto)

Para reproduzir todas as etapas desde as tabelas originais do Kaggle, baixe o dataset e posicione os arquivos brutos em `model/data/` conforme esperado pelos notebooks (por exemplo `bets.csv` e `games.csv`). Esses arquivos estão ignorados no Git por padrão.

## Resultados e artefatos

Alguns resultados/diagnósticos gerados durante o trabalho ficam em:

- **Figuras**: `model/figs/` (ex.: escolhas de hiperparâmetros, visualizações de clusters, exemplos de séries de usuários e dias atípicos)
- **Tabelas processadas**: `model/data/` (ex.: agregações por usuário e por usuário-dia, rótulos de cluster, dataframe de reconstrução)

Para a interpretação completa (metodologia, discussão, limitações e conclusão), consulte `monografia.pdf`.

## Limitações (síntese)

Entre as principais limitações discutidas na monografia:

- Ausência de rótulos clínicos/operacionais (o que impede avaliação supervisionada direta).
- Janela temporal curta do dataset e falta de variáveis contextuais (depósitos/saques, autoexclusão, demografia, etc.).
- Questões de fuso horário e presença de padrões que podem refletir automação (apostas automáticas), que não necessariamente caracterizam comportamento problemático.

## Como citar

Se você usar este repositório ou a metodologia descrita, cite a monografia:

> HERKLOTZ, Matias Cornelsen. **Identificação de desvios comportamentais no iGaming por meio de aprendizado não supervisionado**. 2025. Monografia (Especialização em Inteligência Artificial) – Escola Politécnica, Universidade de São Paulo, São Paulo, 2025.

