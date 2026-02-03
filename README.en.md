# Unsupervised detection of behavioral deviations in iGaming using clustering and autoencoders

**Versão em português**: [`README.md`](README.md)

This repository accompanies the **capstone / final project** (PECE/Poli-USP specialization in Artificial Intelligence) by **Matias Cornelsen Herklotz** (2025), advised by **Prof. Dr. Larissa Driemeier**.

The project’s goal is to **identify behavioral deviations** in iGaming from historical betting data using **unsupervised learning**. The approach combines:

- **User clustering** based on behavioral metrics, in order to identify a **reference group** regarded as more stable.
- **Autoencoders** trained only on that reference group, using **reconstruction error** as a continuous distance-to-expected-behavior measure and enabling **daily deviation analysis**.

The full text (in Portuguese) is available in `monografia.pdf`.

## Abstract (from the monograph)

This work investigates the identification of behavioral deviations in the context of iGaming through unsupervised learning techniques, with the objective of analyzing gambler behavior over time and identifying anomalous patterns. Initially, gamblers are grouped according to behavioral characteristics, which enables the identification of a reference group considered more stable. Based on this group, autoencoders are trained to model expected behavior, using reconstruction error as a distance measure with respect to the observed pattern. This approach enables the daily analysis of individual behavior through the quantification of deviations relative to the target behavior. The results indicate that days with higher reconstruction error tend to exhibit more anomalous behavior, making it possible to employ a threshold to classify betting days. Although the results suggest that the proposed methodology is promising for the identification and monitoring of behavioral deviations, further validation is still required, particularly with respect to the relationship between the detected anomalies and problematic behaviors, as well as the definition and use of thresholds or methods for behavioral distinction.

**Keywords**: Online gambling; iGaming; Behavioral addiction; Clustering; Autoencoders.

## Repository structure

The repository is centered around two notebooks (clustering and autoencoder) and the data/figure artifacts produced during development:

```text
.
├─ monografia.pdf
├─ model/
│  ├─ clustering.ipynb          # feature engineering + user clustering
│  ├─ autoencoder.ipynb         # autoencoder ensemble + reconstruction error per day
│  ├─ helper/
│  │  └─ model_plots.py         # plotting utilities
│  ├─ data/
│  │  ├─ users.parquet          # per-user aggregated table (input to clustering)
│  │  ├─ labeled_users.parquet  # users with assigned cluster labels (clustering output)
│  │  ├─ daily.parquet          # per-user-per-day aggregated table (input to autoencoder)
│  │  └─ recon_df.parquet       # autoencoder outputs (reconstruction error, etc.)
│  └─ figs/
│     ├─ elbow.pdf              # k selection / clustering diagnostics
│     ├─ autoencoder.pdf        # autoencoder architecture / diagnostics
│     ├─ p*.pdf                 # intermediate development figures
│     └─ user_*.pdf             # per-user examples (time series / errors)
├─ pyproject.toml               # Python dependencies
└─ uv.lock                      # lockfile for reproducible environments with uv
```

### Notebooks overview

- **`model/clustering.ipynb`**: an exploratory notebook that implements preprocessing/feature engineering and **clusters users** (final approach in the monograph: **K-Means with 6 clusters** after dimensionality reduction). The most “stable” cluster is selected as the reference for the next stage.
- **`model/autoencoder.ipynb`**: selects users from the reference cluster, trains an **autoencoder ensemble** (simple symmetric architecture), and computes **reconstruction error** per day as a deviation score. The notebook also explores a **thresholding** approach (e.g., the 95th percentile of the reference group) to classify days as atypical.

## Data

### Source

The dataset is public (Kaggle): **bc.game Crash Dataset [Historical]** (bets from the *Crash* game on bc.game, using cryptocurrencies, with USD-convertible fields when applicable).

Link: [Kaggle — bc.game Crash Dataset [Historical]](https://www.kaggle.com/datasets/ccanb23/bcgame-crash-dataset)

### What is versioned in this repository

By default, this repository includes **processed/aggregated** tables in `model/data/` (`.parquet` files) and figures in `model/figs/`, which were used throughout the analysis and discussion.

Large raw files (e.g., `bets.csv`, `games.csv`, or intermediate artifacts such as `bets.parquet`) are listed in `.gitignore` and **should not be committed**.

### Important notes

- **IDs**: users are represented by numeric identifiers (no explicit personal data).
- **Scope**: this work does **not** provide a clinical diagnosis; it measures **pattern deviations** (anomalies) in observed behavior.

## Reproducibility

The notebooks reflect a real development flow (exploratory/iterative). Still, reproduction typically works well by following the steps below.

### Requirements

- **Python**: >= 3.10 (see `pyproject.toml`)
- **Environment manager**: `uv` is recommended (see `uv.lock`)
- **Autoencoder training**: requires **PyTorch** (`torch`). Installation varies by CPU/GPU; follow the official instructions to pick the right wheel/version for your hardware: [PyTorch — Get Started (Local)](https://pytorch.org/get-started/locally/)

### Steps (with `uv`)

1. Sync the environment:

```bash
uv sync
```

2. Open and run the notebooks in your editor (e.g., VS Code/Cursor) using the kernel from the `uv` environment.

3. Suggested execution order:
   - Run `model/clustering.ipynb` to (re)generate `users.parquet` and `labeled_users.parquet`.
   - Run `model/autoencoder.ipynb` to (re)generate `daily.parquet` / `recon_df.parquet` and the analyses based on reconstruction error.

### Running from raw data

To reproduce all steps starting from the original Kaggle tables, download the dataset and place the raw files under `model/data/` as expected by the notebooks (e.g., `bets.csv` and `games.csv`). These files are ignored by Git by default.

## Results and artifacts

Some outputs produced during the work are stored in:

- **Figures**: `model/figs/` (e.g., hyperparameter choices, cluster visualizations, per-user time series examples and atypical days)
- **Processed tables**: `model/data/` (e.g., per-user and per-user-per-day aggregations, cluster labels, reconstruction dataframe)

For the full interpretation (methodology, discussion, limitations, and conclusion), see `monografia.pdf`.

## Limitations (summary)

Key limitations discussed in the monograph include:

- Lack of clinical/operational labels (which prevents direct supervised evaluation).
- Short time window and missing contextual variables (deposits/withdrawals, self-exclusion, demographics, etc.).
- Time-zone issues and patterns that may reflect automation (auto-betting), which do not necessarily indicate problematic behavior.

## How to cite

If you use this repository or the methodology described here, please cite the monograph:

> HERKLOTZ, Matias Cornelsen. **Identificação de desvios comportamentais no iGaming por meio de aprendizado não supervisionado**. 2025. Monografia (Especialização em Inteligência Artificial) – Escola Politécnica, Universidade de São Paulo, São Paulo, 2025.

## License

This repository does not specify a license yet. If you want to make it fully reusable by third parties, consider adding a license (e.g., MIT/BSD-3-Clause for code and an appropriate license for the monograph text, if desired).

