
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors as pc
import math

def plot_bets_analisys(df):
    odd_more_than_2 = (df['odds_adjusted'] >= 2).astype(int)
    odd_more_than_3 = (df['odds_adjusted'] >= 3).astype(int)
    odd_more_than_5 = (df['odds_adjusted'] >= 5).astype(int)

    bet_more_than_01USD = (df['fiat_bet_amount'] >= 0.1).astype(int)
    bet_more_than_1USD = (df['fiat_bet_amount'] >= 1).astype(int)
    bet_more_than_10USD = (df['fiat_bet_amount'] >= 10).astype(int)

    fig, axs = plt.subplots(4, 2, figsize=(14, 14))

    axs[0,1].hist(df['fiat_bet_amount'], bins=50, color="steelblue", alpha=0.8 , range=(df['fiat_bet_amount'].quantile(0.00), df['fiat_bet_amount'].quantile(0.95)))
    axs[0,1].set_title('Distribuição de fiat_bet_amount por aposta p95')
    axs[0,1].set_xlabel('fiat_bet_amount')
    axs[0,1].set_ylabel('Nº de apostas')

    axs[0,0].hist(df['odds_adjusted'], bins=50, color="orange", alpha=0.8, range=(df['odds_adjusted'].quantile(0.00), df['odds_adjusted'].quantile(0.99)))
    axs[0,0].set_title('Distribuição de odds por aposta p99')
    axs[0,0].set_xlabel('odds')
    axs[0,0].set_ylabel('Nº de apostas')

    labels = ["< 2", "≥ 2"]
    size = [ (odd_more_than_2==0).sum(), (odd_more_than_2==1).sum() ]
    axs[1,0].pie(size, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs[1,0].set_title('Odds ≥ 2')

    labels = ["< 3", "≥ 3"]
    size = [ (odd_more_than_3==0).sum(), (odd_more_than_3==1).sum() ]
    axs[1,1].pie(size, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs[1,1].set_title('Odds ≥ 3')

    labels = ["< 5", "≥ 5"]
    size = [ (odd_more_than_5==0).sum(), (odd_more_than_5==1).sum() ]
    axs[2,0].pie(size, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs[2,0].set_title('Odds ≥ 5')

    labels = ["< 0.1 USD", "≥ 0.1 USD"]
    sizes = [ (bet_more_than_01USD==0).sum(), (bet_more_than_01USD==1).sum() ]
    axs[2,1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs[2,1].set_title('Apostas ≥ 1 USD')


    labels = ["< 1 USD", "≥ 1 USD"]
    sizes = [ (bet_more_than_1USD==0).sum(), (bet_more_than_1USD==1).sum() ]
    axs[3,0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs[3,0].set_title('Apostas ≥ 1 USD')


    labels = ["< 10 USD", "≥ 10 USD"]
    sizes = [ (bet_more_than_10USD==0).sum(), (bet_more_than_10USD==1).sum() ]
    axs[3,1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs[3,1].set_title('Apostas ≥ 10 USD')

    plt.tight_layout()
    plt.show()
    
def plot_users_dash(df):
    u = df.groupby('user_id', observed=True).agg({
        'odds_adjusted': 'mean',
        'fiat_bet_amount': 'mean',
        'bet_risk': 'mean'
    })

    user_odd_more_than_2 = (u['odds_adjusted'] >= 2).astype(int)
    user_odd_more_than_3 = (u['odds_adjusted'] >= 3).astype(int)
    user_odd_more_than_5 = (u['odds_adjusted'] >= 5).astype(int)

    user_bet_more_than_m1R = (u['bet_risk'] >= -1).astype(int)
    user_bet_more_than_0R = (u['bet_risk'] >= 0).astype(int)
    user_bet_more_than_1R = (u['bet_risk'] >= 1).astype(int)

    fig2, axs2 = plt.subplots(4, 2, figsize=(14, 14))


    axs2[0,0].hist(u['odds_adjusted'], bins=100, color="orange", alpha=0.8, range=(u['odds_adjusted'].quantile(0.00), u['odds_adjusted'].quantile(0.99)))
    axs2[0,0].set_title('Distribuição da média de odds por jogador (até p99)')
    axs2[0,0].set_xlabel('odds (média)')
    axs2[0,0].set_ylabel('Nº de jogadores')
    axs2[0,0].axvline(x=2, color='steelblue', linestyle='--', linewidth=1)


    axs2[0,1].hist(df['bet_risk'], bins=100, color="steelblue", alpha=0.8, range=(df['bet_risk'].quantile(0.01), df['bet_risk'].quantile(0.99)))
    axs2[0,1].set_title('Distribuição de riscos médios por jogador (p01 - p99)')
    axs2[0,1].set_xlabel('Risco médio')
    axs2[0,1].set_ylabel('Nº de jogadores')
    axs2[0,1].axvline(x=0, color='orange', linestyle='--', linewidth=1)



    labels = ["< 2", "≥ 2"]
    size = [ (user_odd_more_than_2==0).sum(), (user_odd_more_than_2==1).sum() ]
    axs2[1,0].pie(size, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[1,0].set_title('Jogadores com odds média ≥ 2')

    labels = ["< 3", "≥ 3"]
    size = [ (user_odd_more_than_3==0).sum(), (user_odd_more_than_3==1).sum() ]
    axs2[1,1].pie(size, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[1,1].set_title('Jogadores com odds média ≥ 3')

    labels = ["< 5", "≥ 5"]
    size = [ (user_odd_more_than_5==0).sum(), (user_odd_more_than_5==1).sum() ]
    axs2[2,0].pie(size, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[2,0].set_title('Jogadores com odds média ≥ 5')

    labels = ["< -1", "≥ -1"]
    sizes = [ (user_bet_more_than_m1R==0).sum(), (user_bet_more_than_m1R==1).sum() ]
    axs2[2,1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[2,1].set_title('Jogadores com risco médio ≥ -1')

    labels = ["< 0", "≥ 0"]
    sizes = [ (user_bet_more_than_0R==0).sum(), (user_bet_more_than_0R==1).sum() ]
    axs2[3,0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[3,0].set_title('Jogadores com risco médio ≥ 0')

    labels = ["< 1", "≥ 1"]
    sizes = [ (user_bet_more_than_1R==0).sum(), (user_bet_more_than_1R==1).sum() ]
    axs2[3,1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[3,1].set_title('Jogadores com risco médio ≥ 1')

    plt.tight_layout()
    plt.show()


def plot_pca(X: pd.DataFrame, dim: int = 3) -> None:
    """
    Plota X reduzido de acordo com a dimensão especificada.
    - Se dim==2: scatterplot x/y
    - Se dim==3 and >=3 cols: scatter3d x/y/z
    - Se dim==4 and >=4 cols: scatter3d x/y/z, color=a
    - Se dim==5 and >=5 cols: scatter3d x/y/z, color=a (5a dimensão ignorada mas avisada)
    """
    columns = X.columns
    if dim < 2 or dim > 5:
        print("Apenas dim=2,3,4,5 são suportadas.")
        return
    if len(columns) < dim:
        print(f"Seu DataFrame tem apenas {len(columns)} colunas. Precisa de pelo menos {dim}.")
        return

    if dim == 2:
        fig = px.scatter(
            X, x=columns[0], y=columns[1],
            opacity=0.2
        )
        fig.update_traces(marker=dict(size=3, color='blue'))
        fig.update_layout(title='PCA 2D', template='plotly_white')
        fig.show()
    elif dim == 3:
        fig = px.scatter_3d(
            X, x=columns[0], y=columns[1], z=columns[2],
            opacity=0.2
        )
        fig.update_traces(marker=dict(size=2, color='blue'))
        fig.update_layout(title='PCA 3D', template='plotly_white')
        fig.show()
    elif dim == 4:
        fig = px.scatter_3d(
            X, x=columns[0], y=columns[1], z=columns[2],
            color=columns[3],
            opacity=0.4,
            color_continuous_scale='Viridis'
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(title='PCA 4D (cor=4ª dim)', template='plotly_white')
        fig.show()
    elif dim == 5:
        fig = px.scatter_3d(
            X,
            x=columns[0], y=columns[1], z=columns[2],
            color=columns[3],
            opacity=0.4,
            color_continuous_scale='Viridis',
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            title='PCA 5D (mostrando 3D, cor=4ª dim)\n5ª dim não visualizada',
            template='plotly_white'
        )
        fig.show()
        print(f"Atenção: Só mostrando as 3 primeiras dimensões + cor=4ª.\nA 5ª dimensão ({columns[4]}) não está no plot.")

def plot_results2(X: pd.DataFrame, size=3, category_col: str = "cluster") -> None:
    cols = [c for c in X.columns if c != category_col]
    X_plot = X.copy()

    X_plot[category_col] = X_plot[category_col].astype(str)

    palette = px.colors.qualitative.Set1
    clusters = sorted(X_plot[category_col].unique(), key=lambda v: int(v) if v.lstrip("-").isdigit() else v)

    color_discrete_map = {}
    if "-1" in clusters:
        color_discrete_map["-1"] = "gray"

    other_clusters = [c for c in clusters if c != "-1"]
    for i, c in enumerate(other_clusters):
        color_discrete_map[c] = palette[i % len(palette)]

    common_args = dict(
        data_frame=X_plot,
        color=category_col,
        color_discrete_map=color_discrete_map,
        opacity=0.6,
    )

    if size == 2:
        fig = px.scatter(
            x=cols[0],
            y=cols[1],
            **common_args,
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(title="PCA 2D", template="plotly_white", font=dict(size=14), width=1200, height=600)

    else: 
        fig = px.scatter_3d(
            x=cols[0],
            y=cols[1],
            z=cols[2],
            **common_args,
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(title="PCA 3D", template="plotly_white", font=dict(size=14), width=1200, height=600)

    fig.show()


def plot_users_dash_comparison(df1, df2, label1='df1', label2='df2'):
    import matplotlib.pyplot as plt

    u1 = df1.groupby('user_id', observed=True).agg({
        'odds_adjusted': 'mean',
        'fiat_bet_amount': 'mean',
        'bet_risk': 'mean'
    })
    u2 = df2.groupby('user_id', observed=True).agg({
        'odds_adjusted': 'mean',
        'fiat_bet_amount': 'mean',
        'bet_risk': 'mean'
    })

    # Para odds_mean
    user_odd_more_than_2_1 = (u1['odds_adjusted'] >= 2).astype(int)
    user_odd_more_than_2_2 = (u2['odds_adjusted'] >= 2).astype(int)
    user_odd_more_than_3_1 = (u1['odds_adjusted'] >= 3).astype(int)
    user_odd_more_than_3_2 = (u2['odds_adjusted'] >= 3).astype(int)
    user_odd_more_than_5_1 = (u1['odds_adjusted'] >= 5).astype(int)
    user_odd_more_than_5_2 = (u2['odds_adjusted'] >= 5).astype(int)

    user_bet_more_than_m1R_1 = (u1['bet_risk'] >= -1).astype(int)
    user_bet_more_than_m1R_2 = (u2['bet_risk'] >= -1).astype(int)
    user_bet_more_than_0R_1 = (u1['bet_risk'] >= 0).astype(int)
    user_bet_more_than_0R_2 = (u2['bet_risk'] >= 0).astype(int)
    user_bet_more_than_1R_1 = (u1['bet_risk'] >= 1).astype(int)
    user_bet_more_than_1R_2 = (u2['bet_risk'] >= 1).astype(int)

    fig2, axs2 = plt.subplots(8, 2, figsize=(14, 28))

    # 1. Histogram Odds média ({label1}, {df2})
    axs2[0, 0].hist(u1['odds_adjusted'], bins=100, color="orange", alpha=0.8, range=(u1['odds_adjusted'].quantile(0.00), u1['odds_adjusted'].quantile(0.99)))
    axs2[0, 0].set_title(f'Distribuição odds média por jogador ({label1})')
    axs2[0, 0].set_xlabel('odds (média)')
    axs2[0, 0].set_ylabel('Nº de jogadores')
    axs2[0, 0].axvline(x=2, color='steelblue', linestyle='--', linewidth=1)

    axs2[0, 1].hist(u2['odds_adjusted'], bins=100, color="orange", alpha=0.8, range=(u2['odds_adjusted'].quantile(0.00), u2['odds_adjusted'].quantile(0.99)))
    axs2[0, 1].set_title(f'Distribuição odds média por jogador ({label2})')
    axs2[0, 1].set_xlabel('odds (média)')
    axs2[0, 1].set_ylabel('Nº de jogadores')
    axs2[0, 1].axvline(x=2, color='steelblue', linestyle='--', linewidth=1)

    # 2. Histogram risco médio ({label1}, {label2})
    axs2[1, 0].hist(u1['bet_risk'], bins=100, color="steelblue", alpha=0.8, range=(u1['bet_risk'].quantile(0.01), u1['bet_risk'].quantile(0.99)))
    axs2[1, 0].set_title(f'Distribuição de riscos médios ({label1})')
    axs2[1, 0].set_xlabel('Risco médio')
    axs2[1, 0].set_ylabel('Nº de jogadores')
    axs2[1, 0].axvline(x=0, color='orange', linestyle='--', linewidth=1)

    axs2[1, 1].hist(u2['bet_risk'], bins=100, color="steelblue", alpha=0.8, range=(u2['bet_risk'].quantile(0.01), u2['bet_risk'].quantile(0.99)))
    axs2[1, 1].set_title(f'Distribuição de riscos médios ({label2})')
    axs2[1, 1].set_xlabel('Risco médio')
    axs2[1, 1].set_ylabel('Nº de jogadores')
    axs2[1, 1].axvline(x=0, color='orange', linestyle='--', linewidth=1)

    # 3. Pie: odds média >=2 ({label1}, {label2})
    labels = ["< 2", "≥ 2"]
    size1 = [ (user_odd_more_than_2_1 == 0).sum(), (user_odd_more_than_2_1 == 1).sum() ]
    size2 = [ (user_odd_more_than_2_2 == 0).sum(), (user_odd_more_than_2_2 == 1).sum() ]
    axs2[2, 0].pie(size1, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[2, 0].set_title(f'Jogadores com odds média ≥ 2 ({label1})')
    axs2[2, 1].pie(size2, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[2, 1].set_title(f'Jogadores com odds média ≥ 2 ({label2})')

    # 4. Pie: odds média >=3 ({label1}, {label2})
    labels = ["< 3", "≥ 3"]
    size1 = [ (user_odd_more_than_3_1==0).sum(), (user_odd_more_than_3_1==1).sum() ]
    size2 = [ (user_odd_more_than_3_2==0).sum(), (user_odd_more_than_3_2==1).sum() ]
    axs2[3, 0].pie(size1, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[3, 0].set_title(f'Jogadores com odds média ≥ 3 ({label1})')
    axs2[3, 1].pie(size2, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[3, 1].set_title(f'Jogadores com odds média ≥ 3 ({label2})')

    # 5. Pie: odds média >=5 ({label1}, {label2})
    labels = ["< 5", "≥ 5"]
    size1 = [ (user_odd_more_than_5_1==0).sum(), (user_odd_more_than_5_1==1).sum() ]
    size2 = [ (user_odd_more_than_5_2==0).sum(), (user_odd_more_than_5_2==1).sum() ]
    axs2[4, 0].pie(size1, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[4, 0].set_title(f'Jogadores com odds média ≥ 5 ({label1})')
    axs2[4, 1].pie(size2, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[4, 1].set_title(f'Jogadores com odds média ≥ 5 ({label2})')

    # 6. Pie: risco médio >= -1 ({label1}, {label2})
    labels = ["< -1", "≥ -1"]
    size1 = [ (user_bet_more_than_m1R_1==0).sum(), (user_bet_more_than_m1R_1==1).sum() ]
    size2 = [ (user_bet_more_than_m1R_2==0).sum(), (user_bet_more_than_m1R_2==1).sum() ]
    axs2[5, 0].pie(size1, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[5, 0].set_title(f'Risco médio ≥ -1 ({label1})')
    axs2[5, 1].pie(size2, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[5, 1].set_title(f'Risco médio ≥ -1 ({label2})')

    # 7. Pie: risco médio >= 0 ({label1}, {label2})
    labels = ["< 0", "≥ 0"]
    size1 = [ (user_bet_more_than_0R_1==0).sum(), (user_bet_more_than_0R_1==1).sum() ]
    size2 = [ (user_bet_more_than_0R_2==0).sum(), (user_bet_more_than_0R_2==1).sum() ]
    axs2[6, 0].pie(size1, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[6, 0].set_title(f'Risco médio ≥ 0 ({label1})')
    axs2[6, 1].pie(size2, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[6, 1].set_title(f'Risco médio ≥ 0 ({label2})')

    # 8. Pie: risco médio >= 1 ({label1}, {label2})
    labels = ["< 1", "≥ 1"]
    size1 = [ (user_bet_more_than_1R_1==0).sum(), (user_bet_more_than_1R_1==1).sum() ]
    size2 = [ (user_bet_more_than_1R_2==0).sum(), (user_bet_more_than_1R_2==1).sum() ]
    axs2[7, 0].pie(size1, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[7, 0].set_title(f'Risco médio ≥ 1 ({label1})')
    axs2[7, 1].pie(size2, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'orange'])
    axs2[7, 1].set_title(f'Risco médio ≥ 1 ({label2})')

    plt.tight_layout()
    plt.show()

def plot_cluster_feature_comparisons(
    df: pd.DataFrame,
    features: list,
    window: str = "2d",          # "2d", "7d", "14d"
    stat: str = "mean",          # "mean" ou "std"
    cluster_col: str = "cluster_label"
):

    feature_cols = {}
    for f in features:
        col_name = f"{f}_{stat}_{window}"
        if col_name in df.columns:
            feature_cols[f] = col_name

    clusters = df[cluster_col].unique()
    clusters = np.sort(clusters)

    n_feats = len(feature_cols)
    ncols = 2
    nrows = math.ceil(n_feats / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (feat_name, col_name) in zip(axes, feature_cols.items()):
        tmp = df[[cluster_col, col_name]].dropna()

        stats = tmp.groupby(cluster_col)[col_name].agg(["mean", "std", "count"])
        means = stats["mean"].values
        stds = stats["std"].values

        idx_max = np.nanargmax(means)
        highlight_cluster = stats.index[idx_max]

        colors = ["steelblue"] * len(stats)
        colors[idx_max] = "orange"

        ax.bar(stats.index.astype(str), means, yerr=stds, capsize=4, alpha=0.8, color=colors)
        ax.set_title(f"{feat_name} ({stat}_{window})\ncluster destaque: {highlight_cluster}")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(f"{feat_name} ({stat})")


        overall_mean = tmp[col_name].mean()
        ax.axhline(overall_mean, linestyle="--", linewidth=1, color="gray", alpha=0.7)

    for ax in axes[n_feats:]:
        ax.set_visible(False)

    fig.suptitle(f"Comparação de clusters - {stat}_{window}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_user_vs_clusters(
    df_users: pd.DataFrame,
    user_id,
    features: list,
    window: str = "7d",
    stat: str = "mean",
    cluster_col: str = "cluster_label",
    user_col: str = "user_id",
):
    row = df_users[df_users[user_col] == user_id]
    row = row.iloc[0]
    user_cluster = row[cluster_col]

    feature_cols = {
        f: f"{f}_{stat}_{window}"
        for f in features
        if f"{f}_{stat}_{window}" in df_users.columns
    }

    n_feats = len(feature_cols)
    ncols = 2
    nrows = math.ceil(n_feats / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (feat_name, col_name) in zip(axes, feature_cols.items()):
        tmp = df_users[[cluster_col, col_name]].dropna()
        stats = tmp.groupby(cluster_col)[col_name].agg(["mean", "std"])
        

        colors = ["darkgray"] * len(stats)
        idx_user_cluster = list(stats.index).index(user_cluster)
        colors[idx_user_cluster] = "steelblue"

        ax.bar(
            stats.index.astype(str),
            stats["mean"].values,
            yerr=stats["std"].values,
            capsize=3,
            alpha=0.85,
            color=colors,
            error_kw=dict(ecolor="#444444", elinewidth=1.2, alpha=1),
        )
        x = np.arange(len(stats))
        
        ax.scatter(
            x[idx_user_cluster],
            [row[col_name]],
            s=80,
            color="orange",
            edgecolor="black",
            zorder=3,
        )

        ax.set_title(f"{feat_name} ({stat}_{window})")
        ax.set_xlabel("cluster")
        ax.set_ylabel(feat_name)

    for ax in axes[n_feats:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Usuário {user_id} — cluster {str(user_cluster)}",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    plt.show()


def plot_user_daily(
    df_daily: pd.DataFrame,
    user_id,
    features: list,
    date_col: str = "day_of_year",
    user_col: str = "user_id",
):
    tmp = df_daily[df_daily[user_col] == user_id].copy()
    if tmp.empty:
        raise ValueError(f"Usuário {user_id} não encontrado no df diário.")

    tmp = tmp.sort_values(date_col)
    tmp[date_col] = tmp[date_col].astype(str) #tirar isso se der erro

    n_feats = len(features)
    ncols = 1
    nrows = math.ceil(n_feats / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    x = tmp[date_col]

    for ax, feat in zip(axes, features):
        if feat not in tmp.columns:
            ax.set_visible(False)
            continue

        ax.plot(x, tmp[feat], marker="o", linewidth=1.5)
        ax.set_title(feat)
        ax.set_xlabel("dia")
        ax.set_ylabel(feat)

    for ax in axes[n_feats:]:
        ax.set_visible(False)

    fig.suptitle(f"Histórico diário — usuário {user_id}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_user_day_hourly(
    df,
    user_id,
    day_of_year,
    user_col: str = "user_id",
    day_col: str = "day_of_year",
    hour_col: str = "hour",
):
    tmp = df[(df[user_col] == user_id) & (df[day_col] == day_of_year)].copy()
    if tmp.empty:
        raise ValueError(f"Usuário {user_id} não tem registros no dia {day_of_year}.")

    agg = (
        tmp.groupby(hour_col)
        .agg(
            odds_adjusted_mean=("odds_adjusted", "mean"),
            bet_risk_mean=("bet_risk", "mean"),
            user_bet_amount_proportion_mean=("user_bet_amount_proportion", "mean"),
            fiat_bet_amount_mean=("fiat_bet_amount", "mean"),
            n_bets=("bet_id", "count"),
            fiat_bet_amount_sum=("fiat_bet_amount", "sum"),
        )
        .sort_index()
    )

    x = agg.index

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.ravel()

    axes[0].plot(x, agg["n_bets"], marker="o")
    axes[0].set_title("n apostas por hora")
    axes[0].set_xlabel("hora")
    axes[0].set_ylabel("n apostas")

    axes[1].plot(x, agg["fiat_bet_amount_sum"], marker="o")
    axes[1].set_title("soma fiat_bet_amount por hora")
    axes[1].set_xlabel("hora")
    axes[1].set_ylabel("soma fiat_bet_amount")

    axes[2].plot(x, agg["odds_adjusted_mean"], marker="o")
    axes[2].set_title("odds_adjusted (média por hora)")
    axes[2].set_xlabel("hora")
    axes[2].set_ylabel("odds_adjusted_mean")

    axes[3].plot(x, agg["bet_risk_mean"], marker="o")
    axes[3].set_title("bet_risk (média por hora)")
    axes[3].set_xlabel("hora")
    axes[3].set_ylabel("bet_risk_mean")

    axes[4].plot(x, agg["user_bet_amount_proportion_mean"], marker="o")
    axes[4].set_title("user_bet_amount_proportion (média por hora)")
    axes[4].set_xlabel("hora")
    axes[4].set_ylabel("user_bet_amount_proportion_mean")

    axes[5].plot(x, agg["fiat_bet_amount_mean"], marker="o")
    axes[5].set_title("fiat_bet_amount (média por hora)")
    axes[5].set_xlabel("hora")
    axes[5].set_ylabel("fiat_bet_amount_mean")

    fig.suptitle(f"user {user_id} — dia {day_of_year} (agregado por hora)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_user_hour_bets(
    df,
    user_id,
    day_of_year,
    hour,
    user_col: str = "user_id",
    day_col: str = "day_of_year",
    hour_col: str = "hour",
):
    tmp = df[
        (df[user_col] == user_id)
        & (df[day_col] == day_of_year)
        & (df[hour_col] == hour)
    ].copy()

    if tmp.empty:
        raise ValueError(f"Sem apostas para user={user_id}, dia={day_of_year}, hora={hour}.")

    tmp = tmp.sort_values("date")
    x = range(len(tmp))

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x, tmp["odds_adjusted"])
    axes[0].set_ylabel("odds_adjusted")
    axes[0].set_title(f"user {user_id} — dia {day_of_year}, hora {hour}")

    axes[1].plot(x, tmp["fiat_bet_amount"])
    axes[1].set_ylabel("fiat_bet_amount")

    axes[2].plot(x, tmp["bet_risk"])
    axes[2].set_ylabel("bet_risk")

    axes[3].plot(x, tmp["user_bet_amount_proportion"])
    axes[3].set_ylabel("bet_prop")
    axes[3].set_xlabel("índice da aposta (ordem no horário)")

    plt.tight_layout()
    plt.show()
