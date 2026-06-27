import numpy as np
from gprofiler import GProfiler
import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import matplotlib.lines as mlines

from utils import read_pickle, get_df


result_path = '/project_antwerp/TCGA-PRAD/test_results.pkl'
obj = read_pickle(result_path)[0]
df_preds = get_df(obj, "preds")
all_gene_names = df_preds.columns[1:].tolist()

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

seeds = [11, 222, 333, 4444, 55555]
selected_genes = {}
for seed in seeds:
    indices = np.load(f'/project_antwerp/baseline/preprocessing_analysis/index_files/top_1000_genes_indices_welford_seed_{seed}.npy', allow_pickle=True)
    indices = [all_gene_names[i] for i in indices]
    indices = [gene.replace("rna_", "") for gene in indices]
    selected_genes[seed] = indices

background_genes = [gene.replace("rna_", "") for gene in all_gene_names]

# ------------------------------------------------------------------
# Per-seed enrichment analysis (top 1000 genes for each seed)
# ------------------------------------------------------------------

print("\n" + "="*70)
print("PER-SEED ENRICHMENT ANALYSIS")
print("="*70)

outdir_per_seed = "/project_antwerp/baseline/preprocessing_analysis/plots"
os.makedirs(outdir_per_seed, exist_ok=True)

source_colors = {
    "GO:BP": "#1f77b4",
    "REAC": "#d62728"
}

gp = GProfiler(return_dataframe=True)

for seed in seeds:
    seed_genes = selected_genes[seed]
    print(f"\n{'─'*60}")
    print(f"Seed: {seed} | Genes: {len(seed_genes)}")
    print(f"{'─'*60}")

    results = gp.profile(
        organism='hsapiens',
        query=seed_genes,
        background=background_genes,
        sources=['GO:BP', 'REAC'],
        significance_threshold_method='fdr',
        user_threshold=0.05,
        no_evidences=False,
    )

    print(f"Shape of results: {results.shape}")

    if results.empty:
        print(f"No results returned for seed {seed} — skipping.")
        continue

    significant = results[results['significant']]
    if significant.empty:
        print(f"No significant results for seed {seed} — skipping.")
        continue

    significant = significant.sort_values('p_value')
    cols = ['source', 'name', 'p_value', 'precision', 'recall', 'intersection_size', 'term_size', 'intersections']
    print(significant[cols].to_string(index=False))

    # Prepare plot dataframe
    plot_df = significant.copy()
    plot_df = plot_df.dropna(subset=["p_value", "name", "source", "intersection_size", "term_size"])
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_value"])
    plot_df["color"] = plot_df["source"].map(source_colors)
    plot_df["label"] = plot_df["name"].apply(lambda x: "\n".join(textwrap.wrap(x, width=45)))
    bar_df = plot_df.sort_values("neg_log10_p", ascending=True)

    # Bar plot
    plt.figure(figsize=(10, max(5, 0.4 * len(bar_df))))

    plt.barh(
        bar_df["label"],
        bar_df["neg_log10_p"],
        color=bar_df["color"],
        edgecolor="black",
        linewidth=0.6
    )

    plt.xlabel("-log10(FDR)", fontsize=11)
    plt.ylabel("")
    plt.title(f"Enriched pathways — Seed {seed} (GO:BP and Reactome)", fontsize=13, weight="bold")

    handles = [
        mlines.Line2D([], [], color=color, marker='s', linestyle='None',
                      markersize=8, label=src)
        for src, color in source_colors.items()
    ]
    plt.axvline(x=-np.log10(0.05), color="grey", linestyle="--", linewidth=1)
    plt.legend(
        handles=handles + [plt.Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label="FDR = 0.05")],
        title="Source", loc="lower right", frameon=True
    )

    plt.tight_layout()
    out_path = f"{outdir_per_seed}/barplot_seed_{seed}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

print("\nPer-seed enrichment analysis complete.")