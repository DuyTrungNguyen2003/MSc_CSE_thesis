import numpy as np
from gprofiler import GProfiler
import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

from utils import read_pickle, get_df


result_path = '/project_antwerp/TCGA-PRAD/test_results.pkl'
obj = read_pickle(result_path)[0]
df_preds = get_df(obj, "preds")
all_gene_names = df_preds.columns[1:].tolist()

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

seeds = [11, 222, 333, 4444, 55555]
# key: (seed, fold), value: selected genes
selected_genes = {}
for seed in seeds:
    # folds 1-6
    for i in range(1, 7):
        indices = np.load(f'/project_antwerp/baseline/ensemble/logs/selected_indices_seed_{seed}_fold_{i}.npy', allow_pickle=True)
        indices = [all_gene_names[i] for i in indices]
        indices = [gene.replace("rna_", "") for gene in indices]
        selected_genes[(seed, i)] = indices
        # print(f"Selected genes for seed {seed}, fold {i}: {len(indices)}")

background_genes = [gene.replace("rna_", "") for gene in all_gene_names]

# ------------------------------------------------------------------
# Pool all selected genes across seeds and folds for a single enrichment analysis
# ------------------------------------------------------------------

genes_count = {}
for key, genes in selected_genes.items():
    for gene in genes:
        genes_count[gene] = genes_count.get(gene, 0) + 1

# select genes that are selected "fraction" of the time
fractions = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
for fraction in fractions:
    threshold = len(seeds) * 6 * fraction
    pooled_genes = [gene for gene, count in genes_count.items() if count >= threshold]
    print(f"\nNumber of pooled genes (selected in at least {fraction*100}% of cases): {len(pooled_genes)}")

    # ------------------------------------------------------------------
    # Enrichment analysis
    # ------------------------------------------------------------------
    gp = GProfiler(return_dataframe=True)

    results = gp.profile(
        organism='hsapiens',
        query=pooled_genes,
        background=background_genes,      
        # sources=['GO:BP', 'GO:MF', 'KEGG', 'REAC'],
        sources = ['GO:BP', 'REAC'],
        significance_threshold_method='fdr',
        user_threshold=0.05,
        no_evidences=False,
    )

    print("\n--- Raw results summary ---")
    print("Shape of results:", results.shape)
    # print("Available columns in results:", results.columns.tolist())

    # results = results[results['significant']].sort_values('p_value')
    # cols = ['source', 'name', 'p_value', 'precision', 'recall', 'intersection_size', 'term_size', 'intersections']
    # # cols = ['source', 'name', 'description']
    # print(results[cols].to_string(index=False))
    if results.empty:
        print(f"No results returned for fraction {fraction} — skipping.")
        continue

    significant = results[results['significant']]
    if significant.empty:
        print(f"No significant results for fraction {fraction} — skipping.")
        continue

    results = significant.sort_values('p_value')
    cols = ['source', 'name', 'p_value', 'precision', 'recall', 'intersection_size', 'term_size', 'intersections']
    print(results[cols].to_string(index=False))

    # ------------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------------
    outdir = "/project_antwerp/baseline/ensemble/plots"
    os.makedirs(outdir, exist_ok=True)

    plot_df = results.copy()

    # Safety: keep only usable rows
    plot_df = plot_df.dropna(subset=["p_value", "name", "source", "intersection_size", "term_size"])
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_value"])
    plot_df["gene_ratio"] = plot_df["intersection_size"] / plot_df["term_size"]

    # Convert intersections to list if needed
    def parse_intersections(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return [g.strip() for g in x.replace("[", "").replace("]", "").replace("'", "").split(",") if g.strip()]
        return []

    plot_df["gene_list"] = plot_df["intersections"].apply(parse_intersections)

    # Optional: remove weak single-gene terms
    # plot_df_filtered = plot_df[plot_df["intersection_size"] >= 2].copy()
    plot_df_filtered = plot_df.copy()

    # Use top N terms for readable figures
    # top_n = 20
    # top_df = plot_df_filtered.sort_values("p_value").head(top_n).copy()
    top_df = plot_df_filtered.sort_values("p_value").copy()

    # Short labels
    def wrap_label(label, width=45):
        return "\n".join(textwrap.wrap(label, width=width))

    top_df["label"] = top_df["name"].apply(wrap_label)

    ## ------------------------------------------------------------------
    # Bar plot
    # ------------------------------------------------------------------

    bar_df = top_df.sort_values("neg_log10_p", ascending=True)

    source_colors = {
        "GO:BP": "#1f77b4",   # blue
        "REAC": "#d62728"     # red
    }

    bar_df["color"] = bar_df["source"].map(source_colors)

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
    plt.title("Enriched pathways (GO:BP and Reactome)", fontsize=13, weight="bold")

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=color, marker='s', linestyle='None',
                    markersize=8, label=src)
        for src, color in source_colors.items()
    ]

    plt.axvline(x=-np.log10(0.05), color="grey", linestyle="--", linewidth=1)
    plt.legend(handles=handles + [plt.Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label="FDR = 0.05")],
               title="Source", loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(f"{outdir}/barplot_{fraction}.png")
    plt.close()


    print(f"\nSaved enrichment plots to: {outdir}")