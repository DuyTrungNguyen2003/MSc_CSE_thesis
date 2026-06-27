import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import read_pickle, get_df


result_path = '/project_antwerp/TCGA-PRAD/test_results.pkl'
obj = read_pickle(result_path)[0]
df_preds = get_df(obj, "preds")
all_gene_names = df_preds.columns[1:].tolist()

# ------------------------------------------------------------------
# Gene sets
# ------------------------------------------------------------------

sig_49 = [
    "rna_PDGFB", "rna_ASPN", "rna_FOXS1", "rna_SMC4", "rna_FAM72B", "rna_ITGBL1", "rna_LPPR4", "rna_SPAG1",
    "rna_BUB1", "rna_GOLGA7B", "rna_CENPF", "rna_GDF3", "rna_MAPK8IP2", "rna_ESM1", "rna_PRC1", "rna_MYT1",
    "rna_LRFN2", "rna_SHCBP1", "rna_AHRR", "rna_CBX2", "rna_GMNN", "rna_NUF2", "rna_STC2", "rna_RAI14",
    "rna_FGF14", "rna_ZNF467", "rna_TMEM132E", "rna_FAM72D", "rna_CST2", "rna_KIF14", "rna_APLNR", "rna_DLGAP5",
    "rna_CENPE", "rna_IGSF1", "rna_NAAA", "rna_ASPA", "rna_SLC22A1", "rna_TAOK3", "rna_C2orf88", "rna_NCAPD3",
    "rna_GLB1L3", "rna_PAGE4", "rna_ANO7", "rna_EDN3", "rna_TPT1", "rna_ADPGK", "rna_PACSIN3", "rna_GLB1L2",
    "rna_PLOD1"
]

sig_157 = [
    "rna_BCAS1", "rna_ALOX15B", "rna_EYA1", "rna_PAGE4", "rna_BIRC5", "rna_MYBPC1", "rna_MT1G", "rna_EHHADH",
    "rna_MPPED2", "rna_PTTG1", "rna_CRIP2", "rna_CXCL13", "rna_UBE2C", "rna_SERPINA3", "rna_ATP8A2", "rna_INHBA",
    "rna_LMNB1", "rna_CDKN3", "rna_MCM4", "rna_FMO5", "rna_NOTCH3", "rna_NEK2", "rna_PRKCB1", "rna_MT1F",
    "rna_CHRNA2", "rna_SCUBE2", "rna_TOP2A", "rna_RRM2", "rna_JAG1", "rna_CD38", "rna_MRPS12", "rna_BUB1B",
    "rna_RGS4", "rna_BGN", "rna_ANPEP", "rna_GNG4", "rna_ASPN", "rna_CYP27A1", "rna_FKBP1B", "rna_KCNN2",
    "rna_SPAG5", "rna_GPR116", "rna_MT1A", "rna_VEGF", "rna_ARG2", "rna_DPP4", "rna_SFTPA2", "rna_CENPF",
    "rna_DLG7", "rna_SMPDL3A", "rna_SERPINE1", "rna_PGM5", "rna_SC65", "rna_NTRK3", "rna_CYP4F12", "rna_SATB1",
    "rna_ERG", "rna_NELL2", "rna_HSD17B6", "rna_KHDRBS3", "rna_TFF3", "rna_PENK", "rna_F2R", "rna_ABAT",
    "rna_SLC15A2", "rna_PROK1", "rna_COL4A1", "rna_TYMS", "rna_ITPR2", "rna_CDC42BPA", "rna_FXYD1", "rna_NUSAP1",
    "rna_CACNA1D", "rna_BMP6", "rna_DHFR", "rna_XRCC2", "rna_PTN", "rna_TK1", "rna_KLF5", "rna_RARRES1",
    "rna_ESM1", "rna_TRIP13", "rna_PAH", "rna_PTK7", "rna_CYB5A", "rna_PTPRN2", "rna_MGST1", "rna_IGFBP6",
    "rna_MYLK", "rna_PLA2G7", "rna_GRIA3", "rna_MT1X", "rna_SLC4A4", "rna_DLGAP1", "rna_C2", "rna_FOXM1",
    "rna_DDEF2", "rna_CSPG2", "rna_F5", "rna_CUL7", "rna_SEMA3F", "rna_COPS5", "rna_SCG1D2", "rna_CCNB1",
    "rna_GREB1", "rna_PHC2", "rna_GMNN", "rna_CES1", "rna_TBXAS1", "rna_HERC3", "rna_RGS16", "rna_FLRT2",
    "rna_E2F3", "rna_KIAA0040", "rna_TGFB2", "rna_ESRRG", "rna_ARMCX2", "rna_PLA2G10", "rna_RNASE4",
    "rna_HIST1H1T", "rna_ENTPD1", "rna_ELK4", "rna_SHMT2", "rna_NAT1", "rna_EVPL", "rna_LIPH", "rna_MAP3K5",
    "rna_LCN2", "rna_RPE65", "rna_TPST1", "rna_CRISP3", "rna_LAMC1", "rna_ASNS", "rna_MEOX2", "rna_SEMA3C",
    "rna_FST", "rna_FGF7", "rna_HRAS", "rna_PLCG1", "rna_AOX1", "rna_ANG", "rna_UPP1", "rna_MYBL2",
    "rna_TLE1", "rna_IL1R1", "rna_OCLN", "rna_MMP11", "rna_SPARCL1", "rna_AZGP1", "rna_GSTA4", "rna_RAB27A",
    "rna_GMDS", "rna_EPHB6", "rna_NME3", "rna_KIAA0907", "rna_CPT1A", "rna_HOXB7"
]

paper_genes = [
    # TCGA paper: Summary; Results > "The molecular taxonomy of primary prostate cancer"; Figure 1
    "ERG", "ETV1", "ETV4", "FLI1",
    "SPOP", "FOXA1", "IDH1",

    # TCGA paper: Results > "The molecular taxonomy of primary prostate cancer"
    # ETS fusion partners; TMPRSS2 is the most frequent, SLC45A3/NDRG1 also reported
    "TMPRSS2", "SLC45A3", "NDRG1",

    # Review paper: Section 4.2, "Tumour Suppressor Genes"
    # TCGA paper also discusses PTEN/TP53/RB1 in recurrent alterations and metastatic comparison
    "PTEN", "TP53", "RB1",

    # Review paper: Section 4.1, "Gene Families" / MYC family
    # TCGA paper: Results > molecular taxonomy, "other" group; chr8 amplification spanning MYC
    "MYC",

    # Review paper: Section 4.3, "Androgen Receptor (AR) Signalling Genes"
    # TCGA paper: Results > "AR activity is variable in primary prostate cancers"
    "AR",

    # Review paper: Section 2.4.6, "Genetic Factors Associated With Prostate Cancer"
    # TCGA paper: Results > "Recurrently altered genes..." also mentions NKX3-1 somatic mutation
    "NKX3-1",

    # Review paper: Section 2.4.6, hereditary / recombination DNA repair genes
    # Review paper: Section 4.4, "DNA Repair Genes"
    "BRCA1", "BRCA2", "ATM", "ATR", "PALB2", "RAD51", "CHEK2",

    # Review paper: Section 2.4.6, hereditary / mismatch repair genes
    "PMS2", "MSH2", "MLH1",

    # TCGA paper: Figure 5, "Alterations in clinically relevant pathways"
    # TCGA paper: Results > clinically relevant DNA repair alterations
    "RAD51C", "FANCD2", "CDK12",

    # TCGA paper: Figure 5, PI3K/RAS pathway alterations
    # Review paper: Section 4.2, PTEN–PI3K–AKT–mTOR pathway discussion
    "PIK3CA", "PIK3CB", "AKT1", "MTOR",

    # TCGA paper: Results > "Recurrently altered genes..." and Figure 5, RAS/MAPK pathway
    "BRAF", "HRAS", "RAC1", "RRAS2",

    # TCGA paper: Results > "Recurrently altered genes..." / recurrent focal deletions
    "MAP3K1", "MAP3K7",

    # TCGA paper: Results > molecular taxonomy, SPOP-mutant/CHD1-deleted subset
    "CHD1", "SPINK1",

    # TCGA paper: Results > "Recurrently altered genes and their patterns across subtypes"
    "MED12", "CDKN1B", "CTNNB1", "KMT2C", "KMT2D", "APC", "ZMYM3",

    # TCGA paper: Results > recurrent focal amplifications/deletions
    "CCND1", "FGFR1", "WHSC1L1", "SPOPL", "FOXP1", "RYBP", "SHQ1",

    # Review paper: Section 6.2, "Urine-Based Biomarkers" / ExoDx Prostate IntelliScore
    "PCA3", "SPDEF",

    # Review paper: Section 6.2, "Urine-Based Biomarkers" / SelectMDx
    "HOXC6", "DLX1",

    # Review paper: Section 6.1, "Biochemical Markers" / 4K score and PSA kallikrein markers
    "KLK2", "KLK3",

    # Review paper: Section 6, tissue biomarkers / ConfirmMDx
    "RASSF1", "GSTP1"
]

sig_49 = [gene.replace("rna_", "") for gene in sig_49]
sig_157 = [gene.replace("rna_", "") for gene in sig_157]

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

seeds = [11, 222, 333, 4444, 55555]
# key: (seed, fold), value: selected genes
selected_genes = {}
for seed in seeds:
    # folds 1-6
    for i in range(1, 7):
        indices = np.load(f'/project_antwerp/baseline/ensemble_preprocessing/logs/selected_indices_seed_{seed}_fold_{i}.npy', allow_pickle=True)
        indices = [all_gene_names[i] for i in indices]
        indices = [gene.replace("rna_", "") for gene in indices]
        selected_genes[(seed, i)] = indices
        # print(f"Selected genes for seed {seed}, fold {i}: {len(indices)}")

# ------------------------------------------------------------------
# Pool all selected genes across seeds and folds for a single enrichment analysis
# ------------------------------------------------------------------
 
genes_count = {}
for key, genes in selected_genes.items():
    for gene in genes:
        genes_count[gene] = genes_count.get(gene, 0) + 1

print(f"Total unique genes selected across all seeds and folds: {len(genes_count)}")

# ------------------------------------------------------------------
# Full pooled set analysis (no thresholding)
# ------------------------------------------------------------------

pooled_genes_full = list(genes_count.keys())
print(f"\n--- Full pooled set (no threshold): {len(pooled_genes_full)} genes ---")

sig_49_overlap_full    = set(pooled_genes_full) & set(sig_49)
sig_157_overlap_full   = set(pooled_genes_full) & set(sig_157)
paper_genes_overlap_full = set(pooled_genes_full) & set(paper_genes)

print(f"\nNumber of pooled genes overlapping with sig_49:      {len(sig_49_overlap_full)}")
print(f"Overlapping genes with sig_49: {sorted(sig_49_overlap_full)}")
print(f"\nNumber of pooled genes overlapping with sig_157:     {len(sig_157_overlap_full)}")
print(f"Overlapping genes with sig_157: {sorted(sig_157_overlap_full)}")
print(f"\nNumber of pooled genes overlapping with paper_genes: {len(paper_genes_overlap_full)}")
print(f"Overlapping genes with paper_genes: {sorted(paper_genes_overlap_full)}")

sorted_genes_full = sorted(genes_count.items(), key=lambda x: x[1], reverse=True)
gene_rank_full = {gene: rank + 1 for rank, (gene, _) in enumerate(sorted_genes_full)}
max_count = len(seeds) * 6  # 30

# Signature records (full)
sig_records_full = []
for gene in genes_count:
    in_49  = gene in sig_49_overlap_full
    in_157 = gene in sig_157_overlap_full
    if in_49 or in_157:
        sig = "both" if (in_49 and in_157) else ("sig_49" if in_49 else "sig_157")
        sig_records_full.append({"gene": gene, "rank": gene_rank_full[gene], "count": genes_count[gene], "sig": sig})

sig_records_full.sort(key=lambda x: x["rank"])

# Paper records (full)
paper_records_full = []
for gene in genes_count:
    if gene in paper_genes_overlap_full:
        paper_records_full.append({"gene": gene, "rank": gene_rank_full[gene], "count": genes_count[gene]})

paper_records_full.sort(key=lambda x: x["rank"])

# Plot — signatures (full)
if sig_records_full:
    color_map = {"sig_49": "#534AB7", "sig_157": "#0F6E56", "both": "#993C1D"}

    genes  = [r["gene"]  for r in sig_records_full]
    counts = [r["count"] for r in sig_records_full]
    colors = [color_map[r["sig"]] for r in sig_records_full]
    ranks  = [r["rank"]  for r in sig_records_full]

    fig, ax = plt.subplots(figsize=(8, max(4, len(genes) * 0.35)))
    ax.barh(range(len(genes)), counts, color=colors, height=0.6)

    for i, (count, rank) in enumerate(zip(counts, ranks)):
        ax.text(count + 0.3, i, f"rank #{rank}", va="center", fontsize=8, color="#555")

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Selection count", fontsize=10)
    ax.set_xlim(0, max_count + 5)

    patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=patches, fontsize=9, loc="lower right")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("/project_antwerp/baseline/ensemble_preprocessing/plots/signature_overlap_ranks_full.png", dpi=150, bbox_inches="tight")
    plt.show()

# Plot — paper genes (full)
if paper_records_full:
    paper_color = "#C2760C"

    genes  = [r["gene"]  for r in paper_records_full]
    counts = [r["count"] for r in paper_records_full]
    ranks  = [r["rank"]  for r in paper_records_full]

    fig, ax = plt.subplots(figsize=(8, max(4, len(genes) * 0.35)))
    ax.barh(range(len(genes)), counts, color=paper_color, height=0.6)

    for i, (count, rank) in enumerate(zip(counts, ranks)):
        ax.text(count + 0.3, i, f"rank #{rank}", va="center", fontsize=8, color="#555")

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Selection count", fontsize=10)
    ax.set_xlim(0, max_count + 5)

    patches = [mpatches.Patch(color=paper_color, label="paper_genes")]
    ax.legend(handles=patches, fontsize=9, loc="lower right")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("/project_antwerp/baseline/ensemble_preprocessing/plots/paper_genes_overlap_ranks_full.png", dpi=150, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------------
# Fraction loop
# ------------------------------------------------------------------
 
# select genes that are selected "fraction" of the time
fractions = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
for fraction in fractions:
    threshold = len(seeds) * 6 * fraction
    pooled_genes = [gene for gene, count in genes_count.items() if count >= threshold]
    print(f"\nNumber of pooled genes (selected in at least {fraction*100}% of cases): {len(pooled_genes)}")
 
    # ------------------------------------------------------------------
    # Overlapping genes with signatures and paper_genes
    # ------------------------------------------------------------------
 
    sig_49_overlap    = set(pooled_genes) & set(sig_49)
    sig_157_overlap   = set(pooled_genes) & set(sig_157)
    paper_genes_overlap = set(pooled_genes) & set(paper_genes)
 
    print(f"\nNumber of pooled genes overlapping with sig_49:      {len(sig_49_overlap)}")
    print(f"Overlapping genes with sig_49: {sorted(sig_49_overlap)}")
    print(f"\nNumber of pooled genes overlapping with sig_157:     {len(sig_157_overlap)}")
    print(f"Overlapping genes with sig_157: {sorted(sig_157_overlap)}")
    print(f"\nNumber of pooled genes overlapping with paper_genes: {len(paper_genes_overlap)}")
    print(f"Overlapping genes with paper_genes: {sorted(paper_genes_overlap)}")
 
    # ------------------------------------------------------------------
    # Rank and count for overlapping genes — signatures plot
    # ------------------------------------------------------------------
 
    sorted_genes = sorted(genes_count.items(), key=lambda x: x[1], reverse=True)
    gene_rank = {gene: rank + 1 for rank, (gene, _) in enumerate(sorted_genes)}
    max_count = len(seeds) * 6  # 30
 
    # Build records for signature-overlapping genes
    sig_records = []
    for gene in genes_count:
        in_49  = gene in sig_49_overlap
        in_157 = gene in sig_157_overlap
        if in_49 or in_157:
            sig = "both signatures" if (in_49 and in_157) else ("49-gene signature" if in_49 else "157-gene signature")
            sig_records.append({"gene": gene, "rank": gene_rank[gene], "count": genes_count[gene], "sig": sig})
 
    sig_records.sort(key=lambda x: x["rank"])
 
    # ------------------------------------------------------------------
    # Build records for paper_genes-overlapping genes
    # ------------------------------------------------------------------
 
    paper_records = []
    for gene in genes_count:
        if gene in paper_genes_overlap:
            paper_records.append({"gene": gene, "rank": gene_rank[gene], "count": genes_count[gene]})
 
    paper_records.sort(key=lambda x: x["rank"])
 
    # ------------------------------------------------------------------
    # Plot — signatures
    # ------------------------------------------------------------------
 
    if len(sig_records) == 0:
        print(f"No genes selected in at least {fraction*100}% of cases overlap with signatures. Skipping signature plot.")
    else:
        color_map = {"49-gene signature": "#534AB7", "157-gene signature": "#0F6E56", "both signatures": "#993C1D"}
 
        genes  = [r["gene"]  for r in sig_records]
        counts = [r["count"] for r in sig_records]
        colors = [color_map[r["sig"]] for r in sig_records]
        ranks  = [r["rank"]  for r in sig_records]
 
        fig, ax = plt.subplots(figsize=(8, max(4, len(genes) * 0.35)))
 
        ax.barh(range(len(genes)), counts, color=colors, height=0.6)
 
        for i, (count, rank) in enumerate(zip(counts, ranks)):
            ax.text(count + 0.3, i, f"rank #{rank}", va="center", fontsize=8, color="#555")
 
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Selection count", fontsize=10)
        ax.set_xlim(0, max_count + 5)
        ax.axvline(max_count * fraction, color="gray", linestyle="--", linewidth=0.8, label=f"threshold ({fraction*100:.0f}%)")
 
        patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
        ax.legend(handles=patches, fontsize=9, loc="lower right")
 
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"/project_antwerp/baseline/ensemble_preprocessing/plots/signature_overlap_ranks_{fraction}_fraction.png", dpi=150, bbox_inches="tight")
        plt.show()
 
    # ------------------------------------------------------------------
    # Plot — paper_genes
    # ------------------------------------------------------------------
 
    if len(paper_records) == 0:
        print(f"No genes selected in at least {fraction*100}% of cases overlap with paper_genes. Skipping paper_genes plot.")
    else:
        paper_color = "#C2760C"
 
        genes  = [r["gene"]  for r in paper_records]
        counts = [r["count"] for r in paper_records]
        ranks  = [r["rank"]  for r in paper_records]
 
        fig, ax = plt.subplots(figsize=(8, max(4, len(genes) * 0.35)))
 
        ax.barh(range(len(genes)), counts, color=paper_color, height=0.6)
 
        for i, (count, rank) in enumerate(zip(counts, ranks)):
            ax.text(count + 0.3, i, f"rank #{rank}", va="center", fontsize=8, color="#555")
 
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Selection count", fontsize=10)
        ax.set_xlim(0, max_count + 5)
        ax.axvline(max_count * fraction, color="gray", linestyle="--", linewidth=0.8, label=f"threshold ({fraction*100:.0f}%)")
 
        patches = [mpatches.Patch(color=paper_color, label="paper_genes")]
        ax.legend(handles=patches, fontsize=9, loc="lower right")
 
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"/project_antwerp/baseline/ensemble_preprocessing/plots/paper_genes_overlap_ranks_{fraction}_fraction.png", dpi=150, bbox_inches="tight")
        plt.show()
