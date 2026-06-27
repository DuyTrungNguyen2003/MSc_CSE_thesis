import os
import pickle
import pandas as pd
import numpy as np

def read_pickle(path):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects

def get_df(obj, column):
    all_dfs = []
    all_genes = obj['genes']

    for i in range(5):
        pat = obj[f'split_{i}']['patients']
        gene_exp = obj[f'split_{i}'][column]
        df = pd.DataFrame({'patient_id':pat,'gene_exp':gene_exp.squeeze().tolist()})
        df_split = pd.DataFrame(df['gene_exp'].to_list(), index=df.index)
        df_split.columns = all_genes
        combined = pd.concat([df, df_split], axis=1)
        combined = combined.drop(columns='gene_exp')
        all_dfs.append(combined)
        
    total_df = pd.concat(all_dfs).reset_index(drop=True)

    return total_df

'''
E.g.,
result_path = '/project_antwerp/TCGA-PRAD/test_results.pkl'
obj = read_pickle(result_path)[0]

# Load aggregated patient-level predictions 
df_preds = get_df(obj, "preds")
'''