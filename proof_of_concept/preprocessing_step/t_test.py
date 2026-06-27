import h5py
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

def t_test(top_k=1000, chunk_size=10000, seeds=[11, 222, 333, 4444, 55555]):
    # --- CONFIGURATION ---
    hdf5_path = '/project_antwerp/TCGA-PRAD/data/outputs/TCGA/PRAD/merged_tile_preds.hdf5'
    metadata_path = '/project_antwerp/baseline/univariate_filtering/filtered_binarylabel_indices.npy'

    # loading data
    index_file = np.load(metadata_path, allow_pickle=True)
    groups = index_file[:, 0]
    labels = index_file[:, 2].astype(int)

    # get gene count from first tile
    patient_1, tile_1, gs_1 = index_file[0]
    with h5py.File(hdf5_path, 'r') as f:
        genes_count = f[patient_1][tile_1][:].shape[0]

    print(f"Total Database Size: {len(index_file)}")

    for seed in seeds:
        print("\n" + "#"*100)
        print(f"Processing Seed {seed}...")
        
        # -------------------------------------------------------
        # replicate data splits with same seeds
        # -------------------------------------------------------

        sgkf_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(sgkf_outer.split(index_file, labels, groups))

        X_train_val = index_file[train_val_idx]
        y_train_val = labels[train_val_idx]
        groups_train_val = groups[train_val_idx]

        test_indices_meta = index_file[test_idx]

        sgkf_inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        train_sub_idx, val_sub_idx = next(sgkf_inner.split(X_train_val, y_train_val, groups_train_val))

        train_indices_meta = X_train_val[train_sub_idx]
        val_indices_meta = X_train_val[val_sub_idx]

        print(f"Split Summary:")
        print(f"Train: {len(train_indices_meta)} tiles | Unique Patients: {len(np.unique(train_indices_meta[:,0]))}")
        print(f"Val:   {len(val_indices_meta)} tiles   | Unique Patients: {len(np.unique(val_indices_meta[:,0]))}")
        print(f"Test:  {len(test_indices_meta)} tiles  | Unique Patients: {len(np.unique(test_indices_meta[:,0]))}")

        # Verify no patient overlap
        train_pids = set(train_indices_meta[:, 0])
        val_pids = set(val_indices_meta[:, 0])
        test_pids = set(test_indices_meta[:, 0])

        assert train_pids.isdisjoint(val_pids), "Overlap between Train and Val patients!"
        assert train_pids.isdisjoint(test_pids), "Overlap between Train and Test patients!"
        assert val_pids.isdisjoint(test_pids), "Overlap between Val and Test patients!"
        print("Patient leakage check passed: No overlapping patients.")

        # -------------------------------------------------------
        # initialize Welford's stats
        # -------------------------------------------------------
        stats_dict = {
            0: {'n': 0, 'mean': np.zeros(genes_count), 'M2': np.zeros(genes_count)},
            1: {'n': 0, 'mean': np.zeros(genes_count), 'M2': np.zeros(genes_count)}
        }

        # -------------------------------------------------------
        # 3Welford with training data only
        # -------------------------------------------------------
        with h5py.File(hdf5_path, 'r') as f:
            
            total_chunks = int(np.ceil(len(train_indices_meta) / chunk_size))

            for i in range(0, len(train_indices_meta), chunk_size):
                chunk_meta = train_indices_meta[i : i + chunk_size]

                # Split labels
                for label in [0, 1]:
                    subset = chunk_meta[chunk_meta[:, 2].astype(int) == label]
                    
                    if len(subset) == 0:
                        continue

                    # Load tiles for this chunk
                    batch = np.vstack([f[p_id][t_id][:] for p_id, t_id, _ in subset])

                    k = batch.shape[0]

                    # Compute chunk mean
                    mean_chunk = np.mean(batch, axis=0)

                    # Compute chunk M2
                    M2_chunk = np.sum((batch - mean_chunk) ** 2, axis=0)

                    # Merge with global stats
                    s = stats_dict[label]

                    if s['n'] == 0:
                        s['n'] = k
                        s['mean'] = mean_chunk
                        s['M2'] = M2_chunk
                    else:
                        n = s['n']
                        delta = mean_chunk - s['mean']
                        new_n = n + k

                        s['mean'] += delta * (k / new_n)
                        s['M2'] += M2_chunk + (delta**2) * (n * k / new_n)
                        s['n'] = new_n
                        
                print(f"\r  Processed chunk {i // chunk_size + 1}/{total_chunks}", end='', flush=True)

        # -------------------------------------------------------
        # finalize Welford stats
        # -------------------------------------------------------
        def finalize_stats(s):
            # Safety check for single sample
            if s['n'] < 2: 
                return s['mean'], np.zeros_like(s['mean']), s['n']
            mean = s['mean']
            var = s['M2'] / (s['n'] - 1)
            return mean, var, s['n']

        mean0, var0, n0 = finalize_stats(stats_dict[0])
        mean1, var1, n1 = finalize_stats(stats_dict[1])

        # avoid division by zero
        epsilon = 1e-9
        denom = np.sqrt((var0/n0) + (var1/n1) + epsilon)
        
        t_stat = np.divide((mean0 - mean1), denom, out=np.zeros_like(mean0), where=denom!=0)

        # -------------------------------------------------------
        # Welch's t-test
        # -------------------------------------------------------
        vn0 = var0/n0
        vn1 = var1/n1
        df_num = (vn0 + vn1)**2
        df_den = (vn0**2 / (n0 - 1)) + (vn1**2 / (n1 - 1))
        
        # handle NaN in df calculation if variance is 0
        df = np.divide(df_num, df_den, out=np.ones_like(df_num), where=df_den!=0)

        p_values = stats.t.sf(np.abs(t_stat), df) * 2

        # pick top k genes
        top_indices = np.argsort(p_values)[:top_k]
        
        filename = f"top_{top_k}_genes_indices_welford_seed_{seed}.npy"
        np.save(filename, top_indices)
        print(f"\n  Saved: {filename}")



if __name__ == "__main__":
    top_k_list = [2000, 5000]
    for top_k in top_k_list:
        t_test(top_k=top_k)