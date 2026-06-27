import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import h5py
from collections import defaultdict

class GleasonTileDataset(Dataset):
    def __init__(self, hdf5_path, indices_array, scaler=None, feature_indices=None):
        """
        indices_array: (N, 3) -> [patient_id, tile_id, label]
        """
        self.hdf5_path = hdf5_path
        self.indices = indices_array
        self.scaler = scaler
        self.feature_indices = feature_indices
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        patient_id, tile_id, label = self.indices[idx]

        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
            
        gene_expressions = self.file[patient_id][tile_id][:]

        if self.feature_indices is not None:
            gene_expressions = gene_expressions[self.feature_indices]
        
        gene_expressions = gene_expressions.astype(np.float32)
            
        if self.scaler is not None:
            gene_expressions = self.scaler.transform(gene_expressions.reshape(1, -1)).flatten()

        gene_expressions_tensor = torch.from_numpy(gene_expressions)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return gene_expressions_tensor, label_tensor

class PatientTileSampler(Sampler):
    def __init__(self, dataset_indices, tiles_per_patient=10):
        """
        Args:
            dataset_indices (np.array): (N, 3) -> [patient_id, tile_id, label]
            tiles_per_patient (int): Number of tiles to sample per patient per epoch.
        """
        self.tiles_per_patient = tiles_per_patient
        self.dataset_indices = dataset_indices
        
        # Patient -> Label (0 or 1) -> [List of global dataset indices]
        self.patient_map = defaultdict(lambda: {0: [], 1: []})
        
        for global_idx, (pid, _, label) in enumerate(dataset_indices):
            # Convert label to int to use as dict key
            self.patient_map[pid][int(label)].append(global_idx)
            
        self.patient_ids = list(self.patient_map.keys())
        self.num_samples = len(self.patient_ids) * self.tiles_per_patient

    def __iter__(self):
        batch_indices = []
        
        # Shuffle patient order for variety across epochs
        shuffled_pids = list(self.patient_ids)
        np.random.shuffle(shuffled_pids)
        
        for pid in shuffled_pids:
            idxs_0 = self.patient_map[pid][0]
            idxs_1 = self.patient_map[pid][1]
            
            n_avail_0 = len(idxs_0)
            n_avail_1 = len(idxs_1)
            total_avail = n_avail_0 + n_avail_1
            
            if total_avail == 0: continue
            
            # Calculate ratio based on available tiles for this specific patient
            ratio_0 = n_avail_0 / total_avail
            count_0 = int(np.round(self.tiles_per_patient * ratio_0))
            count_1 = self.tiles_per_patient - count_0
            
            def sample_ids(source_indices, count):
                if count <= 0 or len(source_indices) == 0: 
                    return []
                
                # If we need more than available, use all shuffled. 
                # Otherwise, sample without replacement.
                if len(source_indices) > count:
                    return np.random.choice(source_indices, count, replace=False).tolist()
                else:
                    res = list(source_indices)
                    np.random.shuffle(res)
                    return res

            selected_0 = sample_ids(idxs_0, count_0)
            selected_1 = sample_ids(idxs_1, count_1)
            
            current_selection = selected_0 + selected_1
            batch_indices.extend(current_selection)

        return iter(batch_indices)

    def __len__(self):
        return self.num_samples