import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import h5py
import re
from collections import defaultdict

gs_to_label = {'GS3+3': 0, 'GS3+4': 0, 'GS4+3': 1, 
               'GS4+4': 1, 'GS3+5': 1, 'GS5+3': 1, 'GS4+5': 1, 'GS5+4': 1, 'GS5+5': 1}

class GleasonTileDataset(Dataset):
    def __init__(self, hdf5_path, indices_array, scaler=None, feature_indices=None):
        self.hdf5_path = hdf5_path
        self.indices = indices_array
        self.scaler = scaler
        self.feature_indices = feature_indices
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        patient_id, tile_id, gs_label = self.indices[idx]
        label = gs_to_label[gs_label]

        if self.file == None:
            self.file = h5py.File(self.hdf5_path, 'r')
            
        gene_expressions = self.file[patient_id][tile_id][:]

        if self.feature_indices is not None:
            gene_expressions = gene_expressions[self.feature_indices]
        
        gene_expressions = gene_expressions.astype(np.float32)
            
        if self.scaler != None:
            gene_expressions = self.scaler.transform(gene_expressions.reshape(1, -1)).flatten()

        gene_expressions_tensor = torch.from_numpy(gene_expressions)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return gene_expressions_tensor, label_tensor

class PatientTileSampler(Sampler):
    def __init__(self, dataset_indices, gs_to_label, tiles_per_patient=10):
        """
        Args:
            dataset_indices (np.array): The same array passed to the Dataset (N, 3).
            gs_to_label (dict): Mapping from GS string to 0/1 label.
            tiles_per_patient (int): Number of tiles to sample per patient per epoch.
        """
        self.tiles_per_patient = tiles_per_patient
        self.dataset_indices = dataset_indices
        
        # Patient -> Label -> [List of dataset indices]
        self.patient_map = defaultdict(lambda: {0: [], 1: []})
        
        for global_idx, (pid, _, gs_str) in enumerate(dataset_indices):
            label = gs_to_label[gs_str]
            self.patient_map[pid][label].append(global_idx)
            
        self.patient_ids = list(self.patient_map.keys())
        self.num_samples = len(self.patient_ids) * self.tiles_per_patient

    def __iter__(self):
        batch_indices = []
        
        # (Optional, but good for Batch Norm stability if batch_size < num_patients)
        np.random.shuffle(self.patient_ids)
        
        for pid in self.patient_ids:
            idxs_0 = self.patient_map[pid][0]
            idxs_1 = self.patient_map[pid][1]
            
            n_avail_0 = len(idxs_0)
            n_avail_1 = len(idxs_1)
            total_avail = n_avail_0 + n_avail_1
            
            if total_avail == 0: continue
            
            # class ratio
            ratio_0 = n_avail_0 / total_avail
            
            count_0 = int(np.round(self.tiles_per_patient * ratio_0))
            count_1 = self.tiles_per_patient - count_0
            
            # sample indices
            def sample_ids(source_indices, count):
                if count == 0 or len(source_indices) == 0: 
                    return []
                
                # replace = len(source_indices) < count
                # return np.random.choice(source_indices, count, replace=replace).tolist()
                sample = len(source_indices) > count
                if (sample == True):
                    return np.random.choice(source_indices, count, replace=False).tolist()
                else:
                    same_indices =list(source_indices)
                    np.random.shuffle(same_indices)
                    return same_indices

            selected_0 = sample_ids(idxs_0, count_0)
            selected_1 = sample_ids(idxs_1, count_1)
            
            current_selection = selected_0 + selected_1
            batch_indices.extend(current_selection)

        return iter(batch_indices)

    def __len__(self):
        # upper limit
        return self.num_samples