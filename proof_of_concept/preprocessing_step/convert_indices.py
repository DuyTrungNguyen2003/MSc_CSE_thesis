# old npy structure: patient_id, tile_id, gs_label (GSx+y)

import numpy as np

gs_to_label = { 'GS3+3': 0, 
                # 'GS3+4': 0, 'GS4+3': 1, 
                'GS4+4': 1, 'GS3+5': 1, 'GS5+3': 1, 'GS4+5': 1, 'GS5+4': 1, 'GS5+5': 1}

def filter_and_convert_indices(original_indices):
    filtered_list = []
    
    for entry in original_indices:
        patient_id, tile_id, gs_str = entry
        
        if gs_str != 'GS3+4' and gs_str != 'GS4+3':  # Exclude GS3+4 and GS4+3
            # Convert GS string integer label
            label = gs_to_label[gs_str]
            # new npy structure: patient_id, tile_id, label (0 or 1)
            filtered_list.append([patient_id, tile_id, label])
            
    return np.array(filtered_list, dtype=object)

if __name__ == "__main__":
    old_indices = np.load('/project_antwerp/baseline/univariate_filtering/GS_merged_tile_indices.npy', allow_pickle=True)
    new_indices = filter_and_convert_indices(old_indices)
    np.save('/project_antwerp/baseline/univariate_filtering/filtered_binarylabel_indices.npy', new_indices)
    print(len(old_indices), len(new_indices))