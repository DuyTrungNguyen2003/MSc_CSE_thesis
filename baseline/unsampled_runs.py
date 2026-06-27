import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, 
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm
import copy
import random

from data import GleasonTileDataset
from signatures import get_signature
from utils import read_pickle, get_df

import warnings
warnings.filterwarnings('ignore')


def run_raw(seed):
    # ---------------------------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ---------------------------------------------------------------------------------------------

    # hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    PATIENCE = 5
    CLASSIFICATION_THRESHOLD = 0.5
    OUTPUT_SIZE = 1
    INPUT_SIZE = 25587

    # ---------------------------------------------------------------------------------------------

    hdf5_path = '/project_antwerp/TCGA-PRAD/outputs/TCGA/PRAD/merged_tile_preds.hdf5'
    index_path = '/project_antwerp/baseline/averaging_baseline/GS_merged_tile_indices.npy'

    log_file_path = f'/project_antwerp/baseline/averaging_baseline/logs/training_log/log_raw_{seed}.txt'
    test_metrics_path = f'/project_antwerp/baseline/averaging_baseline/logs/test_metric/metrics_raw_{seed}.txt'
    confusion_matrix_path = f'/project_antwerp/baseline/averaging_baseline/logs/CM/CM_test_raw_{seed}.png'
    roc_curve_path = f'/project_antwerp/baseline/averaging_baseline/logs/ROC_plot/ROC_raw_{seed}.png'

    index_file = np.load(index_path, allow_pickle=True)

    ####################################
    # GS 7 out
    remove_scores = {"GS3+4", "GS4+3"}
    index_file = np.array([entry for entry in index_file if entry[2] not in remove_scores], dtype=object)
    ####################################

    gs_to_label = {'GS3+3': 0, 'GS3+4': 0, 'GS4+3': 1, 'GS4+4': 1, 'GS3+5': 1, 'GS5+3': 1, 'GS4+5': 1, 'GS5+4': 1, 'GS5+5': 1}
    label_to_gs = {v: k for k, v in gs_to_label.items()}

    # ---------------------------------------------------------------------------------------------

    # PATIENT SPLITS

    groups = index_file[:, 0]
    labels = np.array([gs_to_label[k] for k in index_file[:, 2]])

    sgkf_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(index_file, labels, groups))

    X_train_val = index_file[train_val_idx]
    y_train_val = labels[train_val_idx]
    groups_train_val = groups[train_val_idx]

    test_indices = index_file[test_idx]

    sgkf_inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(sgkf_inner.split(X_train_val, y_train_val, groups_train_val))

    train_indices = X_train_val[train_sub_idx]
    val_indices = X_train_val[val_sub_idx]

    print(f"Split Summary:")
    print(f"Train: {len(train_indices)} tiles | Unique Patients: {len(np.unique(train_indices[:,0]))}")
    print(f"Val:   {len(val_indices)} tiles   | Unique Patients: {len(np.unique(val_indices[:,0]))}")
    print(f"Test:  {len(test_indices)} tiles  | Unique Patients: {len(np.unique(test_indices[:,0]))}")

    # Verify no patient overlap
    train_pids = set(train_indices[:, 0])
    val_pids = set(val_indices[:, 0])
    test_pids = set(test_indices[:, 0])

    assert train_pids.isdisjoint(val_pids), "Overlap between Train and Val patients!"
    assert train_pids.isdisjoint(test_pids), "Overlap between Train and Test patients!"
    assert val_pids.isdisjoint(test_pids), "Overlap between Val and Test patients!"
    print("Patient leakage check passed: No overlapping patients.")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    scaler = StandardScaler()

    print("Fitting scaler on training data...")
    for X_batch, _ in tqdm(train_loader, desc="Fitting Scaler", unit="batch"):
        scaler.partial_fit(X_batch.numpy())
    print("Scaler fitted!")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices, scaler=scaler)
    val_dataset = GleasonTileDataset(hdf5_path, val_indices, scaler=scaler)
    test_dataset = GleasonTileDataset(hdf5_path, test_indices, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)

    # ---------------------------------------------------------------------------------------------

    # Training function

    def train_one_epoch(dataloader, model, loss_fn, optimizer):
        model.train()
        running_loss = 0.0
        for X, y in dataloader:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            y = y.float().unsqueeze(1) 
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        return running_loss / len(dataloader.dataset)

    # Validation function
    def validate(dataloader, model, loss_fn):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                y = y.float().unsqueeze(1) 
                logits = model(X)
                running_loss += loss_fn(logits, y).item() * X.size(0)
                
                probs = torch.sigmoid(logits)
                
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())

        avg_loss = running_loss / len(dataloader.dataset)
        y_true = torch.cat(all_labels).numpy()
        y_probs = torch.cat(all_probs).numpy()

        try:
            val_auroc = roc_auc_score(y_true, y_probs)
        except ValueError:
            # if batch has only one class
            val_auroc = 0.0

        return avg_loss, val_auroc

    # ---------------------------------------------------------------------------------------------

    # model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
            super().__init__()
            self.output_layer = nn.Linear(input_size, output_size)

        def forward(self, x):
            logits = self.output_layer(x)
            return logits
            
    # ---------------------------------------------------------------------------------------------

    # CLASS WEIGHTS
    train_labels = y_train_val[train_sub_idx]
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    print(f"Training Set Class Balance:")
    print(f"Negative (Class 0): {n_neg}")
    print(f"Positive (Class 1): {n_pos}")
    pos_weight_val = n_neg / (n_pos)
    print(f"Computed positive class weight: {pos_weight_val:.4f}")
    CLASS_WEIGHT = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    # ---------------------------------------------------------------------------------------------

    model = LogisticRegression(input_size=INPUT_SIZE).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHT) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping_metric = float("inf")
    epochs_no_improve = 0
    best_model_weights = None

    # ---------------------------------------------------------------------------------------------

    print("Starting training...\n")
    # --- Training loop ---

    total_start_time = time.time()

    with open(log_file_path, 'w') as log_file:
        for epoch in range(EPOCHS):

            start_time = time.time()
            
            train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer)
            val_loss, val_auroc = validate(val_loader, model, loss_fn)

            epoch_time = time.time() - start_time
        
            log_msg = (f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | "
                    f"Time: {epoch_time:.2f}s\n")
            print(log_msg)
            log_file.write(log_msg)
        
            if val_loss < early_stopping_metric:
                early_stopping_metric = val_loss
                epochs_no_improve = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
        
            if epochs_no_improve == PATIENCE:
                log_file.write("Early stopping triggered")
                print("Early stopping triggered")
                break
                
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.2f} seconds " f"({total_time/60:.2f} minutes)\n")   
            
    # ---------------------------------------------------------------------------------------------

    # --- Test evaluation ---

    model.load_state_dict(best_model_weights)
    model.eval()

    # Helper function to get predictions
    def get_predictions(loader, model):
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = model(X)
                probs = torch.sigmoid(logits)
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())
        y_true = torch.cat(all_labels).numpy().flatten()
        y_probs = torch.cat(all_probs).numpy().flatten()
        return y_true, y_probs

    print("Evaluating on Test set...")
    y_test_true, y_test_probs = get_predictions(test_loader, model)

    test_auroc = roc_auc_score(y_test_true, y_test_probs)

    y_test_preds = (y_test_probs > CLASSIFICATION_THRESHOLD).astype(float)
    cm = confusion_matrix(y_test_true, y_test_preds)

    print(f"Test AUROC: {test_auroc:.4f}")

    with open(test_metrics_path, 'w') as f:
        f.write(f"Test AUROC: {test_auroc:.4f}\n")

    # Save Test CM
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # PLOTS (ROC)
    print("Generating Test Curves...")

    # --- Plot ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test_true, y_test_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved to {roc_curve_path}")

    return test_auroc


def run_L1(seed):
    # ---------------------------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ---------------------------------------------------------------------------------------------

    # hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    PATIENCE = 5
    CLASSIFICATION_THRESHOLD = 0.5
    OUTPUT_SIZE = 1
    INPUT_SIZE = 25587

    # L1 regularisation
    L1_LAMBDA = 1e-3

    # ---------------------------------------------------------------------------------------------

    hdf5_path = '/project_antwerp/TCGA-PRAD/outputs/TCGA/PRAD/merged_tile_preds.hdf5'
    index_path = '/project_antwerp/baseline/averaging_baseline/GS_merged_tile_indices.npy'

    log_file_path = f'/project_antwerp/baseline/averaging_baseline/logs/training_log/log_L1_{seed}.txt'
    test_metrics_path = f'/project_antwerp/baseline/averaging_baseline/logs/test_metric/metrics_L1_{seed}.txt'
    confusion_matrix_path = f'/project_antwerp/baseline/averaging_baseline/logs/CM/CM_L1_{seed}.png'
    roc_curve_path = f'/project_antwerp/baseline/averaging_baseline/logs/ROC_plot/ROC_L1_{seed}.png'

    index_file = np.load(index_path, allow_pickle=True)

    ####################################
    # GS 7 out
    remove_scores = {"GS3+4", "GS4+3"}
    index_file = np.array([entry for entry in index_file if entry[2] not in remove_scores], dtype=object)
    ####################################

    gs_to_label = {'GS3+3': 0, 'GS3+4': 0, 'GS4+3': 1, 'GS4+4': 1, 'GS3+5': 1, 'GS5+3': 1, 'GS4+5': 1, 'GS5+4': 1, 'GS5+5': 1}
    label_to_gs = {v: k for k, v in gs_to_label.items()}

    # ---------------------------------------------------------------------------------------------

    # PATIENT SPLITS

    groups = index_file[:, 0]
    labels = np.array([gs_to_label[k] for k in index_file[:, 2]])

    sgkf_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(index_file, labels, groups))

    X_train_val = index_file[train_val_idx]
    y_train_val = labels[train_val_idx]
    groups_train_val = groups[train_val_idx]

    test_indices = index_file[test_idx]

    sgkf_inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(sgkf_inner.split(X_train_val, y_train_val, groups_train_val))

    train_indices = X_train_val[train_sub_idx]
    val_indices = X_train_val[val_sub_idx]

    print(f"Split Summary:")
    print(f"Train: {len(train_indices)} tiles | Unique Patients: {len(np.unique(train_indices[:,0]))}")
    print(f"Val:   {len(val_indices)} tiles   | Unique Patients: {len(np.unique(val_indices[:,0]))}")
    print(f"Test:  {len(test_indices)} tiles  | Unique Patients: {len(np.unique(test_indices[:,0]))}")

    # Verify no patient overlap
    train_pids = set(train_indices[:, 0])
    val_pids = set(val_indices[:, 0])
    test_pids = set(test_indices[:, 0])

    assert train_pids.isdisjoint(val_pids), "Overlap between Train and Val patients!"
    assert train_pids.isdisjoint(test_pids), "Overlap between Train and Test patients!"
    assert val_pids.isdisjoint(test_pids), "Overlap between Val and Test patients!"
    print("Patient leakage check passed: No overlapping patients.")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    scaler = StandardScaler()

    print("Fitting scaler on training data...")
    for X_batch, _ in tqdm(train_loader, desc="Fitting Scaler", unit="batch"):
        scaler.partial_fit(X_batch.numpy())
    print("Scaler fitted!")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices, scaler=scaler)
    val_dataset = GleasonTileDataset(hdf5_path, val_indices, scaler=scaler)
    test_dataset = GleasonTileDataset(hdf5_path, test_indices, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)

    # ---------------------------------------------------------------------------------------------

    # Training function

    def train_one_epoch(dataloader, model, loss_fn, optimizer, l1_lambda):
        model.train()
        running_loss = 0.0
        for X, y in dataloader:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            y = y.float().unsqueeze(1) 
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)

            #################################################
            # L1 Regularization
            l1_penalty = 0
            l1_penalty = sum(torch.sum(torch.abs(p)) for p in model.parameters())
                
            loss += l1_lambda * l1_penalty
            #################################################
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        return running_loss / len(dataloader.dataset)

    # Validation function
    def validate(dataloader, model, loss_fn):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                y = y.float().unsqueeze(1) 
                logits = model(X)
                running_loss += loss_fn(logits, y).item() * X.size(0)
                
                probs = torch.sigmoid(logits)
                
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())

        avg_loss = running_loss / len(dataloader.dataset)
        y_true = torch.cat(all_labels).numpy()
        y_probs = torch.cat(all_probs).numpy()

        try:
            val_auroc = roc_auc_score(y_true, y_probs)
        except ValueError:
            # if batch has only one class
            val_auroc = 0.0

        return avg_loss, val_auroc

    # ---------------------------------------------------------------------------------------------

    # model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
            super().__init__()
            self.output_layer = nn.Linear(input_size, output_size)

        def forward(self, x):
            logits = self.output_layer(x)
            return logits

    # ---------------------------------------------------------------------------------------------

    # CLASS WEIGHTS
    train_labels = y_train_val[train_sub_idx]
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    print(f"Training Set Class Balance:")
    print(f"Negative (Class 0): {n_neg}")
    print(f"Positive (Class 1): {n_pos}")
    pos_weight_val = n_neg / (n_pos)
    print(f"Computed positive class weight: {pos_weight_val:.4f}")
    CLASS_WEIGHT = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    # ---------------------------------------------------------------------------------------------

    model = LogisticRegression(input_size=INPUT_SIZE).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHT) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping_metric = float("inf")
    epochs_no_improve = 0
    best_model_weights = None

    # ---------------------------------------------------------------------------------------------

    print("Starting training...\n")
    # --- Training loop ---

    total_start_time = time.time()

    with open(log_file_path, 'w') as log_file:
        for epoch in range(EPOCHS):

            start_time = time.time()
            
            train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, L1_LAMBDA)
            val_loss, val_auroc = validate(val_loader, model, loss_fn)

            epoch_time = time.time() - start_time
        
            log_msg = (f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | "
                    f"Time: {epoch_time:.2f}s\n")
            print(log_msg)
            log_file.write(log_msg)
        
            if val_loss < early_stopping_metric:
                early_stopping_metric = val_loss
                epochs_no_improve = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
        
            if epochs_no_improve == PATIENCE:
                log_file.write("Early stopping triggered")
                print("Early stopping triggered")
                break
                
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.2f} seconds " f"({total_time/60:.2f} minutes)\n")   
            
    # ---------------------------------------------------------------------------------------------

    # --- Test evaluation ---

    model.load_state_dict(best_model_weights)
    model.eval()

    # Helper function to get predictions
    def get_predictions(loader, model):
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = model(X)
                probs = torch.sigmoid(logits)
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())
        y_true = torch.cat(all_labels).numpy().flatten()
        y_probs = torch.cat(all_probs).numpy().flatten()
        return y_true, y_probs

    print("Evaluating on Test set...")
    y_test_true, y_test_probs = get_predictions(test_loader, model)

    test_auroc = roc_auc_score(y_test_true, y_test_probs)

    y_test_preds = (y_test_probs > CLASSIFICATION_THRESHOLD).astype(float)
    cm = confusion_matrix(y_test_true, y_test_preds)

    print(f"Test AUROC: {test_auroc:.4f}")

    with open(test_metrics_path, 'w') as f:
        f.write(f"Test AUROC: {test_auroc:.4f}\n")

    # Save Test CM
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # PLOTS (ROC)
    print("Generating Test Curves...")

    # --- Plot ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test_true, y_test_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved to {roc_curve_path}")

    return test_auroc


def run_sig49(seed):
    # ---------------------------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ---------------------------------------------------------------------------------------------

    result_path = '/project_antwerp/TCGA-PRAD/test_results.pkl'
    obj = read_pickle(result_path)[0]

    # Load aggregated patient-level predictions 
    df_preds = get_df(obj, "preds")

    # Get column names except the first column
    all_gene_names = df_preds.columns[1:].tolist()
    print(len(all_gene_names))

    signature = [
        "rna_PDGFB", "rna_ASPN", "rna_FOXS1", "rna_SMC4", "rna_FAM72B", "rna_ITGBL1", "rna_LPPR4", "rna_SPAG1",
        "rna_BUB1", "rna_GOLGA7B", "rna_CENPF", "rna_GDF3", "rna_MAPK8IP2", "rna_ESM1", "rna_PRC1", "rna_MYT1",
        "rna_LRFN2", "rna_SHCBP1", "rna_AHRR", "rna_CBX2", "rna_GMNN", "rna_NUF2", "rna_STC2", "rna_RAI14",
        "rna_FGF14", "rna_ZNF467", "rna_TMEM132E", "rna_FAM72D", "rna_CST2", "rna_KIF14", "rna_APLNR", "rna_DLGAP5",
        "rna_CENPE", "rna_IGSF1", "rna_NAAA", "rna_ASPA", "rna_SLC22A1", "rna_TAOK3", "rna_C2orf88", "rna_NCAPD3",
        "rna_GLB1L3", "rna_PAGE4", "rna_ANO7", "rna_EDN3", "rna_TPT1", "rna_ADPGK", "rna_PACSIN3", "rna_GLB1L2",
        "rna_PLOD1"
    ]

    signature_size = len(signature)
    print(signature_size)

    target_genes_set = set(signature)
    keep_indices = [i for i, gene in enumerate(all_gene_names) if gene in target_genes_set]

    # ---------------------------------------------------------------------------------------------
    # hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    PATIENCE = 5
    CLASSIFICATION_THRESHOLD = 0.5
    OUTPUT_SIZE = 1
    INPUT_SIZE = len(keep_indices)

    # ---------------------------------------------------------------------------------------------

    hdf5_path = '/project_antwerp/TCGA-PRAD/outputs/TCGA/PRAD/merged_tile_preds.hdf5'
    index_path = '/project_antwerp/baseline/averaging_baseline/GS_merged_tile_indices.npy'

    log_file_path = f'/project_antwerp/baseline/averaging_baseline/logs/training_log/log_sig49_{seed}.txt'
    test_metrics_path = f'/project_antwerp/baseline/averaging_baseline/logs/test_metric/metrics_sig49_{seed}.txt'
    confusion_matrix_path = f'/project_antwerp/baseline/averaging_baseline/logs/CM/CM_sig49_{seed}.png'
    roc_curve_path = f'/project_antwerp/baseline/averaging_baseline/logs/ROC_plot/ROC_sig49_{seed}.png'

    index_file = np.load(index_path, allow_pickle=True)

    ####################################
    # GS 7 out
    remove_scores = {"GS3+4", "GS4+3"}
    index_file = np.array([entry for entry in index_file if entry[2] not in remove_scores], dtype=object)
    ####################################

    gs_to_label = {'GS3+3': 0, 'GS3+4': 0, 'GS4+3': 1, 'GS4+4': 1, 'GS3+5': 1, 'GS5+3': 1, 'GS4+5': 1, 'GS5+4': 1, 'GS5+5': 1}
    label_to_gs = {v: k for k, v in gs_to_label.items()}

    # ---------------------------------------------------------------------------------------------

    # PATIENT SPLITS

    groups = index_file[:, 0]
    labels = np.array([gs_to_label[k] for k in index_file[:, 2]])

    sgkf_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(index_file, labels, groups))

    X_train_val = index_file[train_val_idx]
    y_train_val = labels[train_val_idx]
    groups_train_val = groups[train_val_idx]

    test_indices = index_file[test_idx]

    sgkf_inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(sgkf_inner.split(X_train_val, y_train_val, groups_train_val))

    train_indices = X_train_val[train_sub_idx]
    val_indices = X_train_val[val_sub_idx]

    print(f"Split Summary:")
    print(f"Train: {len(train_indices)} tiles | Unique Patients: {len(np.unique(train_indices[:,0]))}")
    print(f"Val:   {len(val_indices)} tiles   | Unique Patients: {len(np.unique(val_indices[:,0]))}")
    print(f"Test:  {len(test_indices)} tiles  | Unique Patients: {len(np.unique(test_indices[:,0]))}")

    # Verify no patient overlap
    train_pids = set(train_indices[:, 0])
    val_pids = set(val_indices[:, 0])
    test_pids = set(test_indices[:, 0])

    assert train_pids.isdisjoint(val_pids), "Overlap between Train and Val patients!"
    assert train_pids.isdisjoint(test_pids), "Overlap between Train and Test patients!"
    assert val_pids.isdisjoint(test_pids), "Overlap between Val and Test patients!"
    print("Patient leakage check passed: No overlapping patients.")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices, feature_indices=keep_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    scaler = StandardScaler()

    print("Fitting scaler on training data...")
    for X_batch, _ in tqdm(train_loader, desc="Fitting Scaler", unit="batch"):
        scaler.partial_fit(X_batch.numpy())
    print("Scaler fitted!")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices, scaler=scaler, feature_indices=keep_indices)
    val_dataset = GleasonTileDataset(hdf5_path, val_indices, scaler=scaler, feature_indices=keep_indices)
    test_dataset = GleasonTileDataset(hdf5_path, test_indices, scaler=scaler, feature_indices=keep_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)

    # ---------------------------------------------------------------------------------------------

    # Training function

    def train_one_epoch(dataloader, model, loss_fn, optimizer):
        model.train()
        running_loss = 0.0
        for X, y in dataloader:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            y = y.float().unsqueeze(1) 
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        return running_loss / len(dataloader.dataset)

    # Validation function
    def validate(dataloader, model, loss_fn):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                y = y.float().unsqueeze(1) 
                logits = model(X)
                running_loss += loss_fn(logits, y).item() * X.size(0)
                
                probs = torch.sigmoid(logits)
                
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())

        avg_loss = running_loss / len(dataloader.dataset)
        y_true = torch.cat(all_labels).numpy()
        y_probs = torch.cat(all_probs).numpy()

        try:
            val_auroc = roc_auc_score(y_true, y_probs)
        except ValueError:
            # if batch has only one class
            val_auroc = 0.0
        
        return avg_loss, val_auroc

    # ---------------------------------------------------------------------------------------------

    # model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
            super().__init__()
            self.output_layer = nn.Linear(input_size, output_size)

        def forward(self, x):
            logits = self.output_layer(x)
            return logits
            
    # ---------------------------------------------------------------------------------------------

    # CLASS WEIGHTS
    train_labels = y_train_val[train_sub_idx]
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    print(f"Training Set Class Balance:")
    print(f"Negative (Class 0): {n_neg}")
    print(f"Positive (Class 1): {n_pos}")
    pos_weight_val = n_neg / (n_pos)
    print(f"Computed positive class weight: {pos_weight_val:.4f}")
    CLASS_WEIGHT = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    # ---------------------------------------------------------------------------------------------

    model = LogisticRegression(input_size=INPUT_SIZE).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHT) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping_metric = float("inf")
    epochs_no_improve = 0
    best_model_weights = None

    # ---------------------------------------------------------------------------------------------

    print("Starting training...\n")
    # --- Training loop ---

    total_start_time = time.time()

    with open(log_file_path, 'w') as log_file:
        for epoch in range(EPOCHS):

            start_time = time.time()
            
            train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer)
            val_loss, val_auroc = validate(val_loader, model, loss_fn)

            epoch_time = time.time() - start_time
        
            log_msg = (f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | "
                    f"Time: {epoch_time:.2f}s\n")
            print(log_msg)
            log_file.write(log_msg)
        
            if val_loss < early_stopping_metric:
                early_stopping_metric = val_loss
                epochs_no_improve = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
        
            if epochs_no_improve == PATIENCE:
                log_file.write("Early stopping triggered")
                print("Early stopping triggered")
                break
                
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.2f} seconds " f"({total_time/60:.2f} minutes)\n")   
            
    # ---------------------------------------------------------------------------------------------

    # --- Test evaluation ---

    model.load_state_dict(best_model_weights)
    model.eval()

    # Helper function to get predictions
    def get_predictions(loader, model):
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = model(X)
                probs = torch.sigmoid(logits)
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())
        y_true = torch.cat(all_labels).numpy().flatten()
        y_probs = torch.cat(all_probs).numpy().flatten()
        return y_true, y_probs

    print("Evaluating on Test set...")
    y_test_true, y_test_probs = get_predictions(test_loader, model)

    test_auroc = roc_auc_score(y_test_true, y_test_probs)

    y_test_preds = (y_test_probs > CLASSIFICATION_THRESHOLD).astype(float)
    cm = confusion_matrix(y_test_true, y_test_preds)

    print(f"Test AUROC: {test_auroc:.4f}")

    with open(test_metrics_path, 'w') as f:
        f.write(f"Test AUROC: {test_auroc:.4f}\n")

    # Save Test CM
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # PLOTS (ROC)
    print("Generating Test Curves...")

    # --- Plot ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test_true, y_test_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved to {roc_curve_path}")

    return test_auroc


def run_sig157(seed):
    # ---------------------------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ---------------------------------------------------------------------------------------------

    result_path = '/project_antwerp/TCGA-PRAD/test_results.pkl'
    obj = read_pickle(result_path)[0]

    # Load aggregated patient-level predictions 
    df_preds = get_df(obj, "preds")

    # Get column names except the first column
    all_gene_names = df_preds.columns[1:].tolist()
    print(len(all_gene_names))

    signature = [
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

    signature_size = len(signature)
    print(signature_size)

    target_genes_set = set(signature)
    keep_indices = [i for i, gene in enumerate(all_gene_names) if gene in target_genes_set]

    # ---------------------------------------------------------------------------------------------
    # hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    PATIENCE = 5
    CLASSIFICATION_THRESHOLD = 0.5
    OUTPUT_SIZE = 1
    INPUT_SIZE = len(keep_indices)

    # ---------------------------------------------------------------------------------------------

    hdf5_path = '/project_antwerp/TCGA-PRAD/outputs/TCGA/PRAD/merged_tile_preds.hdf5'
    index_path = '/project_antwerp/baseline/averaging_baseline/GS_merged_tile_indices.npy'

    log_file_path = f'/project_antwerp/baseline/averaging_baseline/logs/training_log/log_sig157_{seed}.txt'
    test_metrics_path = f'/project_antwerp/baseline/averaging_baseline/logs/test_metric/metrics_sig157_{seed}.txt'
    confusion_matrix_path = f'/project_antwerp/baseline/averaging_baseline/logs/CM/CM_test_sig157_{seed}.png'
    roc_curve_path = f'/project_antwerp/baseline/averaging_baseline/logs/ROC_plot/ROC_sig157_{seed}.png'

    index_file = np.load(index_path, allow_pickle=True)

    ####################################
    # GS 7 out
    remove_scores = {"GS3+4", "GS4+3"}
    index_file = np.array([entry for entry in index_file if entry[2] not in remove_scores], dtype=object)
    ####################################

    gs_to_label = {'GS3+3': 0, 'GS3+4': 0, 'GS4+3': 1, 'GS4+4': 1, 'GS3+5': 1, 'GS5+3': 1, 'GS4+5': 1, 'GS5+4': 1, 'GS5+5': 1}
    label_to_gs = {v: k for k, v in gs_to_label.items()}

    # ---------------------------------------------------------------------------------------------

    # PATIENT SPLITS

    groups = index_file[:, 0]
    labels = np.array([gs_to_label[k] for k in index_file[:, 2]])

    sgkf_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(index_file, labels, groups))

    X_train_val = index_file[train_val_idx]
    y_train_val = labels[train_val_idx]
    groups_train_val = groups[train_val_idx]

    test_indices = index_file[test_idx]

    sgkf_inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(sgkf_inner.split(X_train_val, y_train_val, groups_train_val))

    train_indices = X_train_val[train_sub_idx]
    val_indices = X_train_val[val_sub_idx]

    print(f"Split Summary:")
    print(f"Train: {len(train_indices)} tiles | Unique Patients: {len(np.unique(train_indices[:,0]))}")
    print(f"Val:   {len(val_indices)} tiles   | Unique Patients: {len(np.unique(val_indices[:,0]))}")
    print(f"Test:  {len(test_indices)} tiles  | Unique Patients: {len(np.unique(test_indices[:,0]))}")

    # Verify no patient overlap
    train_pids = set(train_indices[:, 0])
    val_pids = set(val_indices[:, 0])
    test_pids = set(test_indices[:, 0])

    assert train_pids.isdisjoint(val_pids), "Overlap between Train and Val patients!"
    assert train_pids.isdisjoint(test_pids), "Overlap between Train and Test patients!"
    assert val_pids.isdisjoint(test_pids), "Overlap between Val and Test patients!"
    print("Patient leakage check passed: No overlapping patients.")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices, feature_indices=keep_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    scaler = StandardScaler()

    print("Fitting scaler on training data...")
    for X_batch, _ in tqdm(train_loader, desc="Fitting Scaler", unit="batch"):
        scaler.partial_fit(X_batch.numpy())
    print("Scaler fitted!")

    # ---------------------------------------------------------------------------------------------

    train_dataset = GleasonTileDataset(hdf5_path, train_indices, scaler=scaler, feature_indices=keep_indices)
    val_dataset = GleasonTileDataset(hdf5_path, val_indices, scaler=scaler, feature_indices=keep_indices)
    test_dataset = GleasonTileDataset(hdf5_path, test_indices, scaler=scaler, feature_indices=keep_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=8, prefetch_factor=4)

    # ---------------------------------------------------------------------------------------------

    # Training function

    def train_one_epoch(dataloader, model, loss_fn, optimizer):
        model.train()
        running_loss = 0.0
        for X, y in dataloader:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            y = y.float().unsqueeze(1) 
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        return running_loss / len(dataloader.dataset)

    # Validation function
    def validate(dataloader, model, loss_fn):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                y = y.float().unsqueeze(1) 
                logits = model(X)
                running_loss += loss_fn(logits, y).item() * X.size(0)
                
                probs = torch.sigmoid(logits)
                
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())

        avg_loss = running_loss / len(dataloader.dataset)
        y_true = torch.cat(all_labels).numpy()
        y_probs = torch.cat(all_probs).numpy()

        try:
            val_auroc = roc_auc_score(y_true, y_probs)
        except ValueError:
            # if batch has only one class
            val_auroc = 0.0
        
        return avg_loss, val_auroc

    # ---------------------------------------------------------------------------------------------

    # model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
            super().__init__()
            self.output_layer = nn.Linear(input_size, output_size)

        def forward(self, x):
            logits = self.output_layer(x)
            return logits
            
    # ---------------------------------------------------------------------------------------------

    # CLASS WEIGHTS
    train_labels = y_train_val[train_sub_idx]
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    print(f"Training Set Class Balance:")
    print(f"Negative (Class 0): {n_neg}")
    print(f"Positive (Class 1): {n_pos}")
    pos_weight_val = n_neg / (n_pos)
    print(f"Computed positive class weight: {pos_weight_val:.4f}")
    CLASS_WEIGHT = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    # ---------------------------------------------------------------------------------------------

    model = LogisticRegression(input_size=INPUT_SIZE).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHT) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping_metric = float("inf")
    epochs_no_improve = 0
    best_model_weights = None

    # ---------------------------------------------------------------------------------------------

    print("Starting training...\n")
    # --- Training loop ---

    total_start_time = time.time()

    with open(log_file_path, 'w') as log_file:
        for epoch in range(EPOCHS):

            start_time = time.time()
            
            train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer)
            val_loss, val_auroc = validate(val_loader, model, loss_fn)

            epoch_time = time.time() - start_time
        
            log_msg = (f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | "
                    f"Time: {epoch_time:.2f}s\n")
            print(log_msg)
            log_file.write(log_msg)
        
            if val_loss < early_stopping_metric:
                early_stopping_metric = val_loss
                epochs_no_improve = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
        
            if epochs_no_improve == PATIENCE:
                log_file.write("Early stopping triggered")
                print("Early stopping triggered")
                break
                
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.2f} seconds " f"({total_time/60:.2f} minutes)\n")   
            
    # ---------------------------------------------------------------------------------------------

    # --- Test evaluation ---

    model.load_state_dict(best_model_weights)
    model.eval()

    # Helper function to get predictions
    def get_predictions(loader, model):
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = model(X)
                probs = torch.sigmoid(logits)
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())
        y_true = torch.cat(all_labels).numpy().flatten()
        y_probs = torch.cat(all_probs).numpy().flatten()
        return y_true, y_probs

    print("Evaluating on Test set...")
    y_test_true, y_test_probs = get_predictions(test_loader, model)

    test_auroc = roc_auc_score(y_test_true, y_test_probs)

    y_test_preds = (y_test_probs > CLASSIFICATION_THRESHOLD).astype(float)
    cm = confusion_matrix(y_test_true, y_test_preds)

    print(f"Test AUROC: {test_auroc:.4f}")

    with open(test_metrics_path, 'w') as f:
        f.write(f"Test AUROC: {test_auroc:.4f}\n")

    # Save Test CM
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # PLOTS (ROC)
    print("Generating Test Curves...")

    # --- Plot ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test_true, y_test_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved to {roc_curve_path}")

    return test_auroc