import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
import copy
import warnings

import wandb

from data import GleasonTileDataset, PatientTileSampler
from model import L0LogisticRegression

warnings.filterwarnings('ignore')


def _train_single_fold(
    fold_idx,
    train_indices,
    val_indices,
    test_indices,
    y_train_val,
    train_sub_idx,
    seed,
    k,
    keep_indices,
    hdf5_path,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    OUTPUT_SIZE,
    INPUT_SIZE,
    L0_LAMBDA,
    WEIGHT_DECAY,
    droprate_init,
    temperature,
    BETA_EMA,
    local_rep,
    use_wandb,
    extract_features,
    device,
    wandb_run_name=None,
    wandb_config=None,
):
    """Train and evaluate one inner CV fold. Returns (test_auroc, pruned_features)."""

    # ------------------------------------------------------------------
    # W&B — one run per fold
    # ------------------------------------------------------------------
    if use_wandb:
        fold_config = dict(wandb_config or {})
        fold_config['fold'] = fold_idx
        wandb.init(
            project='L0_gating',
            name=f'{wandb_run_name}_fold{fold_idx}',
            config=fold_config,
        )

    # ------------------------------------------------------------------
    # Datasets & Scaler
    # ------------------------------------------------------------------
    if k == 25587:
        train_dataset_raw = GleasonTileDataset(hdf5_path, train_indices)
    else:
        train_dataset_raw = GleasonTileDataset(hdf5_path, train_indices, feature_indices=keep_indices)

    raw_loader = DataLoader(train_dataset_raw, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    scaler = StandardScaler()
    print(f"  [Fold {fold_idx}] Fitting scaler...")
    for X_batch, _ in raw_loader:
        scaler.partial_fit(X_batch.numpy())
    print(f"  [Fold {fold_idx}] Scaler fitted.")

    train_sampler = PatientTileSampler(dataset_indices=train_indices, tiles_per_patient=100)

    if k == 25587:
        train_dataset = GleasonTileDataset(hdf5_path, train_indices, scaler=scaler)
        val_dataset   = GleasonTileDataset(hdf5_path, val_indices,   scaler=scaler)
        test_dataset  = GleasonTileDataset(hdf5_path, test_indices,  scaler=scaler)
    else:
        train_dataset = GleasonTileDataset(hdf5_path, train_indices, scaler=scaler, feature_indices=keep_indices)
        val_dataset   = GleasonTileDataset(hdf5_path, val_indices,   scaler=scaler, feature_indices=keep_indices)
        test_dataset  = GleasonTileDataset(hdf5_path, test_indices,  scaler=scaler, feature_indices=keep_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler,
                              pin_memory=True, num_workers=8, prefetch_factor=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=8, prefetch_factor=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=8, prefetch_factor=4)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    N_train = len(train_indices)
    scaled_weight_decay = WEIGHT_DECAY * N_train

    train_labels = y_train_val[train_sub_idx]
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    pos_weight_val = n_neg / (n_pos + 1e-6)
    CLASS_WEIGHT = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    model = L0LogisticRegression(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        beta_ema=BETA_EMA,
        L0_LAMBDA=L0_LAMBDA,
        scaled_weight_decay=scaled_weight_decay,
        droprate_init=droprate_init,
        temperature=temperature,
        local_rep=local_rep,
    ).to(device)

    if model.beta_ema > 0.:
        model.avg_param = [a.to(device) for a in model.avg_param]

    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------------
    # Inner helpers
    # ------------------------------------------------------------------
    total_steps = 0
    exp_flops, exp_l0 = [], []

    def train_one_epoch(dataloader, epoch):
        nonlocal total_steps, exp_flops, exp_l0
        model.train()
        running_loss = 0.0
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y = y.float().unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y) - (model.regularization() / N_train)
            loss.backward()
            optimizer.step()
            model.output_layer.constrain_parameters()
            if model.beta_ema > 0.:
                model.update_ema()
            running_loss += loss.item() * X.size(0)
            total_steps += 1

            e_fl, e_l0 = model.output_layer.count_expected_flops_and_l0()
            exp_flops.append(e_fl)
            exp_l0.append(e_l0)
            if use_wandb:
                wandb.log({
                    'stats_comp/exp_flops': e_fl,
                    'stats_comp/exp_l0':    e_l0,
                }, step=total_steps)

        train_loss = running_loss / len(dataloader.dataset)
        if use_wandb:
            wandb.log({'train/loss': train_loss, 'epoch': epoch}, step=total_steps)
        return train_loss

    def validate(dataloader, epoch):
        if model.beta_ema > 0.:
            old_params = model.get_params()
            model.load_ema_params()

        model.eval()
        running_loss = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                y = y.float().unsqueeze(1)
                logits = model(X)
                running_loss += loss_fn(logits, y).item() * X.size(0)
                probs = torch.sigmoid(logits)
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())

        if model.beta_ema > 0.:
            model.load_params(old_params)

        avg_loss = running_loss / len(dataloader.dataset)
        y_true  = torch.cat(all_labels).numpy()
        y_probs = torch.cat(all_probs).numpy()

        try:
            val_auroc = roc_auc_score(y_true, y_probs)
        except ValueError:
            val_auroc = 0.0

        gate_stats = {}
        with torch.no_grad():
            gates = model.output_layer.sample_z(batch_size=1, sample=False).squeeze().cpu().numpy()
        num_zero   = int(np.sum(gates == 0.0))
        num_active = int(np.sum(gates > 0.0))
        g_min, g_max, g_mean = float(gates.min()), float(gates.max()), float(gates.mean())
        gate_stats['layer0'] = {'active': num_active, 'pruned': num_zero,
                                'min': g_min, 'max': g_max, 'mean': g_mean}
        gate_stats['total_active'] = num_active
        gate_stats['total_pruned'] = num_zero

        if use_wandb:
            log_dict = {
                'val/loss':  avg_loss,
                'val/auroc': val_auroc,
                'epoch':     epoch,
            }
            mode_z = model.output_layer.sample_z(1, sample=False).view(-1)
            log_dict['mode_z/layer0']      = wandb.Histogram(mode_z.cpu().data.numpy())
            log_dict['gates/layer0_min']   = g_min
            log_dict['gates/layer0_max']   = g_max
            log_dict['gates/layer0_mean']  = g_mean
            wandb.log(log_dict, step=total_steps)

        return avg_loss, val_auroc, gate_stats

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    early_stopping_metric = float("inf")
    epochs_no_improve   = 0
    best_model_weights  = None
    best_gate_stats     = {}

    print(f"  [Fold {fold_idx}] Starting training...")
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(train_loader, epoch)
        val_loss, val_auroc, curr_gate_stats = validate(val_loader, epoch)
        epoch_time = time.time() - start_time

        print(f"  [Fold {fold_idx}] Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | "
              f"Pruned: {curr_gate_stats['total_pruned']}/{curr_gate_stats['total_active'] + curr_gate_stats['total_pruned']} | "              
              f"Gate Min: {curr_gate_stats['layer0']['min']:.3f} | Gate Max: {curr_gate_stats['layer0']['max']:.3f} | Gate Mean: {curr_gate_stats['layer0']['mean']:.3f} | "
        )

        if val_loss < early_stopping_metric:
            early_stopping_metric = val_loss
            epochs_no_improve = 0
            best_gate_stats = curr_gate_stats
            if model.beta_ema > 0.:
                old_params = model.get_params()
                model.load_ema_params()
                best_model_weights = copy.deepcopy(model.state_dict())
                model.load_params(old_params)
            else:
                best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve == PATIENCE:
            print(f"  [Fold {fold_idx}] Early stopping triggered")
            break

    total_time = time.time() - total_start_time
    print(f"  [Fold {fold_idx}] Total training time: {total_time:.2f}s")
    print(f"  [Fold {fold_idx}] Best val loss: {early_stopping_metric:.4f}")
    if best_gate_stats:
        print(f"  [Fold {fold_idx}] Gate stats at best val loss — "
              f"active={best_gate_stats['total_active']}, pruned={best_gate_stats['total_pruned']}")

    # ------------------------------------------------------------------
    # Test evaluation with best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict(best_model_weights)
    model.eval()

    # ------------------------------------------------------------------
    # Optionally save selected feature indices
    # ------------------------------------------------------------------
    if extract_features:
        with torch.no_grad():
            gates = model.output_layer.sample_z(batch_size=1, sample=False).squeeze().cpu().numpy()
        selected_local_indices = np.where(gates > 0.0)[0]
        if k != 25587:
            keep_indices_arr = np.array(keep_indices)
            selected_gene_indices = keep_indices_arr[selected_local_indices]
        else:
            selected_gene_indices = selected_local_indices
        save_path = (f'/project_antwerp/baseline/ensemble_preprocessing/logs/'
                     f'selected_indices_seed_{seed}_fold_{fold_idx}.npy')
        np.save(save_path, selected_gene_indices)
        print(f"  [Fold {fold_idx}] Saved selected indices to {save_path}")

    def get_predictions(loader):
        all_labels, all_probs = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(X)
                probs  = torch.sigmoid(logits)
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())
        y_true  = torch.cat(all_labels).numpy().flatten()
        y_probs = torch.cat(all_probs).numpy().flatten()
        return y_true, y_probs

    y_test_true, y_test_probs = get_predictions(test_loader)
    test_auroc = roc_auc_score(y_test_true, y_test_probs)
    pruned_features = best_gate_stats.get('total_pruned', 0)

    print(f"  [Fold {fold_idx}] Test AUROC: {test_auroc:.4f} | Pruned features: {pruned_features}")

    if use_wandb:
        wandb.log({
            'test/auroc':            test_auroc,
            'test/pruned_features':  pruned_features,
            'test/selected_features': INPUT_SIZE - pruned_features,
        })
        wandb.finish()

    return test_auroc, pruned_features


def run_baseline_training(
    seed,
    L0_LAMBDA=1.0,
    WEIGHT_DECAY=5e-4,
    droprate_init=0.5,
    temperature=2./3.,
    learning_rate=1e-2,
    k=1000,
    local_rep=False,
    use_wandb=True,
    extract_features=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    keep_indices = None
    if k != 25587:
        welford_gene_indices = np.load(
            f'/project_antwerp/baseline/ensemble_preprocessing/index_files/top_{k}_genes_indices_welford_seed_{seed}.npy',
            allow_pickle=True,
        )
        keep_indices = welford_gene_indices.tolist()

    # Hyperparameters
    BATCH_SIZE  = 128
    EPOCHS      = 200
    PATIENCE    = 10
    OUTPUT_SIZE = 1
    INPUT_SIZE  = k

    hdf5_path  = '/project_antwerp/TCGA-PRAD/data/outputs/TCGA/PRAD/merged_tile_preds.hdf5'
    index_path = '/project_antwerp/baseline/ensemble_preprocessing/index_files/filtered_binarylabel_indices.npy'
    index_file = np.load(index_path, allow_pickle=True)

    run_name = f'Ensemble_preprocessing_seed{seed}'
    wandb_config = {
        'seed':          seed,
        'k':             k,
        'L0_LAMBDA':     L0_LAMBDA,
        'WEIGHT_DECAY':  WEIGHT_DECAY,
        'droprate_init': droprate_init,
        'temperature':   temperature,
        'learning_rate': learning_rate,
        'local_rep':     local_rep,
        'BETA_EMA':      0.999,
        'BATCH_SIZE':    BATCH_SIZE,
        'EPOCHS':        EPOCHS,
        'PATIENCE':      PATIENCE,
    }

    # ------------------------------------------------------------------
    # Outer split — held-out test set (unchanged)
    # ------------------------------------------------------------------
    groups = index_file[:, 0]
    labels = index_file[:, 2].astype(int)

    sgkf_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(index_file, labels, groups))

    X_train_val      = index_file[train_val_idx]
    y_train_val      = labels[train_val_idx]
    groups_train_val = groups[train_val_idx]
    test_indices     = index_file[test_idx]

    # ------------------------------------------------------------------
    # Inner CV — iterate all folds (was: only next())
    # ------------------------------------------------------------------
    sgkf_inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)

    fold_results = []  # list of (fold_idx, test_auroc, pruned_features)

    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(
        sgkf_inner.split(X_train_val, y_train_val, groups_train_val)
    ):
        train_indices = X_train_val[train_sub_idx]
        val_indices   = X_train_val[val_sub_idx]

        print(f"\n{'='*60}")
        print(f"Seed {seed} | Fold {fold_idx + 1}/6")
        print(f"  Train: {len(train_indices)} tiles | {len(np.unique(train_indices[:, 0]))} patients")
        print(f"  Val:   {len(val_indices)} tiles   | {len(np.unique(val_indices[:, 0]))} patients")
        print(f"  Test:  {len(test_indices)} tiles  | {len(np.unique(test_indices[:, 0]))} patients")

        # Patient-leakage check
        train_pids = set(train_indices[:, 0])
        val_pids   = set(val_indices[:, 0])
        test_pids  = set(test_indices[:, 0])
        assert train_pids.isdisjoint(val_pids),   "Overlap between Train and Val patients!"
        assert train_pids.isdisjoint(test_pids),  "Overlap between Train and Test patients!"
        assert val_pids.isdisjoint(test_pids),    "Overlap between Val and Test patients!"

        test_auroc, pruned_features = _train_single_fold(
            fold_idx=fold_idx + 1,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            y_train_val=y_train_val,
            train_sub_idx=train_sub_idx,
            seed=seed,
            k=k,
            keep_indices=keep_indices,
            hdf5_path=hdf5_path,
            BATCH_SIZE=BATCH_SIZE,
            LEARNING_RATE=learning_rate,
            EPOCHS=EPOCHS,
            PATIENCE=PATIENCE,
            OUTPUT_SIZE=OUTPUT_SIZE,
            INPUT_SIZE=INPUT_SIZE,
            L0_LAMBDA=L0_LAMBDA,
            WEIGHT_DECAY=WEIGHT_DECAY,
            droprate_init=droprate_init,
            temperature=temperature,
            BETA_EMA=0.999,
            local_rep=local_rep,
            use_wandb=use_wandb,
            extract_features=extract_features,
            device=device,
            wandb_run_name=run_name,
            wandb_config=wandb_config,
        )

        fold_results.append((fold_idx + 1, test_auroc, pruned_features))

    # ------------------------------------------------------------------
    # Aggregate results across folds
    # ------------------------------------------------------------------
    aurocs   = [r[1] for r in fold_results]
    pruned   = [r[2] for r in fold_results]
    avg_auroc   = float(np.mean(aurocs))
    avg_pruned  = float(np.mean(pruned))


    return {
        'fold_results':       fold_results,   # [(fold_idx, test_auroc, pruned_features), ...]
        'avg_test_auroc':     avg_auroc,
        'avg_pruned_features': avg_pruned,
    }