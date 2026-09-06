import os
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = 'parsed_graphs'                  # Directory containing .pt graph files
TARGET_GRAPH_FILE = 'CO2_828_graph.pt'      # Specific graph file for training, or None for all
INFERENCE_TARGET = 'CO2_728_graph.pt'       # Unaltered target graph for post-training inference
CORRUPTION_RATE = 0.08                      # Fraction of transitions to perturb (e.g., 8%)
ERROR_WEIGHTS = {                           # Relative probability of simulated human error types
    'qn_shift': 0.40,                       # Small quantum number offset (+/-1, +/-2 in J/v)
    'qn_swap': 0.30,                        # Swap upper and lower state assignments
    'random_relink': 0.30                   # Re-link transition to an arbitrary state
}
TRAIN_SPLIT_RATIO = 0.80                    # 80% train edges, 20% validation edges
NUM_LAYERS = 3                              # Number of SpectroConv message passing layers
HIDDEN_DIM = 64                             # Latent node embedding dimension
LEARNING_RATE = 0.001                       # Optimizer learning rate
WEIGHT_DECAY = 1e-4                         # L2 regularization
EPOCHS = 300                                # Training epochs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==============================================================================
# Perturbation & Simulated Human Error Engine
# ==============================================================================
def build_qn_neighbor_map(state_qns, num_nodes):
    """
    Builds a dictionary mapping state index -> list of candidate neighboring
    state indices that differ by small integer offsets (+/-1, +/-2 in J or v).
    """
    if not state_qns or len(state_qns) == 0:
        return {}

    # Extract numeric representation where possible
    numeric_qns = []
    for qn in state_qns:
        num_vec = []
        for val in qn:
            try:
                num_vec.append(float(val))
            except ValueError:
                num_vec.append(hash(str(val)) % 1000)
        numeric_qns.append(np.array(num_vec))
    numeric_qns = np.array(numeric_qns)

    neighbor_map = {}
    # Find neighbors with Manhattan or L1 distance <= 2 in quantum number space
    for i in range(num_nodes):
        # Pick a local search window to keep mapping O(N)
        start = max(0, i - 100)
        end = min(num_nodes, i + 100)
        window = numeric_qns[start:end]
        diffs = np.sum(np.abs(window - numeric_qns[i]), axis=1)
        valid_local = np.where((diffs > 0) & (diffs <= 2.0))[0]
        
        if len(valid_local) > 0:
            neighbor_map[i] = (start + valid_local).tolist()
        else:
            # Fallback to nearest indices
            neighbor_map[i] = [max(0, i - 1), min(num_nodes - 1, i + 1)]

    return neighbor_map


def perturb_transitions(raw_data, corruption_rate=CORRUPTION_RATE, error_weights=ERROR_WEIGHTS, neighbor_map=None):
    """
    Applies synthetic human misassignment perturbations to a fraction of forward transitions.
    Returns a corrupted PyG Data object with ground-truth binary edge labels (0 = clean, 1 = corrupted).
    """
    num_nodes = raw_data.num_nodes if hasattr(raw_data, 'num_nodes') and raw_data.num_nodes is not None else raw_data.x.shape[0]
    total_directed_edges = raw_data.edge_index.shape[1]
    num_transitions = total_directed_edges // 2

    # Forward edge endpoints
    src_nodes = raw_data.edge_index[0, :num_transitions].clone().cpu().numpy()
    dst_nodes = raw_data.edge_index[1, :num_transitions].clone().cpu().numpy()
    fwd_attr = raw_data.edge_attr[:num_transitions].clone().cpu().numpy()

    num_corrupt = max(1, int(num_transitions * corruption_rate))
    corrupt_indices = np.random.choice(num_transitions, size=num_corrupt, replace=False)
    corrupt_mask = np.zeros(num_transitions, dtype=bool)
    corrupt_mask[corrupt_indices] = True

    # Error type probabilities
    types = list(error_weights.keys())
    probs = np.array([error_weights[t] for t in types])
    probs = probs / probs.sum()

    for idx in corrupt_indices:
        err_type = np.random.choice(types, p=probs)
        u, v = src_nodes[idx], dst_nodes[idx]

        if err_type == 'qn_shift':
            # Shift destination or source to a neighboring QN state
            target_node = v if np.random.rand() > 0.5 else u
            if neighbor_map and target_node in neighbor_map and len(neighbor_map[target_node]) > 0:
                new_node = np.random.choice(neighbor_map[target_node])
            else:
                offset = np.random.choice([-2, -1, 1, 2])
                new_node = int(np.clip(target_node + offset, 0, num_nodes - 1))
            
            if target_node == v:
                dst_nodes[idx] = new_node
            else:
                src_nodes[idx] = new_node

        elif err_type == 'qn_swap':
            # Invert transition endpoints (upper <-> lower swap)
            src_nodes[idx] = v
            dst_nodes[idx] = u

        elif err_type == 'random_relink':
            # Re-route destination to a completely random state
            rand_node = np.random.randint(0, num_nodes)
            while rand_node == u:
                rand_node = np.random.randint(0, num_nodes)
            dst_nodes[idx] = rand_node

    # Reconstruct bidirectional edge tensors
    new_src = np.concatenate([src_nodes, dst_nodes])
    new_dst = np.concatenate([dst_nodes, src_nodes])
    edge_index = torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long)

    # Edge attributes: forward [v, ev], reverse [-v, ev]
    rev_attr = fwd_attr.copy()
    rev_attr[:, 0] = -rev_attr[:, 0]
    edge_attr = torch.tensor(np.concatenate([fwd_attr, rev_attr], axis=0), dtype=torch.float)

    # Ground truth labels for all edges (0 = clean, 1 = corrupted)
    fwd_labels = corrupt_mask.astype(np.float32)
    edge_labels = torch.tensor(np.concatenate([fwd_labels, fwd_labels]), dtype=torch.float)

    corrupted_data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_labels,
        num_nodes=num_nodes,
        num_transitions=num_transitions,
        num_corrupt=num_corrupt,
        molecule_name=getattr(raw_data, 'molecule_name', 'Unknown')
    )

    return corrupted_data


# ==============================================================================
# Model Architecture
# ==============================================================================
class SpectroConv(MessagePassing):
    """
    Message passing layer that aggregates node states weighted by transition wavenumber.
    """
    def __init__(self, hidden_dim):
        super(SpectroConv, self).__init__(aggr='mean')

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.energy_proj = nn.Linear(1, hidden_dim)

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        trans_E = edge_attr[:, 0:1]
        state_interaction = self.msg_mlp(torch.cat([x_i, x_j], dim=-1))
        energy_shift = self.energy_proj(trans_E)
        return state_interaction + energy_shift

    def update(self, aggr_out, x):
        new_state = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(new_state) + x


class TransitionAnomalyGNN(nn.Module):
    """
    Graph Neural Network for bad/corrupted transition detection on spectroscopic networks.
    Permutation-invariant and fully inductive (does not rely on node IDs or fixed embedding tables).
    Combines multi-hop SpectroConv message passing with an energy-differential edge anomaly head.
    """
    def __init__(self, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM):
        super().__init__()

        # Universal learnable initial node state (broadcasted across all nodes)
        self.init_node_feature = nn.Parameter(torch.zeros(1, hidden_dim))

        # Message passing layers
        self.convs = nn.ModuleList([
            SpectroConv(hidden_dim) for _ in range(num_layers)
        ])

        # Inductive Energy-Differential Edge Classifier Head
        # Evaluates purely relative features: [(h_v - h_u), |h_v - h_u|, edge_attr]
        in_dim = (2 * hidden_dim) + 2
        self.edge_classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, edge_index, edge_attr, num_nodes=None):
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1

        # Broadcast initial latent state across all nodes
        h = self.init_node_feature.expand(num_nodes, -1)

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)

        # Purely relative edge feature extraction (independent of state IDs)
        u_nodes = edge_index[0]
        v_nodes = edge_index[1]

        h_u = h[u_nodes]
        h_v = h[v_nodes]
        h_diff = h_v - h_u
        h_diff_abs = torch.abs(h_diff)

        edge_features = torch.cat([h_diff, h_diff_abs, edge_attr], dim=-1)
        edge_logits = self.edge_classifier(edge_features).squeeze(-1)

        return edge_logits


# ==============================================================================
# Training & Evaluation Loop
# ==============================================================================
def train_and_evaluate(raw_data, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    print(f"\n{'='*75}")
    print(f"Training Bad Transition Detector on {getattr(raw_data, 'molecule_name', 'Graph')}")
    print(f"{'='*75}")

    num_nodes = raw_data.num_nodes if hasattr(raw_data, 'num_nodes') and raw_data.num_nodes is not None else raw_data.x.shape[0]
    total_directed_edges = raw_data.edge_index.shape[1]
    num_transitions = total_directed_edges // 2

    # Pre-build QN neighbor map for realistic QN-shift perturbations
    neighbor_map = build_qn_neighbor_map(getattr(raw_data, 'state_qns', None), num_nodes)

    # Edge-level train / validation partition on transitions
    indices = np.random.permutation(num_transitions)
    split_idx = int(num_transitions * TRAIN_SPLIT_RATIO)
    train_trans_idx = indices[:split_idx]
    val_trans_idx = indices[split_idx:]

    model = TransitionAnomalyGNN(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Class imbalance weighting
    pos_weight_val = (1.0 - CORRUPTION_RATE) / CORRUPTION_RATE
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    print(f"Nodes: {num_nodes:,} | Total Transitions: {num_transitions:,} | Corruption Rate: {CORRUPTION_RATE*100:.1f}%")
    print(f"Train Transitions: {len(train_trans_idx):,} | Val Transitions: {len(val_trans_idx):,}")
    print(f"Class Imbalance Pos Weight: {pos_weight_val:.2f} | Device: {device}\n")

    train_loss_history = []
    val_loss_history = []
    val_roc_history = []
    val_pr_history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Generate on-the-fly random perturbations per epoch
        graph_data = perturb_transitions(raw_data, corruption_rate=CORRUPTION_RATE, neighbor_map=neighbor_map).to(device)

        # Forward transition indices mask
        train_mask = torch.tensor(train_trans_idx, dtype=torch.long, device=device)
        val_mask = torch.tensor(val_trans_idx, dtype=torch.long, device=device)

        # -------------------
        # Training Step
        # -------------------
        model.train()
        optimizer.zero_grad()

        edge_logits = model(graph_data.edge_index, graph_data.edge_attr, num_nodes=num_nodes)
        
        # Train loss computed on forward train transitions
        train_logits = edge_logits[train_mask]
        train_targets = graph_data.edge_label[train_mask]
        train_loss = loss_fn(train_logits, train_targets)

        train_loss.backward()
        optimizer.step()

        # -------------------
        # Validation Step
        # -------------------
        model.eval()
        with torch.no_grad():
            val_logits = edge_logits[val_mask]
            val_targets = graph_data.edge_label[val_mask]
            val_loss = loss_fn(val_logits, val_targets)

            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_true = val_targets.cpu().numpy()

            val_roc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else 0.5
            val_pr = average_precision_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else 0.0

        train_loss_history.append(train_loss.item())
        val_loss_history.append(val_loss.item())
        val_roc_history.append(val_roc)
        val_pr_history.append(val_pr)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val ROC-AUC: {val_roc:.4f} | Val PR-AUC: {val_pr:.4f}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s.")

    # -------------------
    # Final Comprehensive Evaluation
    # -------------------
    model.eval()
    with torch.no_grad():
        final_data = perturb_transitions(raw_data, corruption_rate=CORRUPTION_RATE, neighbor_map=neighbor_map).to(device)
        val_mask = torch.tensor(val_trans_idx, dtype=torch.long, device=device)
        logits = model(final_data.edge_index, final_data.edge_attr, num_nodes=num_nodes)[val_mask]
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = final_data.edge_label[val_mask].cpu().numpy()

    final_roc = roc_auc_score(targets, probs)
    final_pr_auc = average_precision_score(targets, probs)

    # Find best F1 threshold
    precision_curve, recall_curve, thresholds = precision_recall_curve(targets, probs)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_thresh_idx]

    preds_binary = (probs >= best_threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds_binary, average='binary', zero_division=0)
    cm = confusion_matrix(targets, preds_binary)

    print(f"\n{'='*75}")
    print("FINAL VALIDATION PERFORMANCE METRICS")
    print(f"{'='*75}")
    print(f" - ROC-AUC Score:             {final_roc:.4f}")
    print(f" - PR-AUC (Average Precision): {final_pr_auc:.4f}")
    print(f" - Optimal Decision Threshold: {best_threshold:.4f}")
    print(f" - Precision at Threshold:     {prec:.4f}")
    print(f" - Recall at Threshold:        {rec:.4f}")
    print(f" - F1-Score:                   {f1:.4f}")
    print(f" - Confusion Matrix (TN, FP / FN, TP):\n{cm}")

    # -------------------
    # Visualization Plots
    # -------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss & Metric Curves
    axs[0].plot(train_loss_history, label='Train Loss', color='royalblue')
    axs[0].plot(val_loss_history, label='Val Loss', color='crimson')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Weighted BCE Loss')
    axs[0].set_title('Training & Validation Loss')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 2. ROC & PR Curves
    fpr, tpr, _ = roc_curve(targets, probs)
    axs[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {final_roc:.3f})')
    axs[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('Receiver Operating Characteristic')
    axs[1].legend(loc='lower right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Anomaly Score Distributions
    clean_scores = probs[targets == 0]
    bad_scores = probs[targets == 1]
    axs[2].hist(clean_scores, bins=30, alpha=0.6, color='seagreen', label=f'Clean Transitions (N={len(clean_scores)})', density=True)
    axs[2].hist(bad_scores, bins=30, alpha=0.6, color='firebrick', label=f'Corrupted Transitions (N={len(bad_scores)})', density=True)
    axs[2].axvline(best_threshold, color='black', linestyle=':', lw=2, label=f'Threshold ({best_threshold:.2f})')
    axs[2].set_xlabel('Predicted Corruption Probability')
    axs[2].set_ylabel('Density')
    axs[2].set_title('Anomaly Score Separation')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_filename = f"bad_transition_detection_{getattr(raw_data, 'molecule_name', 'graph')}.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"\nEvaluation plots saved to: {plot_filename}", flush=True)
    plt.close(fig)

    return model, {
        'roc_auc': final_roc,
        'pr_auc': final_pr_auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'best_threshold': best_threshold,
        'confusion_matrix': cm
    }


# ==============================================================================
# Inference on Unaltered Target Graph
# ==============================================================================
def run_inference_on_unaltered(model, target_file, threshold=0.5, device=DEVICE):
    if not os.path.exists(target_file):
        print(f"Error: Inference file not found: {target_file}")
        return

    print(f"\n{'='*75}")
    print(f"RUNNING INFERENCE ON UNALTERED GRAPH: {os.path.basename(target_file)}")
    print(f"{'='*75}")

    infer_data = torch.load(target_file, weights_only=False)
    infer_nodes = infer_data.num_nodes if hasattr(infer_data, 'num_nodes') and infer_data.num_nodes is not None else infer_data.x.shape[0]
    infer_transitions = infer_data.edge_index.shape[1] // 2

    print(f"Loaded {getattr(infer_data, 'molecule_name', 'Graph')}: {infer_nodes:,} nodes, {infer_transitions:,} transitions (Unaltered)")

    model.eval()
    with torch.no_grad():
        infer_logits = model(infer_data.edge_index.to(device), infer_data.edge_attr.to(device), num_nodes=infer_nodes)
        fwd_logits = infer_logits[:infer_transitions]
        fwd_probs = torch.sigmoid(fwd_logits).cpu().numpy()

    print(f"\nInference Score Summary across {infer_transitions:,} Transitions:")
    print(f" - Mean Anomaly Probability:   {fwd_probs.mean():.4f}")
    print(f" - Median Anomaly Probability: {np.median(fwd_probs):.4f}")
    print(f" - Std Deviation:              {fwd_probs.std():.4f}")
    print(f" - Min / Max Probability:      {fwd_probs.min():.4f} / {fwd_probs.max():.4f}")

    flagged_indices = np.where(fwd_probs >= threshold)[0]
    flagged_count = len(flagged_indices)
    flagged_pct = (flagged_count / infer_transitions) * 100

    print(f"\nThreshold-Based Anomaly Flagging (Threshold = {threshold:.4f}):")
    print(f" - Flagged Potentially Corrupted Transitions: {flagged_count:,} / {infer_transitions:,} ({flagged_pct:.2f}%)")
    print(f" - Consistent / Clean Transitions:            {infer_transitions - flagged_count:,} / {infer_transitions:,} ({100 - flagged_pct:.2f}%)")

    top_k = min(15, infer_transitions)
    top_indices = np.argsort(-fwd_probs)[:top_k]

    src = infer_data.edge_index[0, :infer_transitions].cpu().numpy()
    dst = infer_data.edge_index[1, :infer_transitions].cpu().numpy()
    wavenumbers = infer_data.edge_attr[:infer_transitions, 0].cpu().numpy()
    uncertainties = infer_data.edge_attr[:infer_transitions, 1].cpu().numpy()

    print(f"\nTop {top_k} Most Suspicious Transitions in {os.path.basename(target_file)}:")
    print(f"{'Rank':<5} {'Trans Idx':<10} {'State u':<9} {'State v':<9} {'Wavenumber (cm^-1)':<20} {'Uncertainty':<14} {'Anomaly Prob':<12}")
    print("-" * 85)
    for rank, idx in enumerate(top_indices, 1):
        u_idx, v_idx = src[idx], dst[idx]
        wn = wavenumbers[idx]
        unc = uncertainties[idx]
        prob = fwd_probs[idx]
        print(f"{rank:<5} {idx:<10} {u_idx:<9} {v_idx:<9} {wn:<20.6f} {unc:<14.2e} {prob:<12.4f}")


# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: DATA_DIR '{DATA_DIR}' does not exist.")
        return

    if TARGET_GRAPH_FILE:
        graph_files = [os.path.join(DATA_DIR, TARGET_GRAPH_FILE)]
    else:
        graph_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.pt')))

    if not graph_files:
        print(f"No .pt files found in '{DATA_DIR}'.")
        return

    trained_model = None
    best_thresh = 0.5

    for gf in graph_files:
        if not os.path.exists(gf):
            print(f"File not found: {gf}")
            continue

        raw_data = torch.load(gf, weights_only=False)
        trained_model, metrics = train_and_evaluate(raw_data, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
        best_thresh = metrics.get('best_threshold', 0.5)

    if trained_model is not None and INFERENCE_TARGET:
        infer_path = os.path.join(DATA_DIR, INFERENCE_TARGET)
        run_inference_on_unaltered(trained_model, infer_path, threshold=best_thresh, device=DEVICE)


if __name__ == '__main__':
    main()
