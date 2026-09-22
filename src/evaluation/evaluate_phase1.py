import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Add the 'src' directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dummy_dataset import DummyECGDataset
from data.masking import apply_block_masking
from data.ptbxl_dataset import PTBXLECGDataset
from data.chapman_dataset import ChapmanECGDataset
from data.mimic_iv_ecg_dataset import MIMICIVECGDataset
from models.foundation_encoder import FoundationEncoder

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Phase 1 Foundation Encoder on Test ECGs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/foundation_encoder_phase1.pth",
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--dataset", type=str, default="all3",
                        choices=["all3", "both", "ptbxl", "chapman", "mimic", "dummy"],
                        help="Test dataset: 'all3' (PTB-XL + Chapman + MIMIC-IV-ECG test splits), "
                             "'both' (PTB-XL + Chapman), 'ptbxl', 'chapman', 'mimic', or 'dummy'")
    parser.add_argument("--ptbxl_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset directory")
    parser.add_argument("--chapman_dir", type=str, default="data/Chapman", help="Path to Chapman dataset directory")
    parser.add_argument("--mimic_dir", type=str, default="data/MIMIC-IV-ECG", help="Path to MIMIC-IV-ECG dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test samples (for fast evaluation)")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Block masking ratio (default: 0.6 per Sec 4.2)")
    parser.add_argument("--patch_size", type=int, default=50, help="Patch token size (default: 50)")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer encoder depth (default: 8)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Transformer embedding dimension (default: 256)")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads (default: 8)")
    parser.add_argument("--save_plot", type=str, default="reports/reconstruction_sample.png",
                        help="File path to save sample waveform reconstruction plot")
    parser.add_argument("--output_json", type=str, default="reports/eval_metrics.json",
                        help="File path to save numerical evaluation metrics in JSON format")
    return parser.parse_args()

def evaluate_phase1(args=None):
    if args is None:
        args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f" Phase 1 Foundation Encoder: Masked Reconstruction Evaluation ")
    print(f"============================================================")
    print(f"Using device       : {device}")
    print(f"Model Checkpoint   : {args.checkpoint}")
    print(f"Mask Ratio         : {args.mask_ratio * 100:.0f}% contiguous block masking")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'. Train the model first via train_phase1.py.")

    # 1. Load Test Dataset (held-out splits never seen in training)
    datasets = []
    if args.dataset in ["all3", "both", "ptbxl"]:
        if os.path.exists(os.path.join(args.ptbxl_dir, "ptbxl_database.csv")):
            print(f"Loading PTB-XL held-out test split (fold 10 benchmark)...")
            ptbxl_test = PTBXLECGDataset(data_dir=args.ptbxl_dir, sampling_rate=500, split="test")
            print(f"  -> PTB-XL test records: {len(ptbxl_test)}")
            datasets.append(ptbxl_test)
        else:
            print(f"Warning: PTB-XL dataset not found at '{args.ptbxl_dir}'.")

    if args.dataset in ["all3", "both", "chapman"]:
        if os.path.exists(os.path.join(args.chapman_dir, "RECORDS")):
            print(f"Loading Chapman held-out test split (10% test fold)...")
            chapman_test = ChapmanECGDataset(data_dir=args.chapman_dir, split="test")
            print(f"  -> Chapman test records: {len(chapman_test)}")
            datasets.append(chapman_test)
        else:
            print(f"Warning: Chapman dataset not found at '{args.chapman_dir}'.")

    if args.dataset in ["all3", "mimic"]:
        mimic_csv = os.path.join(args.mimic_dir, "record_list.csv")
        if os.path.exists(mimic_csv):
            print(f"Loading MIMIC-IV-ECG held-out test split (10% patient-level fold)...")
            mimic_test = MIMICIVECGDataset(data_dir=args.mimic_dir, split="test")
            print(f"  -> MIMIC-IV-ECG test records: {len(mimic_test)}")
            datasets.append(mimic_test)
        else:
            print(f"Warning: MIMIC-IV-ECG dataset not found at '{args.mimic_dir}'.")
    if not datasets:
        print("Falling back to dummy test dataset...")
        test_dataset = DummyECGDataset(num_samples=args.max_samples or 100, num_channels=12, seq_length=5000)
    elif len(datasets) == 1:
        test_dataset = datasets[0]
    else:
        test_dataset = ConcatDataset(datasets)

    if args.max_samples is not None and len(test_dataset) > args.max_samples:
        test_dataset = Subset(test_dataset, list(range(args.max_samples)))
        print(f"Subsampled test dataset to {len(test_dataset)} records.")

    print(f"Total test records to evaluate: {len(test_dataset)}")
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Initialize Model and Load Trained Weights
    print("Loading FoundationEncoder architecture and weights...")
    model = FoundationEncoder(
        in_channels=12,
        patch_size=args.patch_size,
        seq_length=5000,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.0,  # disable dropout during testing
        learned_pos=True,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Evaluation Loop
    total_masked_sq_err = 0.0
    total_masked_elements = 0
    total_overall_sq_err = 0.0
    total_overall_elements = 0
    total_unmasked_sq_err = 0.0
    total_unmasked_elements = 0

    # Per-lead metrics
    per_lead_sq_err = np.zeros(12, dtype=np.float64)
    per_lead_masked_counts = np.zeros(12, dtype=np.int64)

    first_batch_sample = None

    print("\nRunning test evaluation...")
    with torch.no_grad():
        for batch_idx, original_signals in enumerate(dataloader):
            original_signals = original_signals.to(device) # (B, 12, 5000)
            B, C, L = original_signals.shape

            # Apply identical block masking as used in pre-training
            masked_signals, mask = apply_block_masking(
                original_signals,
                patch_size=args.patch_size,
                mask_ratio=args.mask_ratio
            ) # mask: (B, 1, 5000)

            # Reconstruct signal with trained model
            reconstructed_signals = model(masked_signals)

            # Store first sample for optional plotting
            if first_batch_sample is None:
                first_batch_sample = (
                    original_signals[0].cpu().numpy(),
                    masked_signals[0].cpu().numpy(),
                    reconstructed_signals[0].cpu().numpy(),
                    mask[0, 0].cpu().numpy()
                )

            # Errors
            diff = reconstructed_signals - original_signals
            sq_diff = diff ** 2 # (B, 12, 5000)

            # Masked patch MSE
            masked_sq_diff = sq_diff * mask
            masked_elem_count = mask.sum().item() * C
            total_masked_sq_err += masked_sq_diff.sum().item()
            total_masked_elements += masked_elem_count

            # Overall sequence MSE
            total_overall_sq_err += sq_diff.sum().item()
            total_overall_elements += sq_diff.numel()

            # Unmasked patch MSE (fidelity on visible patches)
            unmasked_mask = 1.0 - mask
            unmasked_sq_diff = sq_diff * unmasked_mask
            unmasked_elem_count = unmasked_mask.sum().item() * C
            total_unmasked_sq_err += unmasked_sq_diff.sum().item()
            total_unmasked_elements += unmasked_elem_count

            # Per-lead masked squared errors
            for lead_idx in range(12):
                lead_err = masked_sq_diff[:, lead_idx, :].sum().item()
                lead_count = mask.sum().item()
                per_lead_sq_err[lead_idx] += lead_err
                per_lead_masked_counts[lead_idx] += int(lead_count)

    # 4. Compute Final Metrics
    masked_mse = total_masked_sq_err / max(total_masked_elements, 1)
    overall_mse = total_overall_sq_err / max(total_overall_elements, 1)
    unmasked_mse = total_unmasked_sq_err / max(total_unmasked_elements, 1)
    masked_rmse = np.sqrt(masked_mse)

    lead_mses = {}
    for i, name in enumerate(LEAD_NAMES):
        count = max(per_lead_masked_counts[i], 1)
        lead_mses[name] = float(per_lead_sq_err[i] / count)

    # 5. Display Clean Formatted Results
    print("\n" + "=" * 60)
    print("                PHASE 1 TEST EVALUATION RESULTS             ")
    print("=" * 60)
    print(f" PRIMARY METRIC (Masked Patch MSE) : {masked_mse:.6f}")
    print(f" Root Mean Squared Error (RMSE)    : {masked_rmse:.6f}")
    print(f" Overall Full-Sequence MSE        : {overall_mse:.6f}")
    print(f" Visible (Unmasked) Patch MSE      : {unmasked_mse:.6f}")
    print("-" * 60)
    print(" Per-Lead Masked Reconstruction MSE Breakdown:")
    for name, lmse in lead_mses.items():
        print(f"   Lead {name:<4} : MSE = {lmse:.6f}")
    print("=" * 60)

    # 6. Save JSON Report
    metrics_summary = {
        "checkpoint": args.checkpoint,
        "test_dataset": args.dataset,
        "total_test_records": len(test_dataset),
        "mask_ratio": args.mask_ratio,
        "masked_mse": float(masked_mse),
        "masked_rmse": float(masked_rmse),
        "overall_mse": float(overall_mse),
        "unmasked_mse": float(unmasked_mse),
        "per_lead_mse": lead_mses,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"Metrics saved to: {args.output_json}")

    # 7. Optional Plot Generation (Sample Waveform Comparison)
    if args.save_plot and first_batch_sample is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            orig, masked, recon, m_mask = first_batch_sample
            plot_path = os.path.normpath(args.save_plot)
            plot_dir = os.path.dirname(plot_path)
            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            time_axis = np.arange(orig.shape[1]) / 500.0  # seconds

            # Plot Lead II (standard rhythm monitoring lead)
            lead_idx = 1 # Lead II
            lead_label = LEAD_NAMES[lead_idx]

            # Original
            axes[0].plot(time_axis, orig[lead_idx], color="#1f77b4", linewidth=1.2, label=f"Original (Lead {lead_label})")
            axes[0].set_title(f"Original 12-Lead ECG Signal (Lead {lead_label})", fontsize=12, fontweight="bold")
            axes[0].set_ylabel("Amplitude (normalized)")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right")

            # Masked input
            axes[1].plot(time_axis, masked[lead_idx], color="#d62728", linewidth=1.2, label=f"Masked Input (60% block mask)")
            # Highlight masked areas in red shade
            axes[1].fill_between(time_axis, -3, 3, where=(m_mask > 0.5), color="#ff9896", alpha=0.3, label="Masked span")
            axes[1].set_title(f"Masked Input Signal (60% Block Masking)", fontsize=12, fontweight="bold")
            axes[1].set_ylabel("Amplitude (normalized)")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right")

            # Reconstructed vs Original
            axes[2].plot(time_axis, orig[lead_idx], color="#1f77b4", alpha=0.5, linestyle="--", label="Ground Truth")
            axes[2].plot(time_axis, recon[lead_idx], color="#2ca02c", linewidth=1.4, label="Model Reconstruction")
            axes[2].set_title(f"Reconstructed Waveform (MSE: {masked_mse:.4f})", fontsize=12, fontweight="bold")
            axes[2].set_xlabel("Time (seconds)")
            axes[2].set_ylabel("Amplitude (normalized)")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc="upper right")

            plt.tight_layout()
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            plt.close('all')
            print(f"Sample waveform visualization saved to: {plot_path}")
        except Exception as e:
            print(f"Could not generate plot (skipped): {e}")

    return metrics_summary

if __name__ == "__main__":
    evaluate_phase1()
