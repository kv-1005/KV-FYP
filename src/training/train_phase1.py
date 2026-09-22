import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Add the 'src' directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dummy_dataset import DummyECGDataset
from data.masking import apply_block_masking
from data.ptbxl_dataset import PTBXLECGDataset
from data.chapman_dataset import ChapmanECGDataset
from data.mimic_iv_ecg_dataset import MIMICIVECGDataset
from models.foundation_encoder import FoundationEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Self-Supervised Foundation Pre-Training (Project A)")
    parser.add_argument("--dataset", type=str, default="all3",
                        choices=["all3", "both", "ptbxl", "chapman", "mimic", "dummy"],
                        help="Dataset to use: 'all3' (PTB-XL + Chapman + MIMIC-IV-ECG), "
                             "'both' (PTB-XL + Chapman), 'ptbxl', 'chapman', 'mimic', or 'dummy'")
    parser.add_argument("--ptbxl_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset directory")
    parser.add_argument("--chapman_dir", type=str, default="data/Chapman", help="Path to Chapman dataset directory")
    parser.add_argument("--mimic_dir", type=str, default="data/MIMIC-IV-ECG", help="Path to MIMIC-IV-ECG dataset directory")
    parser.add_argument("--use_dummy", action="store_true", help="Use synthetic dummy dataset instead of real data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate (default: 2e-4 per Sec 4.2)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay (default: 0.05 per Sec 4.2)")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer encoder depth (default: 8 per Sec 4.2)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Hidden dimension (default: 256 per Sec 4.2)")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads (default: 8 per Sec 4.2)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability (default: 0.1 per Sec 4.2)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of dataset samples (for fast debug/CPU testing)")
    parser.add_argument("--patch_size", type=int, default=50, help="Patch size for transformer embedding (50 = 100ms)")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Block masking ratio (default: 0.6 / 60% per Sec 4.2)")
    parser.add_argument("--save_path", type=str, default="checkpoints/foundation_encoder_phase1.pth", help="Path to save model checkpoint")
    parser.add_argument("--best_save_path", type=str, default="checkpoints/foundation_encoder_best.pth", help="Path to save best checkpoint")
    parser.add_argument("--no_val", action="store_true", help="Disable validation evaluation during training")
    parser.add_argument("--val_samples", type=int, default=64, help="Max validation samples per epoch for fast validation (default: 64)")
    parser.add_argument("--test", action="store_true", help="Run test evaluation after training finishes")
    parser.add_argument("--test_samples", type=int, default=64, help="Max samples for test evaluation (default: 64)")
    return parser.parse_args()

def train_phase1(args=None):
    if args is None:
        args = parse_args()

    # 1. Configuration (per Section 4.1 & 4.2)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    NUM_CHANNELS = 12
    SEQ_LENGTH = 5000
    PATCH_SIZE = args.patch_size
    MASK_RATIO = args.mask_ratio
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data Preparation
    train_datasets = []
    val_datasets = []
    
    if args.use_dummy or args.dataset == "dummy":
        print("Initializing dummy dataset for Adult ECG (12-lead)...")
        num_samples = args.max_samples or 320
        dataset = DummyECGDataset(num_samples=num_samples, num_channels=NUM_CHANNELS, seq_length=SEQ_LENGTH)
        val_dataset = DummyECGDataset(num_samples=min(num_samples // 4, 64), num_channels=NUM_CHANNELS, seq_length=SEQ_LENGTH)
    else:
        # Load PTB-XL if requested
        if args.dataset in ["all3", "both", "ptbxl"]:
            if os.path.exists(os.path.join(args.ptbxl_dir, "ptbxl_database.csv")):
                print(f"Loading PTB-XL from '{args.ptbxl_dir}' (500 Hz, 12 leads, 5000 samples)...")
                ptbxl_train = PTBXLECGDataset(data_dir=args.ptbxl_dir, sampling_rate=500, split="train")
                print(f"Loaded {len(ptbxl_train)} training records from PTB-XL.")
                train_datasets.append(ptbxl_train)

                if not args.no_val:
                    ptbxl_val = PTBXLECGDataset(data_dir=args.ptbxl_dir, sampling_rate=500, split="val")
                    val_datasets.append(ptbxl_val)
            else:
                print(f"Warning: PTB-XL not found at '{args.ptbxl_dir}'.")

        # Load Chapman if requested
        if args.dataset in ["all3", "both", "chapman"]:
            if os.path.exists(os.path.join(args.chapman_dir, "RECORDS")):
                print(f"Loading Chapman from '{args.chapman_dir}' (500 Hz, 12 leads, 5000 samples)...")
                chapman_train = ChapmanECGDataset(data_dir=args.chapman_dir, split="train")
                print(f"Loaded {len(chapman_train)} training records from Chapman.")
                train_datasets.append(chapman_train)

                if not args.no_val:
                    chapman_val = ChapmanECGDataset(data_dir=args.chapman_dir, split="val")
                    val_datasets.append(chapman_val)
            else:
                print(f"Warning: Chapman not found at '{args.chapman_dir}'.")

        # Load MIMIC-IV-ECG if requested
        if args.dataset in ["all3", "mimic"]:
            mimic_csv = os.path.join(args.mimic_dir, "record_list.csv")
            if os.path.exists(mimic_csv):
                print(f"Loading MIMIC-IV-ECG from '{args.mimic_dir}' (500 Hz, 12 leads, 5000 samples)...")
                mimic_train = MIMICIVECGDataset(data_dir=args.mimic_dir, split="train")
                print(f"Loaded {len(mimic_train)} training records from MIMIC-IV-ECG.")
                train_datasets.append(mimic_train)

                if not args.no_val:
                    mimic_val = MIMICIVECGDataset(data_dir=args.mimic_dir, split="val")
                    val_datasets.append(mimic_val)
            else:
                print(f"Warning: MIMIC-IV-ECG not found at '{args.mimic_dir}'.")

        if not train_datasets:
            print("No real datasets found. Falling back to dummy dataset...")
            num_samples = args.max_samples or 320
            dataset = DummyECGDataset(num_samples=num_samples, num_channels=NUM_CHANNELS, seq_length=SEQ_LENGTH)
            val_dataset = None
        elif len(train_datasets) == 1:
            dataset = train_datasets[0]
        else:
            dataset = ConcatDataset(train_datasets)
            src_names = " + ".join(
                ["PTB-XL"] * int(args.dataset in ["all3", "both", "ptbxl"])
                + ["Chapman"] * int(args.dataset in ["all3", "both", "chapman"])
                + ["MIMIC-IV-ECG"] * int(args.dataset in ["all3", "mimic"])
            )
            print(f"Combined total training records: {len(dataset)} ({src_names}).")

        # Subsample training set if max_samples is specified
        if args.max_samples is not None and len(dataset) > args.max_samples:
            dataset = Subset(dataset, list(range(args.max_samples)))
            print(f"Subsampled training set to {len(dataset)} records for this run.")

        # Prepare validation dataset
        if not args.no_val and val_datasets:
            if len(val_datasets) == 1:
                val_dataset = val_datasets[0]
            else:
                val_dataset = ConcatDataset(val_datasets)
            if args.val_samples is not None and len(val_dataset) > args.val_samples:
                val_dataset = Subset(val_dataset, list(range(args.val_samples)))
        else:
            val_dataset = None

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_dataset is not None else None

    # 3. Model Initialization (Section 4.2)
    print(f"Initializing Transformer Foundation Encoder (depth={args.num_layers}, dim={args.embed_dim}, heads={args.num_heads}, dropout={args.dropout})...")
    model = FoundationEncoder(
        in_channels=NUM_CHANNELS, 
        patch_size=PATCH_SIZE, 
        seq_length=SEQ_LENGTH,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learned_pos=True,
    ).to(device)

    # 4. Loss and Optimizer (Section 4.2: AdamW, weight_decay=0.05 on non-bias, non-norm parameters)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'pos_embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=LEARNING_RATE)

    # Cosine Annealing learning rate schedule
    total_steps = len(dataloader) * EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    # Ensure checkpoint directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')

    # 5. Training Loop
    print(f"Starting Training - Phase 1: Self-Supervised Foundation Learning ({EPOCHS} epochs, mask_ratio={MASK_RATIO})")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, original_signals in enumerate(dataloader):
            original_signals = original_signals.to(device)
            
            # Apply contiguous span block masking (Section 4.2)
            masked_signals, mask = apply_block_masking(
                original_signals, 
                patch_size=PATCH_SIZE, 
                mask_ratio=MASK_RATIO
            )
            
            optimizer.zero_grad()
            
            # Forward pass: reconstruct signal from masked input
            reconstructed_signals = model(masked_signals)
            
            # Pre-training loss: MSE computed strictly on masked patches (Section 4.2)
            masked_diff = (reconstructed_signals - original_signals) * mask
            loss = (masked_diff ** 2).sum() / mask.sum().clamp(min=1.0)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / max(len(dataloader), 1)
        current_lr = scheduler.get_last_lr()[0]

        # Validation evaluation
        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for v_signals in val_dataloader:
                    v_signals = v_signals.to(device)
                    v_masked, v_mask = apply_block_masking(v_signals, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO)
                    v_recon = model(v_masked)
                    v_diff = (v_recon - v_signals) * v_mask
                    v_loss = (v_diff ** 2).sum() / v_mask.sum().clamp(min=1.0)
                    total_val_loss += v_loss.item()
            avg_val_loss = total_val_loss / max(len(val_dataloader), 1)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), args.best_save_path)
                print(f"  -> Saved new best model checkpoint to '{args.best_save_path}' (Val Loss: {best_val_loss:.6f})")
        else:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.6e}")

    print("Phase 1 pre-training completed successfully!")
    
    # Save the latest foundation encoder checkpoint
    torch.save(model.state_dict(), args.save_path)
    print(f"Latest model checkpoint saved to '{args.save_path}'")

    # 6. Automatic Test Evaluation (if --test specified)
    if args.test:
        print("\n--- Running Held-Out Test Evaluation ---")
        from evaluation.evaluate_phase1 import evaluate_phase1
        eval_args = argparse.Namespace(
            checkpoint=args.best_save_path if os.path.exists(args.best_save_path) else args.save_path,
            dataset=args.dataset,
            ptbxl_dir=args.ptbxl_dir,
            chapman_dir=args.chapman_dir,
            mimic_dir=args.mimic_dir,
            batch_size=args.batch_size,
            max_samples=args.test_samples,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            save_plot="reports/reconstruction_sample.png",
            output_json="reports/eval_metrics.json",
        )
        evaluate_phase1(eval_args)

if __name__ == "__main__":
    train_phase1()
