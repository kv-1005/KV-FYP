import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add the 'src' directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dummy_dataset import DummyECGDataset, apply_random_masking
from models.foundation_encoder import FoundationEncoder

def train_phase1():
    # 1. Configuration
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CHANNELS = 12
    SEQ_LENGTH = 5000
    PATCH_SIZE = 50
    MASK_RATIO = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Initializing dummy dataset for Adult ECG (12-lead)...")
    dataset = DummyECGDataset(num_samples=320, num_channels=NUM_CHANNELS, seq_length=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Model Initialization
    print("Initializing Transformer Foundation Encoder...")
    model = FoundationEncoder(
        in_channels=NUM_CHANNELS, 
        patch_size=PATCH_SIZE, 
        seq_length=SEQ_LENGTH,
        embed_dim=256,
        num_layers=4,
        num_heads=8
    ).to(device)

    # 4. Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting Training - Phase 1: Self-Supervised Foundation Learning")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, original_signals in enumerate(dataloader):
            original_signals = original_signals.to(device)
            
            # Apply masking to simulate the self-supervised objective
            masked_signals, mask = apply_random_masking(
                original_signals, 
                patch_size=PATCH_SIZE, 
                mask_ratio=MASK_RATIO
            )
            
            optimizer.zero_grad()
            
            # Forward pass: reconstruct the signal from the masked input
            reconstructed_signals = model(masked_signals)
            
            # Compute loss. We only care about reconstructing the masked portions
            # Alternatively, can compute loss over the whole sequence.
            # Mask is 1 where the signal was hidden, 0 where it was visible
            
            # Compute MSE on the masked patches only
            loss = criterion(reconstructed_signals * mask, original_signals * mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    print("Phase 1 pre-training completed successfully!")
    
    # Save the foundation encoder checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/foundation_encoder_phase1.pth')
    print("Model checkpoint saved to 'checkpoints/foundation_encoder_phase1.pth'")

if __name__ == "__main__":
    train_phase1()
