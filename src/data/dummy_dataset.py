import torch
from torch.utils.data import Dataset

class DummyECGDataset(Dataset):
    """
    A dummy dataset for simulating 12-lead adult ECG recordings.
    This generates random noise tensors to test the model architecture 
    and training loop before the real dataset is ready.
    """
    def __init__(self, num_samples=1000, num_channels=12, seq_length=5000):
        """
        Args:
            num_samples (int): Total number of fake samples.
            num_channels (int): Number of ECG channels (default 12 for adult ECG).
            seq_length (int): Length of the ECG sequence (configurable).
        """
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random tensor representing an ECG signal
        # Shape: (Channels, Sequence Length)
        # Using normal distribution to simulate normalized signals
        signal = torch.randn(self.num_channels, self.seq_length)
        return signal

def apply_random_masking(signal, patch_size=50, mask_ratio=0.5):
    """
    Utility function to apply random masking to an ECG signal.
    Args:
        signal (torch.Tensor): Shape (Batch, Channels, SeqLen) or (Channels, SeqLen)
        patch_size (int): Size of the contiguous patches to mask.
        mask_ratio (float): Fraction of patches to mask (e.g., 0.5 for 50%).
    Returns:
        masked_signal (torch.Tensor): The signal with selected patches set to 0.
        mask (torch.Tensor): Binary mask where 1 indicates a masked position.
    """
    # Handle both unbatched and batched inputs
    if signal.dim() == 2:
        signal = signal.unsqueeze(0)
    
    B, C, L = signal.shape
    num_patches = L // patch_size
    
    # Generate random scores for each patch to determine which ones to mask
    rand_scores = torch.rand(B, num_patches, device=signal.device)
    
    # Sort scores to pick the top mask_ratio portions
    num_mask = int(mask_ratio * num_patches)
    
    # Create patch-level mask
    patch_mask = torch.zeros(B, num_patches, device=signal.device)
    
    # Get indices of top 'num_mask' values (these will be masked)
    _, mask_indices = torch.topk(rand_scores, k=num_mask, dim=-1)
    
    # Scatter 1s into the mask positions
    patch_mask.scatter_(1, mask_indices, 1.0)
    
    # Expand patch mask back to full sequence length
    # Shape: (B, num_patches, 1) -> (B, num_patches, patch_size) -> (B, L)
    full_mask = patch_mask.unsqueeze(-1).repeat(1, 1, patch_size).view(B, -1)
    
    # If the original signal length isn't perfectly divisible by patch_size,
    # pad the mask up to the original length with 0s (unmasked).
    if full_mask.shape[1] < L:
        pad_len = L - full_mask.shape[1]
        padding = torch.zeros(B, pad_len, device=signal.device)
        full_mask = torch.cat([full_mask, padding], dim=1)
    
    # The mask needs to be broadcastable over channels: (B, 1, L)
    full_mask = full_mask.unsqueeze(1)
    
    # Apply the mask (set masked regions to 0)
    masked_signal = signal * (1.0 - full_mask)
    
    return masked_signal, full_mask
