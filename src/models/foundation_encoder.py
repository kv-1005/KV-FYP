import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (Batch, SeqLen, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class FoundationEncoder(nn.Module):
    def __init__(self, in_channels=12, patch_size=50, embed_dim=256, num_layers=4, num_heads=8, seq_length=5000):
        """
        Transformer-based Foundation Encoder for ECG self-supervised learning.
        
        Args:
            in_channels (int): Number of ECG leads (default 12).
            patch_size (int): The size of the signal patch to be embedded as a single token.
            embed_dim (int): The embedding dimension for the transformer.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            seq_length (int): The full length of the input sequence.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        assert seq_length % patch_size == 0, "seq_length must be divisible by patch_size"
        self.num_patches = seq_length // patch_size
        
        # Patch Embedding using 1D Convolution
        # It takes in (B, in_channels, L) and outputs (B, embed_dim, L/patch_size)
        self.patch_embed = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=self.num_patches)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reconstruction Head
        # Projects the embedding back to the patch size per channel
        # We need to output in_channels * patch_size values per token
        self.reconstruction_head = nn.Linear(embed_dim, in_channels * patch_size)
        
    def forward_features(self, x):
        """
        Passes the input through the transformer encoder and returns the latent representation.
        This will be used later as the perceptual features in Phase 2.
        
        Args:
            x: (B, C, L) where C is channels, L is sequence length
        Returns:
            features: (B, num_patches, embed_dim)
        """
        # Patch embedding: (B, C, L) -> (B, embed_dim, num_patches)
        x = self.patch_embed(x)
        
        # Transpose to (B, num_patches, embed_dim) for Transformer
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer Encoder
        features = self.transformer(x)
        return features

    def forward(self, x):
        """
        Forward pass for masked reconstruction (Phase 1).
        
        Args:
            x: Masked ECG signal (B, C, L)
        Returns:
            reconstructed_signal: The reconstructed ECG signal (B, C, L)
        """
        # Get latent representations
        features = self.forward_features(x)
        
        # Reconstruct the signal patches: (B, num_patches, in_channels * patch_size)
        out = self.reconstruction_head(features)
        
        B, N, _ = out.shape
        # Reshape to separate channels and patch values: (B, num_patches, in_channels, patch_size)
        out = out.view(B, N, self.in_channels, self.patch_size)
        
        # Permute to (B, in_channels, num_patches, patch_size)
        out = out.permute(0, 2, 1, 3)
        
        # Reshape to recover the full sequence: (B, in_channels, num_patches * patch_size)
        reconstructed_signal = out.reshape(B, self.in_channels, -1)
        
        return reconstructed_signal
