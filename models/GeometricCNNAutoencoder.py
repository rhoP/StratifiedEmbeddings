#!/usr/bin/env python3
"""
Geometric CNN Autoencoder with Mesh Morphing and Consistency Loss.

This module implements a geometry-aware autoencoder that:
1. Uses a reference domain for all simulations
2. Morphs meshes to handle different geometries
3. Trains with overlap consistency for local patches
4. Preserves geometric structure in latent space
"""

import os
import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import distance_matrix

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available")
    TORCH_AVAILABLE = False


# =========================
# Reference Mesh Utilities
# =========================

class ReferenceDomain:
    """
    Reference domain for mesh morphing.

    This defines a canonical coordinate system that all simulations
    can be mapped to.
    """

    def __init__(self, grid_size=(128, 128), bounds=None):
        """
        Args:
            grid_size: Size of reference grid (H, W)
            bounds: Domain bounds dict with keys: x_min, x_max, y_min, y_max
        """
        self.grid_size = grid_size

        if bounds is None:
            # Default bounds for cylinder flow
            bounds = {'x_min': -10.0, 'x_max': 40.0,
                     'y_min': -15.0, 'y_max': 15.0}

        self.bounds = bounds

        # Create reference grid
        height, width = grid_size
        x = np.linspace(bounds['x_min'], bounds['x_max'], width)
        y = np.linspace(bounds['y_min'], bounds['y_max'], height)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

        # Store grid coordinates as (N, 2) array
        self.ref_coords = np.stack([self.grid_x.ravel(),
                                    self.grid_y.ravel()], axis=1)

    def get_grid(self):
        """Return reference grid coordinates."""
        return self.grid_x, self.grid_y

    def interpolate_field(self, coordinates, field_values):
        """
        Interpolate field onto reference grid.

        Args:
            coordinates: Nx2 or Nx3 array of coordinates
            field_values: Nx1 or NxM array of field values

        Returns:
            grid_field: HxW or HxWxM interpolated field
        """
        # Use 2D coordinates
        if coordinates.shape[1] == 3:
            coords_2d = coordinates[:, :2]
        else:
            coords_2d = coordinates

        # Interpolate
        if field_values.ndim == 1:
            # Scalar field
            grid_field = griddata(coords_2d, field_values,
                                 (self.grid_x, self.grid_y),
                                 method='linear', fill_value=0)
        else:
            # Vector field
            n_components = field_values.shape[1]
            grid_field = np.zeros((*self.grid_size, n_components))

            for i in range(n_components):
                grid_field[:, :, i] = griddata(coords_2d, field_values[:, i],
                                              (self.grid_x, self.grid_y),
                                              method='linear', fill_value=0)

        return grid_field


class MeshMorpher:
    """
    Morphs meshes between different geometries using Radial Basis Functions.
    """

    def __init__(self, reference_shape, target_shape, smoothness=1.0):
        """
        Args:
            reference_shape: Dict with reference geometry parameters
            target_shape: Dict with target geometry parameters
            smoothness: RBF smoothness parameter
        """
        self.reference_shape = reference_shape
        self.target_shape = target_shape
        self.smoothness = smoothness
        self.displacement_field = None

        # Compute displacement field
        self._compute_displacement()

    def _compute_displacement(self):
        """Compute displacement field from reference to target."""
        # For cylinder: compute displacement based on radius/position changes
        ref_r = self.reference_shape.get('radius', 0.5)
        ref_x = self.reference_shape.get('x', 0.0)
        ref_y = self.reference_shape.get('y', 0.0)

        tar_r = self.target_shape.get('radius', 0.5)
        tar_x = self.target_shape.get('x', 0.0)
        tar_y = self.target_shape.get('y', 0.0)

        # Store displacement parameters
        self.displacement_params = {
            'dx': tar_x - ref_x,
            'dy': tar_y - ref_y,
            'dr': tar_r - ref_r
        }

    def morph_coordinates(self, coords):
        """
        Morph coordinates from reference to target geometry.

        Args:
            coords: Nx2 array of coordinates

        Returns:
            morphed_coords: Nx2 array of morphed coordinates
        """
        # Simple morphing: translate and scale based on cylinder position/size
        morphed = coords.copy()

        # Get displacement parameters
        dx = self.displacement_params['dx']
        dy = self.displacement_params['dy']
        dr = self.displacement_params['dr']

        ref_x = self.reference_shape.get('x', 0.0)
        ref_y = self.reference_shape.get('y', 0.0)
        ref_r = self.reference_shape.get('radius', 0.5)

        # Compute distance from reference cylinder center
        dist = np.sqrt((coords[:, 0] - ref_x)**2 + (coords[:, 1] - ref_y)**2)

        # Apply radial scaling (decays with distance)
        decay = np.exp(-dist / (5 * ref_r))

        # Translate
        morphed[:, 0] += dx * decay
        morphed[:, 1] += dy * decay

        # Radial scaling near cylinder
        if abs(dr) > 1e-6:
            # Direction from center
            direction_x = (coords[:, 0] - ref_x) / (dist + 1e-10)
            direction_y = (coords[:, 1] - ref_y) / (dist + 1e-10)

            # Apply radial displacement
            radial_disp = dr * decay
            morphed[:, 0] += radial_disp * direction_x
            morphed[:, 1] += radial_disp * direction_y

        return morphed

    def inverse_morph_coordinates(self, coords):
        """
        Inverse morphing: from target to reference.

        Args:
            coords: Nx2 array of coordinates in target space

        Returns:
            ref_coords: Nx2 array of coordinates in reference space
        """
        # Approximate inverse by negating displacement
        ref_coords = coords.copy()

        dx = self.displacement_params['dx']
        dy = self.displacement_params['dy']
        dr = self.displacement_params['dr']

        tar_x = self.target_shape.get('x', 0.0)
        tar_y = self.target_shape.get('y', 0.0)
        tar_r = self.target_shape.get('radius', 0.5)

        # Compute distance from target cylinder center
        dist = np.sqrt((coords[:, 0] - tar_x)**2 + (coords[:, 1] - tar_y)**2)

        # Apply inverse decay
        decay = np.exp(-dist / (5 * tar_r))

        # Inverse translate
        ref_coords[:, 0] -= dx * decay
        ref_coords[:, 1] -= dy * decay

        # Inverse radial scaling
        if abs(dr) > 1e-6:
            direction_x = (coords[:, 0] - tar_x) / (dist + 1e-10)
            direction_y = (coords[:, 1] - tar_y) / (dist + 1e-10)

            radial_disp = -dr * decay
            ref_coords[:, 0] += radial_disp * direction_x
            ref_coords[:, 1] += radial_disp * direction_y

        return ref_coords

    def get_displacement_field(self, grid_coords):
        """
        Get displacement field on a grid.

        Args:
            grid_coords: HxWx2 array of grid coordinates

        Returns:
            displacement: HxWx2 array of displacement vectors
        """
        shape = grid_coords.shape[:-1]
        coords_flat = grid_coords.reshape(-1, 2)

        # Morph coordinates
        morphed = self.morph_coordinates(coords_flat)

        # Compute displacement
        displacement = morphed - coords_flat

        return displacement.reshape(*shape, 2)


# =========================
# Patch Extraction
# =========================

class GeometricPatchExtractor:
    """
    Extracts patches with geometric information.
    """

    def __init__(self, patch_size=(64, 64), stride=16):
        """
        Args:
            patch_size: Size of patches (H, W)
            stride: Stride for patch extraction
        """
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches_with_positions(self, flow_field, displacement_field=None):
        """
        Extract patches from flow and displacement fields.

        Args:
            flow_field: HxWxC flow field
            displacement_field: HxWx2 displacement field (optional)

        Returns:
            flow_patches: List of flow patches
            disp_patches: List of displacement patches (if provided)
            positions: List of (i, j) patch positions
            overlap_map: Dict mapping positions to overlapping patches
        """
        if flow_field.ndim == 2:
            flow_field = flow_field[:, :, np.newaxis]

        height, width = flow_field.shape[:2]
        patch_h, patch_w = self.patch_size

        flow_patches = []
        disp_patches = [] if displacement_field is not None else None
        positions = []

        # Extract patches
        for i in range(0, height - patch_h + 1, self.stride):
            for j in range(0, width - patch_w + 1, self.stride):
                # Extract flow patch
                flow_patch = flow_field[i:i+patch_h, j:j+patch_w, :]
                flow_patches.append(flow_patch)

                # Extract displacement patch if provided
                if displacement_field is not None:
                    disp_patch = displacement_field[i:i+patch_h, j:j+patch_w, :]
                    disp_patches.append(disp_patch)

                positions.append((i, j))

        # Compute overlap map
        overlap_map = self._compute_overlap_map(positions, patch_h, patch_w)

        return (np.array(flow_patches),
                np.array(disp_patches) if disp_patches else None,
                positions,
                overlap_map)

    def _compute_overlap_map(self, positions, patch_h, patch_w):
        """
        Compute which patches overlap.

        Returns:
            overlap_map: Dict mapping patch index to list of overlapping patch indices
        """
        overlap_map = {}

        for idx1, (i1, j1) in enumerate(positions):
            overlaps = []

            for idx2, (i2, j2) in enumerate(positions):
                if idx1 == idx2:
                    continue

                # Check if patches overlap
                if (abs(i1 - i2) < patch_h and abs(j1 - j2) < patch_w):
                    overlaps.append(idx2)

            overlap_map[idx1] = overlaps

        return overlap_map


# =========================
# Geometric CNN Autoencoder
# =========================

class GeometricConvAutoencoder(nn.Module):
    """
    Geometry-aware convolutional autoencoder.

    Takes both flow patches and shape displacement patches as input.
    """

    def __init__(self, latent_dim=64, flow_channels=1, geom_channels=2, base_channels=32):
        """
        Args:
            latent_dim: Dimension of latent space
            flow_channels: Number of flow field channels
            geom_channels: Number of geometry channels (displacement field)
            base_channels: Base number of filters
        """
        super(GeometricConvAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.flow_channels = flow_channels
        self.geom_channels = geom_channels

        # Combined input channels
        input_channels = flow_channels + geom_channels

        # Encoder (64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4)
        self.encoder = nn.Sequential(
            # Conv1: 64x64 -> 32x32
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Conv2: 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            # Conv3: 16x16 -> 8x8
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            # Conv4: 8x8 -> 4x4
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
        )

        # Latent space
        self.flatten_dim = base_channels * 8 * 4 * 4
        self.fc_encode = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64)
        self.decoder = nn.Sequential(
            # Deconv1: 4x4 -> 8x8
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            # Deconv2: 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            # Deconv3: 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Deconv4: 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels, flow_channels, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, flow_patch, geom_patch=None):
        """
        Encode flow and geometry patches.

        Args:
            flow_patch: Flow field patch (N, C_flow, H, W)
            geom_patch: Geometry patch (N, C_geom, H, W) - optional

        Returns:
            z: Latent representation (N, latent_dim)
        """
        # Combine flow and geometry
        if geom_patch is not None:
            x = torch.cat([flow_patch, geom_patch], dim=1)
        else:
            # If no geometry provided, use zeros
            batch_size = flow_patch.size(0)
            geom_dummy = torch.zeros(batch_size, self.geom_channels,
                                    *flow_patch.shape[2:],
                                    device=flow_patch.device)
            x = torch.cat([flow_patch, geom_dummy], dim=1)

        # Encode
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        z = self.fc_encode(x)

        return z

    def decode(self, z):
        """Decode latent representation."""
        x = self.fc_decode(z)
        x = x.reshape(x.size(0), -1, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, flow_patch, geom_patch=None):
        """Forward pass."""
        z = self.encode(flow_patch, geom_patch)
        recon = self.decode(z)
        return recon, z


def geometric_autoencoder_loss(model, flow_patches, geom_patches, overlap_map,
                               lambda_consistency=0.1, lambda_metric=0.01,
                               batch_indices=None):
    """
    Compute loss with consistency and metric regularization.

    Args:
        model: GeometricConvAutoencoder model
        flow_patches: Flow field patches (N, C, H, W)
        geom_patches: Geometry patches (N, C_geom, H, W)
        overlap_map: Dict mapping patch indices to overlapping patches
        lambda_consistency: Weight for consistency loss
        lambda_metric: Weight for metric regularization
        batch_indices: Optional list of global indices for patches in this batch

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    # Reconstruction loss
    recon, z = model(flow_patches, geom_patches)
    L_recon = nn.functional.mse_loss(recon, flow_patches)

    # Consistency loss for overlapping patches
    L_consistency = 0.0
    n_overlaps = 0

    # If batch_indices provided, filter overlap_map to only include patches in this batch
    if batch_indices is not None:
        batch_set = set(batch_indices)
        batch_size = len(batch_indices)

        # Create mapping from global index to batch index
        global_to_batch = {global_idx: batch_idx for batch_idx, global_idx in enumerate(batch_indices)}

        # Compute consistency only for patches in this batch
        for batch_idx, global_idx in enumerate(batch_indices):
            if global_idx not in overlap_map:
                continue

            overlapping_global = overlap_map[global_idx]

            for overlap_global in overlapping_global:
                # Only compute if the overlapping patch is also in this batch
                if overlap_global in batch_set:
                    overlap_batch_idx = global_to_batch[overlap_global]

                    # Compute consistency between batch_idx and overlap_batch_idx
                    z_i = z[batch_idx:batch_idx+1]
                    z_j = z[overlap_batch_idx:overlap_batch_idx+1]
                    L_consistency += nn.functional.mse_loss(z_i, z_j)
                    n_overlaps += 1
    else:
        # Assume overlap_map indices match batch indices (legacy behavior)
        batch_size = z.size(0)

        for idx, overlapping_indices in overlap_map.items():
            # Skip if index out of bounds for this batch
            if idx >= batch_size:
                continue

            if len(overlapping_indices) == 0:
                continue

            z_i = z[idx:idx+1]

            for overlap_idx in overlapping_indices:
                # Skip if overlap index out of bounds
                if overlap_idx >= batch_size:
                    continue

                z_j = z[overlap_idx:overlap_idx+1]
                L_consistency += nn.functional.mse_loss(z_i, z_j)
                n_overlaps += 1

    if n_overlaps > 0:
        L_consistency /= n_overlaps

    # Metric regularization (encourage non-degenerate metric)
    # Compute approximate Jacobian of encoder
    L_metric = 0.0
    if lambda_metric > 0:
        # Simple regularization: encourage latent variance
        L_metric = -torch.log(z.var(dim=0).mean() + 1e-8)

    # Total loss
    total_loss = L_recon + lambda_consistency * L_consistency + lambda_metric * L_metric

    loss_dict = {
        'recon': L_recon.item(),
        'consistency': L_consistency.item() if isinstance(L_consistency, torch.Tensor) else L_consistency,
        'metric': L_metric.item() if isinstance(L_metric, torch.Tensor) else L_metric,
        'total': total_loss.item()
    }

    return total_loss, loss_dict


# =========================
# Dataset
# =========================

class GeometricPatchDataset(Dataset):
    """Dataset for geometric patches."""

    def __init__(self, flow_patches, geom_patches=None):
        """
        Args:
            flow_patches: Array of flow patches (N, H, W, C)
            geom_patches: Array of geometry patches (N, H, W, C_geom)
        """
        # Convert to torch tensors (N, C, H, W)
        self.flow_patches = torch.FloatTensor(flow_patches).permute(0, 3, 1, 2)

        if geom_patches is not None:
            self.geom_patches = torch.FloatTensor(geom_patches).permute(0, 3, 1, 2)
        else:
            # Dummy geometry patches
            self.geom_patches = torch.zeros(len(flow_patches), 2,
                                          flow_patches.shape[1],
                                          flow_patches.shape[2])

    def __len__(self):
        return len(self.flow_patches)

    def __getitem__(self, idx):
        # Return index along with data for batch tracking
        return self.flow_patches[idx], self.geom_patches[idx], idx


# =========================
# Training
# =========================

def train_geometric_autoencoder(model, train_loader, val_loader, overlap_map,
                                epochs=50, lr=1e-3, device='cpu',
                                lambda_consistency=0.1, lambda_metric=0.01,
                                save_dir=None):
    """
    Train geometric autoencoder with consistency loss.

    Args:
        model: GeometricConvAutoencoder
        train_loader: Training data loader
        val_loader: Validation data loader
        overlap_map: Overlap map for consistency loss
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        lambda_consistency: Weight for consistency loss
        lambda_metric: Weight for metric regularization
        save_dir: Directory to save checkpoints

    Returns:
        history: Training history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_recon': [],
        'train_consistency': [],
        'val_loss': []
    }

    best_val_loss = float('inf')

    print(f"\n{'='*70}")
    print(f"Training Geometric CNN Autoencoder")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Latent dimension: {model.latent_dim}")
    print(f"Consistency weight: {lambda_consistency}")
    print(f"Metric weight: {lambda_metric}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_recons = []
        train_consistencies = []

        for batch_data in tqdm(train_loader,
                              desc=f"Epoch {epoch+1}/{epochs} [Train]",
                              leave=False):
            # Unpack batch (includes indices now)
            flow_batch, geom_batch, batch_indices = batch_data
            flow_batch = flow_batch.to(device)
            geom_batch = geom_batch.to(device)

            # Convert batch_indices to list of integers
            batch_indices = batch_indices.tolist()

            # Forward pass with geometric loss
            loss, loss_dict = geometric_autoencoder_loss(
                model, flow_batch, geom_batch, overlap_map,
                lambda_consistency=lambda_consistency,
                lambda_metric=lambda_metric,
                batch_indices=batch_indices
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss_dict['total'])
            train_recons.append(loss_dict['recon'])
            train_consistencies.append(loss_dict['consistency'])

        avg_train_loss = np.mean(train_losses)
        avg_train_recon = np.mean(train_recons)
        avg_train_consistency = np.mean(train_consistencies)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_data in val_loader:
                # Unpack batch (includes indices now)
                flow_batch, geom_batch, _ = batch_data
                flow_batch = flow_batch.to(device)
                geom_batch = geom_batch.to(device)

                recon, _ = model(flow_batch, geom_batch)
                loss = nn.functional.mse_loss(recon, flow_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_consistency'].append(avg_train_consistency)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f} "
              f"(Recon: {avg_train_recon:.6f}, Cons: {avg_train_consistency:.6f}) "
              f"Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if save_dir and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(save_dir) / 'best_geometric_model.pth')

    return history


if __name__ == '__main__':
    print("Geometric CNN Autoencoder Module")
    print("This module provides geometry-aware autoencoders for CFD")
    print("See GeometricCNNAutoencoderTraining.py for training script")
