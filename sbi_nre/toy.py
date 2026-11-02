"""A toy simulator: an absorption line on a sloped continuum.

Four parameters, drawn uniformly from PRIOR:

    signal(x) = 1 + slope * x - depth * exp(-(x - centre)^2 / (2 width^2))

on a grid of wavelengths x in [-1, 1]. The simulations are noise-free; noise is
added during training (see transforms.add_noise).
"""
import torch

PRIOR = {'depth': (0.05, 0.5), 'centre': (-0.6, 0.6), 'width': (0.03, 0.2), 'slope': (-0.3, 0.3)}


def wavelengths(n_bins=100):
    return torch.linspace(-1, 1, n_bins)


def simulate_toy(n, *, n_bins=100, seed=0):
    """Return a grid {param: [n], 'signal': [n, n_bins]} of noise-free toy spectra."""
    generator = torch.Generator().manual_seed(seed)
    grid = {name: low + (high - low) * torch.rand(n, generator=generator)
            for name, (low, high) in PRIOR.items()}
    x = wavelengths(n_bins)
    line = torch.exp(-0.5 * ((x - grid['centre'][:, None]) / grid['width'][:, None]) ** 2)
    grid['signal'] = 1 + grid['slope'][:, None] * x - grid['depth'][:, None] * line
    return grid
