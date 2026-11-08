"""Load, split and batch a simulation grid.

A grid is a plain dictionary of tensors that all have the same first dimension
(one entry per simulation), for example::

    {'amplitude': [N], 'centre': [N], 'signal': [N, 100]}

Parameters are 1D tensors; simulated data can have any shape after the first
dimension. Save it with ``save_grid`` as ``.pt`` (or as ``.npz`` with NumPy).
"""
from pathlib import Path
import numpy as np
import torch


def n_samples(data):
    """Number of simulations; checks that every entry has the same length."""
    lengths = {len(value) for value in data.values()}
    if len(lengths) != 1:
        raise ValueError(f'All entries need the same first dimension, got lengths {sorted(lengths)}.')
    return lengths.pop()


def load_grid(path, *, device='cpu'):
    """Load a grid from ``.pt`` (a saved dictionary of tensors) or ``.npz``."""
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path) as archive:
            data = {key: torch.as_tensor(archive[key]) for key in archive.files}
    else:
        data = torch.load(path, map_location=device, weights_only=True)
    data = {key: torch.as_tensor(value) for key, value in data.items()}
    # float32 before moving: Apple GPUs (MPS) do not support float64.
    data = {key: (value.float() if value.is_floating_point() else value).to(device) for key, value in data.items()}
    n_samples(data)
    return data


def save_grid(data, path):
    """Save a grid as ``.pt``."""
    n_samples(data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: torch.as_tensor(value).cpu() for key, value in data.items()}, path)


def select(data, index):
    """The simulations at ``index`` (an integer tensor, slice or boolean mask)."""
    return {key: value[index] for key, value in data.items()}


def to_device(data, device):
    """Move a grid to ``device``; float64 becomes float32 on Apple GPUs (MPS), which lack float64."""
    mps = torch.device(device).type == 'mps'
    return {key: (value.float() if mps and value.dtype == torch.float64 else value).to(device)
            for key, value in data.items()}


def split(data, fractions=(0.8, 0.1, 0.1), *, seed=0):
    """Randomly split a grid, e.g. into training, validation and test sets.

    The same seed always gives the same split.
    """
    n = n_samples(data)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    sizes = [int(round(f * n)) for f in fractions]
    sizes[-1] = n - sum(sizes[:-1])
    parts, start = [], 0
    for size in sizes:
        parts.append(select(data, order[start:start + size].to(next(iter(data.values())).device)))
        start += size
    return parts


def batches(data, batch_size, *, shuffle=False, generator=None):
    """Iterate over the grid in batches of ``batch_size`` simulations."""
    n = n_samples(data)
    device = next(iter(data.values())).device
    order = (torch.randperm(n, generator=generator).to(device) if shuffle
             else torch.arange(n, device=device))
    for start in range(0, n, batch_size):
        yield select(data, order[start:start + batch_size])
