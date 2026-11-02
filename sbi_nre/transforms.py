"""Augmentation and preprocessing steps.

A transform is a function that takes the data dictionary and returns a new one::

    @transform
    def my_step(data, *, key='obs'):
        return {**data, f'{key}_log': data[key].log10()}

Runs store their steps by name, as ``step('my_step', key='obs')``, so they can be
saved to ``config.json`` and repeated when a run is loaded. Define (or import) your
own transforms before loading a run that uses them.

* augment steps run on simulations only, during training (e.g. adding noise);
* preprocess steps run on everything given to the network: simulations during
  training and real observations afterwards (e.g. normalisation).

Keyword arguments must be JSON-friendly (numbers, strings, lists).
"""
import torch

TRANSFORMS = {}


def transform(function):
    """Register a function so that runs can refer to it by name."""
    TRANSFORMS[function.__name__] = function
    return function


def step(name, **kwargs):
    """One augmentation or preprocessing step, as stored in a run's config."""
    if name not in TRANSFORMS:
        raise KeyError(f'Unknown transform {name!r}; register it with @transform. Known: {sorted(TRANSFORMS)}')
    return {'name': name, 'kwargs': kwargs}


def apply(steps, data):
    """Apply the steps in order and return the new data dictionary."""
    for s in steps:
        if s['name'] not in TRANSFORMS:
            raise KeyError(f"Transform {s['name']!r} is not registered; define or import it first.")
        data = TRANSFORMS[s['name']](data, **s['kwargs'])
    return data


@transform
def add_noise(data, *, key='signal', out='obs', sigma_key='sigma', sigma_range=(0.01, 0.1), log_uniform=True):
    """Add Gaussian noise to ``data[key]`` and store the result as ``data[out]``.

    Every simulation gets its own noise level, drawn from ``sigma_range``
    (log-uniform by default), saved as ``data[sigma_key]``. Give that key to the
    network as context to tell it how noisy each spectrum is.
    """
    clean = data[key]
    n = len(clean)
    low, high = torch.tensor(sigma_range, dtype=clean.dtype, device=clean.device)
    u = torch.rand(n, device=clean.device, dtype=clean.dtype)
    sigma = low * (high / low) ** u if log_uniform else low + (high - low) * u
    noisy = clean + sigma.view(n, *[1] * (clean.ndim - 1)) * torch.randn_like(clean)
    return {**data, out: noisy, sigma_key: sigma}


@transform
def normalise_minmax(data, *, key='obs'):
    """Scale every spectrum to the range 0-1.

    Adds ``{key}_norm`` and keeps the original minimum and maximum as
    ``{key}_min`` and ``{key}_max``, so the network can still use the absolute level.
    """
    x = data[key]
    low = x.amin(dim=-1, keepdim=True)
    high = x.amax(dim=-1, keepdim=True)
    return {**data, f'{key}_norm': (x - low) / (high - low + 1e-12),
            f'{key}_min': low.squeeze(-1), f'{key}_max': high.squeeze(-1)}


@transform
def log10(data, *, keys):
    """Replace each entry in ``keys`` by its base-10 logarithm."""
    return {**data, **{key: data[key].log10() for key in keys}}
