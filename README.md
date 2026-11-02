# SBI_NRE

A small, readable toolkit for **neural ratio estimation (NRE)**, a method of
simulation-based inference. Train a network on any grid of simulations, save the
run, load it again, and compute posteriors, in a few lines of code and with only
PyTorch, NumPy and Matplotlib.

## The idea in one paragraph

You have a simulator: give it parameters θ and it produces data x (for example a
spectrum). You want the posterior p(θ | x) for an observation, but the likelihood
is unknown or too expensive. NRE trains a classifier to distinguish *matching*
pairs (x and the θ that produced it) from *mismatched* pairs (x with θ from another
simulation). The trained classifier gives the ratio r(x, θ) = p(θ | x) / p(θ), so
weighting samples from the prior by r gives the posterior. Once trained, the
network gives the posterior for any new observation in a fraction of a second.

## Install

```bash
git clone <this repository> SBI_NRE
cd SBI_NRE
python -m pip install -e .          # needs torch, numpy, matplotlib
```

Then open `notebooks/01_quickstart.ipynb` and run all cells (a few minutes on a
laptop CPU, less on a GPU).

## Notebooks

| Notebook | What it shows |
|---|---|
| `01_quickstart.ipynb` | The whole workflow on a toy model (an absorption line on a continuum), with explanations. |
| `02_your_own_grid.ipynb` | A template: point it at your own grid, choose parameters and inputs in one settings cell, train and check. |

## The workflow in code

```python
from sbi_nre import *

grid = load_grid('data/my_grid.pt')                     # {param: [N], 'signal': [N, 100], ...}
training, validation, test = split(grid, (0.8, 0.1, 0.1))

augment = [step('add_noise', key='signal', out='obs', sigma_range=[0.01, 0.1])]   # training only
preprocess = [step('normalise_minmax', key='obs')]                                # always
example = apply(preprocess, apply(augment, training))

model = build_model(example, param_groups=[('a', 'b'), ('c',)],
                    observation=['obs_norm', 'obs_min', 'obs_max'], context=['sigma'])
train(model, training, validation, 'runs/my_run', augment=augment, preprocess=preprocess)

run = load_run('runs/my_run')
result = posterior(run, observation={'obs': my_spectrum, 'sigma': 0.03}, prior_samples=validation)
plot_posterior(result)
summarise(result)          # {param: (16th, 50th, 84th percentile)}
```

## Key terms

| Term | Meaning here |
|---|---|
| **grid** | A dictionary of tensors, one entry per simulation: parameters `[N]` and data `[N, ...]`. Saved as `.pt` or `.npz`. |
| **prior** | The distribution the grid's parameters were drawn from. Held-out simulations serve as prior samples. |
| **parameter group** | A set of one or two parameters whose (1D or 2D) posterior the network learns. Any combination is allowed. |
| **observation** | The measured data given to the network (e.g. a normalised spectrum). |
| **context** | Other known information given to the network (e.g. the noise level, a known stellar property). |
| **augment** | Steps applied to simulations during training only, e.g. adding noise. Redrawn for every batch. |
| **preprocess** | Steps applied to everything the network sees, simulations and real data, e.g. normalisation. |
| **run** | A folder with the settings (`config.json`), losses (`history.json`) and network weights (`best.pt`, `last.pt`). |
| **posterior result** | Prior samples with weights, per parameter group; `summarise`, `marginal` and the plots read it. |

## Adding your own step

```python
from sbi_nre import transform, step

@transform
def log_spectrum(data, *, key='obs'):
    return {**data, f'{key}_log': data[key].log10()}

preprocess = [step('log_spectrum', key='obs')]
```

The step is saved by name and settings. Define (or import) the function again
before loading a run that uses it.

## Package layout

```text
sbi_nre/
  data.py        load, save, split and batch grids
  transforms.py  augmentation and preprocessing steps (add_noise, normalise_minmax, log10, your own)
  network.py     the network: shared embedding plus one classifier per parameter group
  training.py    the NRE loss and the training loop; writes the run folder
  runs.py        load a run
  posterior.py   posteriors by re-weighting prior samples; summaries; save/load
  plotting.py    training curves, posteriors, true-vs-median checks
  toy.py         the toy simulator used in the quickstart
notebooks/       01 quickstart, 02 template for your own grid
tests/           quick checks: python -m pytest
```

`data/` and `runs/` hold your grids and training runs and are not tracked by Git.

## Good to know

- **The loss.** It starts near 2 ln 2 ≈ 1.39 (guessing) and falls as the network
  learns. A validation loss that rises while the training loss keeps falling means
  overfitting: use more simulations, a smaller network, or stop earlier (`best.pt`
  already keeps the best epoch).
- **Inputs and parameters are standardised automatically** using the training set,
  so you do not need to rescale them yourself.
- **Prior samples** must follow the same prior as the training grid. Held-out
  simulations (the validation set) do this automatically. More prior samples give a
  smoother posterior.
- **Reproducibility.** `split` and `train` take a `seed`; all settings are stored in
  `config.json`.
- **Checking your network.** A true-vs-median plot on test simulations (notebooks
  01 and 02) is the first check; formal calibration tests (e.g. coverage) come next.
