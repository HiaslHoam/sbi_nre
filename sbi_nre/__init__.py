"""sbi_nre: a small, readable neural ratio estimation (NRE) toolkit.

Typical workflow::

    from sbi_nre import *

    grid = load_grid('data/my_grid.pt')
    training, validation, test = split(grid)
    augment = [step('add_noise', key='signal', sigma_range=[0.01, 0.1])]
    preprocess = [step('normalise_minmax', key='obs')]

    example = apply(preprocess, apply(augment, training))
    model = build_model(example, param_groups=[('a', 'b'), ('c',)],
                        observation=['obs_norm', 'obs_min', 'obs_max'], context=['sigma'])
    train(model, training, validation, 'runs/my_run', augment=augment, preprocess=preprocess)

    run = load_run('runs/my_run')
    result = posterior(run, observation, prior_samples=test)
"""
from .data import batches, load_grid, n_samples, save_grid, select, split, to_device
from .network import RatioEstimator, build_model
from .posterior import (coverage, credibility_of_truth, effective_sample_size, load_posterior, marginal, posterior, save_posterior,
                        summarise, uniform_prior_samples, weighted_quantiles)
from .plotting import plot_coverage, plot_history, plot_marginals, plot_posterior, plot_true_vs_median
from .runs import Run, load_run
from .training import nre_loss, train
from .transforms import TRANSFORMS, apply, step, transform

__all__ = [
    'batches', 'load_grid', 'n_samples', 'save_grid', 'select', 'split', 'to_device',
    'RatioEstimator', 'build_model',
    'coverage', 'credibility_of_truth', 'effective_sample_size', 'load_posterior', 'marginal', 'posterior', 'save_posterior', 'summarise',
    'uniform_prior_samples', 'weighted_quantiles',
    'plot_coverage', 'plot_history', 'plot_marginals', 'plot_posterior', 'plot_true_vs_median',
    'Run', 'load_run', 'nre_loss', 'train', 'TRANSFORMS', 'apply', 'step', 'transform',
]
