"""Posteriors from a trained network.

The network estimates r(x, θ) = p(θ | x) / p(θ). Given samples from the prior
(e.g. the parameters of held-out simulations), weighting each sample by r turns
them into weighted samples of the posterior. No MCMC is needed.

A posterior result is a dictionary {group name: {'samples': {param: tensor},
'weights': tensor, 'log_ratio': tensor}}, one entry per parameter group, with
weights summing to one.
"""
import warnings
import numpy as np
import torch
from .data import n_samples, select
from .network import group_name
from .transforms import apply


def _batched(observation, device):
    """Add a batch dimension of one to a single observation."""
    return {key: torch.as_tensor(value, device=device).float()[None] for key, value in observation.items()}


@torch.no_grad()
def posterior(run, observation, prior_samples, *, batch_size=10_000, min_ess=100):
    """Weighted posterior samples for one observation, for every parameter group.

    observation: a single example, e.g. {'obs': [100], 'sigma': scalar}. It is
        preprocessed exactly like the training data (augmentation is not applied).
    prior_samples: {param: [M]} drawn from the training prior.
    min_ess: warn if a group's effective sample size is below this (None: never).
        Joint groups of many parameters have narrow posteriors and need many more
        prior samples than 1D or 2D groups.
    """
    model = run.model.eval()
    device = model.input_mean.device
    data = apply(run.preprocess, _batched(observation, device))
    summary = model.summary(data)
    m = n_samples({p: prior_samples[p] for p in model.param_names})
    result = {}
    for group in model.param_groups:
        samples = {p: torch.as_tensor(prior_samples[p], device=device).float() for p in group}
        log_r = torch.cat([
            model.log_ratio(summary.expand(len(chunk[group[0]]), -1), chunk, group)
            for chunk in (select(samples, slice(i, i + batch_size)) for i in range(0, m, batch_size))])
        weights = torch.softmax(log_r, 0)
        result[group_name(group)] = {'samples': samples, 'weights': weights, 'log_ratio': log_r}
        if min_ess is not None and effective_sample_size(weights) < min_ess:
            warnings.warn(f'Group {group_name(group)!r}: effective sample size below {min_ess}. '
                          'The posterior (and coverage) is unreliable; use more prior samples.',
                          stacklevel=2)
    return result


@torch.no_grad()
def credibility_of_truth(run, observations, prior_samples, *, batch_size=10_000, min_ess=100):
    """For every test simulation and parameter group: the credibility level of its true parameters.

    This is the posterior mass of all prior samples that the network finds more
    probable than the truth (a higher ratio). The truth lies inside every credible
    region with a higher level than this. For a well-calibrated network these
    levels are spread uniformly between 0 and 1. Returns {group name: array [n]}.
    """
    model = run.model.eval()
    device = model.input_mean.device
    levels = {group_name(g): [] for g in model.param_groups}
    for i in range(n_samples(observations)):
        observation = select(observations, i)
        result = posterior(run, observation, prior_samples, batch_size=batch_size, min_ess=min_ess)
        summary = model.summary(apply(run.preprocess, _batched(observation, device)))
        for group in model.param_groups:
            truth = {p: torch.as_tensor(observation[p], device=device).float()[None] for p in group}
            log_r_truth = model.log_ratio(summary, truth, group)
            group_result = result[group_name(group)]
            levels[group_name(group)].append(
                float(group_result['weights'][group_result['log_ratio'] > log_r_truth].sum()))
    return {name: np.array(values) for name, values in levels.items()}


def uniform_prior_samples(ranges, n, *, seed=0):
    """Draw n parameter sets from a uniform prior {param: (low, high)}.

    Prior samples need only parameters, no simulations, so when the prior is known
    you can draw as many as you like: more samples give a smoother posterior.
    """
    generator = torch.Generator().manual_seed(seed)
    return {p: low + (high - low) * torch.rand(n, generator=generator) for p, (low, high) in ranges.items()}


def effective_sample_size(weights):
    """How many equally weighted samples the weighted samples are worth.

    Below a few hundred the posterior is noisy: use more prior samples.
    """
    return float(1 / (weights ** 2).sum())


def marginal(result, param, groups=None):
    """1D posterior of one parameter, pooled with equal weight from every group containing it.

    groups: optional list of group names (e.g. ['depth|width', 'depth']) to pool
        from; by default every group containing the parameter. Pooling mixes the
        groups' posteriors, so one poorly calibrated (e.g. too wide) group makes
        the pooled marginal worse: leave it out here.
    """
    groups = [g for name, g in result.items()
              if param in g['samples'] and (groups is None or name in groups)]
    if not groups:
        raise KeyError(f'{param!r} is in none of the selected parameter groups.')
    samples = torch.cat([g['samples'][param] for g in groups])
    weights = torch.cat([g['weights'] / len(groups) for g in groups])
    return samples, weights


def weighted_quantiles(samples, weights, quantiles=(0.16, 0.5, 0.84)):
    order = torch.argsort(samples)
    cumulative = torch.cumsum(weights[order], 0)
    cumulative = cumulative / cumulative[-1]
    index = torch.searchsorted(cumulative, torch.tensor(quantiles, device=samples.device, dtype=cumulative.dtype))
    return samples[order][index.clamp(max=len(samples) - 1)]


def summarise(result, groups=None):
    """{param: (16th, 50th, 84th percentile)}: median and 68% interval of each parameter.

    groups: optional list of group names to pool from (see marginal).
    """
    params = dict.fromkeys(p for name, g in result.items() if groups is None or name in groups
                           for p in g['samples'])
    return {p: tuple(weighted_quantiles(*marginal(result, p, groups)).tolist()) for p in params}


def coverage(run, observations, prior_samples, *, levels=(0.68, 0.95), groups=None):
    """Fraction of test simulations whose true value lies in the central credible interval.

    observations: a batch of (noisy) test simulations, including the true parameters.
    groups: optional list of group names to pool the marginals from (see marginal).
    For a well-calibrated network the fraction matches the level (68% of truths
    inside the 68% interval). Higher fractions mean the posteriors are too wide
    (conservative), lower ones that they are too narrow (overconfident).
    """
    params = [p for p in run.model.param_names
              if any(p in g and (groups is None or group_name(g) in groups) for g in run.model.param_groups)]
    inside = {p: {level: 0 for level in levels} for p in params}
    n = n_samples(observations)
    for i in range(n):
        observation = select(observations, i)
        result = posterior(run, observation, prior_samples)
        for p in params:
            samples, weights = marginal(result, p, groups)
            truth = float(observation[p])
            for level in levels:
                low, high = weighted_quantiles(samples, weights, ((1 - level) / 2, (1 + level) / 2)).tolist()
                inside[p][level] += low <= truth <= high
    return {p: {level: count / n for level, count in counts.items()} for p, counts in inside.items()}


def save_posterior(result, path):
    """Save a posterior result as .npz (keys 'group/param' and 'group/weights')."""
    arrays = {}
    for name, group in result.items():
        arrays[f'{name}/weights'] = group['weights'].cpu().numpy()
        for p, values in group['samples'].items():
            arrays[f'{name}/{p}'] = values.cpu().numpy()
    np.savez_compressed(path, **arrays)


def load_posterior(path):
    result = {}
    with np.load(path) as archive:
        for key in archive.files:
            name, field = key.split('/')
            group = result.setdefault(name, {'samples': {}, 'weights': None})
            if field == 'weights':
                group['weights'] = torch.as_tensor(archive[key])
            else:
                group['samples'][field] = torch.as_tensor(archive[key])
    return result
