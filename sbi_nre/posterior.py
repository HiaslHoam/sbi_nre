"""Posteriors from a trained network.

The network estimates r(x, θ) = p(θ | x) / p(θ). Given samples from the prior
(e.g. the parameters of held-out simulations), weighting each sample by r turns
them into weighted samples of the posterior. No MCMC is needed.

A posterior result is a dictionary {group name: {'samples': {param: tensor},
'weights': tensor}}, one entry per parameter group, with weights summing to one.
"""
import numpy as np
import torch
from .data import n_samples, select
from .network import group_name
from .transforms import apply


def _batched(observation, device):
    """Add a batch dimension of one to a single observation."""
    return {key: torch.as_tensor(value, device=device).float()[None] for key, value in observation.items()}


@torch.no_grad()
def posterior(run, observation, prior_samples, *, batch_size=10_000):
    """Weighted posterior samples for one observation, for every parameter group.

    observation: a single example, e.g. {'obs': [100], 'sigma': scalar}. It is
        preprocessed exactly like the training data (augmentation is not applied).
    prior_samples: {param: [M]} drawn from the training prior.
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
        result[group_name(group)] = {'samples': samples, 'weights': torch.softmax(log_r, 0)}
    return result


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


def marginal(result, param):
    """1D posterior of one parameter, pooled with equal weight from every group containing it."""
    groups = [g for g in result.values() if param in g['samples']]
    if not groups:
        raise KeyError(f'{param!r} is in no parameter group.')
    samples = torch.cat([g['samples'][param] for g in groups])
    weights = torch.cat([g['weights'] / len(groups) for g in groups])
    return samples, weights


def weighted_quantiles(samples, weights, quantiles=(0.16, 0.5, 0.84)):
    order = torch.argsort(samples)
    cumulative = torch.cumsum(weights[order], 0)
    cumulative = cumulative / cumulative[-1]
    index = torch.searchsorted(cumulative, torch.tensor(quantiles, device=samples.device, dtype=cumulative.dtype))
    return samples[order][index.clamp(max=len(samples) - 1)]


def summarise(result):
    """{param: (16th, 50th, 84th percentile)}: median and 68% interval of each parameter."""
    params = dict.fromkeys(p for g in result.values() for p in g['samples'])
    return {p: tuple(weighted_quantiles(*marginal(result, p)).tolist()) for p in params}


def coverage(run, observations, prior_samples, *, levels=(0.68, 0.95)):
    """Fraction of test simulations whose true value lies in the central credible interval.

    observations: a batch of (noisy) test simulations, including the true parameters.
    For a well-calibrated network the fraction matches the level (68% of truths
    inside the 68% interval). Higher fractions mean the posteriors are too wide
    (conservative), lower ones that they are too narrow (overconfident).
    """
    params = run.model.param_names
    inside = {p: {level: 0 for level in levels} for p in params}
    n = n_samples(observations)
    for i in range(n):
        observation = select(observations, i)
        result = posterior(run, observation, prior_samples)
        for p in params:
            samples, weights = marginal(result, p)
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
