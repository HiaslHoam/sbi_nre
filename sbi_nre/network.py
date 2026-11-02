"""The neural network.

Neural ratio estimation trains a classifier to tell matching pairs (data,
parameters), which come from the same simulation, from mismatched pairs, where
the parameters belong to a different simulation. The classifier's output (logit)
then estimates

    log r(x, θ) = log p(θ | x) - log p(θ),

the log ratio of posterior to prior. Multiplying the prior by r gives the posterior.

The network has two parts:

* an embedding, shared by all parameter groups, that compresses the inputs
  (observation and context) into a short summary vector;
* one small classifier ("tail") per parameter group, which sees the summary
  and the group's parameters. A group of one parameter gives its 1D marginal
  posterior, a group of two parameters its 2D marginal.

Inputs and parameters are standardised (zero mean, unit variance) with
statistics computed once from the training set and stored in the network.
"""
import torch
from torch import nn


def mlp(n_in, hidden, n_out):
    """Fully connected network with ReLU activations."""
    layers = []
    for width in hidden:
        layers += [nn.Linear(n_in, width), nn.ReLU()]
        n_in = width
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


def group_name(group):
    """Name of a parameter group, e.g. ('a', 'b') -> 'a|b'."""
    return '|'.join(group)


class RatioEstimator(nn.Module):
    """Shared embedding plus one ratio classifier per parameter group.

    observation_sizes, context_sizes: {key: number of values per simulation}.
        Observations are the measured data (e.g. a spectrum); context is extra
        known information (e.g. the noise level). The network treats both the
        same way: they are concatenated into one input vector.
    param_groups: parameter groups, e.g. [('a', 'b'), ('c',)].
    """

    def __init__(self, observation_sizes, context_sizes, param_groups, *, summary_size=64,
                 hidden=(256, 256), tail_hidden=(128, 128)):
        super().__init__()
        input_sizes = {**observation_sizes, **context_sizes}
        self.settings = {'observation_sizes': dict(observation_sizes),
                         'context_sizes': dict(context_sizes),
                         'param_groups': [list(g) for g in param_groups],
                         'summary_size': summary_size, 'hidden': list(hidden),
                         'tail_hidden': list(tail_hidden)}
        self.input_keys = tuple(input_sizes)
        self.param_groups = tuple(tuple(g) for g in param_groups)
        self.param_names = tuple(dict.fromkeys(p for g in self.param_groups for p in g))
        n_inputs = sum(input_sizes.values())
        self.embedding = mlp(n_inputs, hidden, summary_size)
        self.tails = nn.ModuleDict({group_name(g): mlp(summary_size + len(g), tail_hidden, 1)
                                    for g in self.param_groups})
        self.register_buffer('input_mean', torch.zeros(n_inputs))
        self.register_buffer('input_std', torch.ones(n_inputs))
        self.register_buffer('param_mean', torch.zeros(len(self.param_names)))
        self.register_buffer('param_std', torch.ones(len(self.param_names)))

    def _raw_inputs(self, data):
        n = len(data[self.input_keys[0]])
        return torch.cat([data[key].reshape(n, -1).float() for key in self.input_keys], dim=1)

    def _raw_params(self, data, names):
        return torch.stack([data[name].float() for name in names], dim=1)

    @torch.no_grad()
    def fit_scaling(self, data):
        """Store the mean and spread of inputs and parameters of (training) data."""
        x = self._raw_inputs(data)
        theta = self._raw_params(data, self.param_names)
        self.input_mean.copy_(x.mean(0))
        self.input_std.copy_(x.std(0).clamp_min(1e-8))
        self.param_mean.copy_(theta.mean(0))
        self.param_std.copy_(theta.std(0).clamp_min(1e-8))

    def summary(self, data):
        """Embedding of the observation and context: [N, summary_size]."""
        return self.embedding((self._raw_inputs(data) - self.input_mean) / self.input_std)

    def log_ratio(self, summary, params, group):
        """log r for one parameter group; params is a dict holding the group's parameters."""
        index = [self.param_names.index(name) for name in group]
        theta = (self._raw_params(params, group) - self.param_mean[index]) / self.param_std[index]
        return self.tails[group_name(group)](torch.cat([summary, theta], dim=1)).squeeze(1)


def build_model(data, param_groups, observation, context=(), **network):
    """Create a RatioEstimator whose input sizes are read from example data.

    data: a (preprocessed) batch containing the observation and context keys.
    param_groups: e.g. [('a', 'b'), ('c',)].
    observation, context: lists of data keys given to the network.
    network: optional summary_size, hidden and tail_hidden.
    """
    missing = ({p for g in param_groups for p in g} | set(observation) | set(context)) - set(data)
    if missing:
        raise KeyError(f'Not in the data: {sorted(missing)}. Available: {sorted(data)}')
    sizes = lambda keys: {key: data[key][0].numel() for key in keys}
    return RatioEstimator(sizes(observation), sizes(context), param_groups, **network)
