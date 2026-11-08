"""Plots for training curves and posteriors."""
import math
import matplotlib.pyplot as plt
import numpy as np
from .posterior import marginal, weighted_quantiles


def _zoom(samples, weights, pad=0.3):
    """Plot range around the bulk of a weighted 1D distribution."""
    low, high = weighted_quantiles(samples, weights, (0.001, 0.999)).tolist()
    span = (high - low) or abs(high) or 1.
    return low - pad * span, high + pad * span


def plot_history(history, *, groups=True):
    """Training and validation loss per epoch; optionally the validation loss of every group."""
    fig, axes = plt.subplots(1, 2 if groups else 1, figsize=(11 if groups else 5.5, 4), squeeze=False)
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['train_loss'], label='training')
    ax.plot(history['epoch'], history['val_loss'], label='validation')
    best = int(np.argmin(history['val_loss']))
    ax.plot(history['epoch'][best], history['val_loss'][best], 'ko', label='best')
    ax.set(xlabel='epoch', ylabel='loss (mean over groups)')
    ax.legend()
    if groups:
        ax = axes[0, 1]
        for name, values in history['val_loss_groups'].items():
            ax.plot(history['epoch'], values, label=name)
        ax.set(xlabel='epoch', ylabel='validation loss')
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_posterior(result, truths=None, *, bins=40, ncols=4, zoom=True):
    """One panel per parameter group: a weighted histogram (1D) or 2D histogram (2D).

    zoom=True limits the axes to the region where the posterior lies; False shows
    the full range of the prior samples.
    """
    names = list(result)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, min(ncols, len(names)), figsize=(3.5 * min(ncols, len(names)), 3.2 * nrows),
                             squeeze=False)
    truths = truths or {}
    for ax, name in zip(axes.flat, names):
        weights_t = result[name]['weights']
        ranges = {p: _zoom(v, weights_t) if zoom else (float(v.min()), float(v.max()))
                  for p, v in result[name]['samples'].items()}
        samples = {p: v.cpu().numpy() for p, v in result[name]['samples'].items()}
        weights = weights_t.cpu().numpy()
        params = list(samples)
        if len(params) == 1:
            p = params[0]
            ax.hist(samples[p], bins=bins, range=ranges[p], weights=weights, color='tomato', alpha=.7)
            if p in truths:
                ax.axvline(float(truths[p]), color='k', ls='--')
            ax.set(xlabel=p, yticks=[])
        else:
            x, y = params[:2]
            ax.hist2d(samples[x], samples[y], bins=bins, range=[ranges[x], ranges[y]],
                      weights=weights, cmap='Reds')
            if x in truths and y in truths:
                ax.plot(float(truths[x]), float(truths[y]), 'k+', ms=14, mew=2)
            ax.set(xlabel=x, ylabel=y)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_marginals(result, truths=None, *, bins=40, zoom=True, groups=None):
    """1D posterior of every parameter, pooled over the groups that contain it.

    groups: optional list of group names to pool from (see posterior.marginal).
    """
    params = list(dict.fromkeys(p for name, g in result.items() if groups is None or name in groups
                                for p in g['samples']))
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3), squeeze=False)
    for ax, p in zip(axes.flat, params):
        samples, weights = marginal(result, p, groups)
        span = _zoom(samples, weights) if zoom else (float(samples.min()), float(samples.max()))
        ax.hist(samples.cpu().numpy(), bins=bins, range=span, weights=weights.cpu().numpy(),
                color='tomato', alpha=.7)
        if truths and p in truths:
            ax.axvline(float(truths[p]), color='k', ls='--', label='truth')
        ax.set(xlabel=p, yticks=[])
    fig.tight_layout()
    return fig


def plot_coverage(levels, *, ncols=4):
    """Expected coverage: for every credibility level, the fraction of test simulations
    whose true parameters lie inside the credible region of that level.

    levels: {group name: array} from credibility_of_truth. A calibrated network
    follows the diagonal (within the grey band, the scatter expected from a finite
    test set). Above it the posteriors are too wide (conservative), below it too
    narrow (overconfident).
    """
    names = list(levels)
    ncols = min(ncols, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    alpha = np.linspace(0, 1, 101)
    for ax, name in zip(axes.flat, names):
        values = np.asarray(levels[name])
        band = 1.96 * np.sqrt(alpha * (1 - alpha) / len(values))
        ax.fill_between(alpha, alpha - band, alpha + band, color='grey', alpha=.3, lw=0)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.plot(alpha, (values[:, None] <= alpha).mean(0), color='tomato', lw=2)
        ax.text(.05, .9, 'conservative', fontsize=8, color='grey')
        ax.text(.55, .05, 'overconfident', fontsize=8, color='grey')
        ax.set(xlim=(0, 1), ylim=(0, 1), title=name, aspect='equal',
               xlabel='credibility level', ylabel='fraction of truths inside')
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_true_vs_median(truths, medians, *, errors=None):
    """True value against posterior median for many test simulations, one panel per parameter.

    truths, medians: {param: array}; errors: optional {param: (lower, upper) arrays}.
    Points on the diagonal are perfect recoveries.
    """
    params = list(medians)
    fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3.4), squeeze=False)
    for ax, p in zip(axes.flat, params):
        t, m = np.asarray(truths[p]), np.asarray(medians[p])
        if errors:
            low, high = errors[p]
            ax.errorbar(t, m, yerr=[m - np.asarray(low), np.asarray(high) - m], fmt='none',
                        ecolor='tomato', alpha=.2)
        ax.plot(t, m, '.', color='tomato', ms=4)
        lims = [min(t.min(), m.min()), max(t.max(), m.max())]
        ax.plot(lims, lims, 'k--', lw=1)
        r2 = 1 - np.sum((m - t) ** 2) / np.sum((t - t.mean()) ** 2)
        ax.set(xlabel=f'true {p}', ylabel=f'posterior median', title=f'R² = {r2:.2f}')
    fig.tight_layout()
    return fig
