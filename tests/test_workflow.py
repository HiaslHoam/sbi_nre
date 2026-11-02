"""Quick end-to-end checks on a tiny toy grid (CPU, a few seconds)."""
import torch
from sbi_nre import (apply, build_model, coverage, load_posterior, load_run, posterior, save_posterior,
                     select, split, step, summarise, train, transform)
from sbi_nre.toy import simulate_toy

AUGMENT = [step('add_noise', key='signal', sigma_range=[0.01, 0.05])]
PREPROCESS = [step('normalise_minmax', key='obs')]
GROUPS = [('depth', 'width'), ('slope',)]
INPUTS = dict(observation=['obs_norm', 'obs_min', 'obs_max'], context=['sigma'])


def trained_run(tmp_path):
    training, validation, test = split(simulate_toy(2000), seed=0)
    example = apply(PREPROCESS, apply(AUGMENT, training))
    model = build_model(example, GROUPS, **INPUTS, summary_size=16, hidden=[32], tail_hidden=[32])
    history = train(model, training, validation, tmp_path / 'run', augment=AUGMENT,
                    preprocess=PREPROCESS, max_epochs=2, batch_size=128)
    return history, validation, test


def test_train_save_load_posterior(tmp_path):
    history, validation, test = trained_run(tmp_path)
    assert len(history['val_loss']) == 2
    run = load_run(tmp_path / 'run')
    observation = select(apply(AUGMENT, select(test, slice(0, 1))), 0)
    result = posterior(run, observation, validation)
    assert set(result) == {'depth|width', 'slope'}
    for group in result.values():
        assert torch.isclose(group['weights'].sum(), torch.tensor(1.))
    assert set(summarise(result)) == {'depth', 'width', 'slope'}
    fractions = coverage(run, apply(AUGMENT, select(test, slice(0, 5))), validation)
    assert all(0 <= f <= 1 for counts in fractions.values() for f in counts.values())
    save_posterior(result, tmp_path / 'post.npz')
    loaded = load_posterior(tmp_path / 'post.npz')
    assert torch.allclose(loaded['slope']['weights'], result['slope']['weights'])


def test_run_folder_restores_settings(tmp_path):
    trained_run(tmp_path)
    run = load_run(tmp_path / 'run', 'last')
    assert run.augment == AUGMENT and run.preprocess == PREPROCESS
    assert run.model.param_groups == tuple(GROUPS)
    assert (tmp_path / 'run' / 'best.pt').is_file() and len(run.history['epoch']) == 2


def test_custom_transform_and_split_sizes():
    @transform
    def double(data, *, key):
        return {**data, key: 2 * data[key]}
    grid = simulate_toy(100)
    assert torch.allclose(apply([step('double', key='signal')], grid)['signal'], 2 * grid['signal'])
    assert [len(part['depth']) for part in split(grid, (0.5, 0.3, 0.2))] == [50, 30, 20]
