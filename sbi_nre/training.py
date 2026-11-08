"""Train a RatioEstimator and save the run.

Each training step takes a batch of simulations and builds

* matching pairs: data and the parameters that produced it (label 1);
* mismatched pairs: the same data with parameters shuffled across the batch (label 0).
  With n_negatives > 1, every observation is paired with that many differently
  shuffled parameter sets; their loss is averaged so that matching and mismatched
  pairs keep equal weight (the optimal classifier is still log r). Mismatched
  pairs cost no simulations, and many of them help groups with narrow posteriors
  (e.g. joint groups of many parameters), whose classifier otherwise rarely sees
  mismatched parameters close to the truth and learns posteriors that are too wide.

Every parameter group's classifier learns to tell them apart (binary
cross-entropy). The loss reported is the mean over groups; it starts near
2 ln 2 ≈ 1.39 (guessing) and decreases as the network learns.

Augmentation (e.g. noise) is applied afresh to every training batch, so the
network never sees exactly the same noisy spectrum twice. The validation set is
augmented once, with a fixed seed, so its loss is comparable between epochs.
"""
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import batches, n_samples, select, to_device
from .network import group_name
from .transforms import apply


def nre_loss(model, data, generator=None, n_negatives=1):
    """{group name: loss} for one batch of preprocessed data.

    n_negatives: number of mismatched parameter sets per observation.
    """
    summary = model.summary(data)
    n = len(summary)
    perm_device = generator.device if generator is not None else summary.device
    shuffled = torch.cat([torch.randperm(n, device=perm_device, generator=generator)
                          for _ in range(n_negatives)]).to(summary.device)
    repeated_summary = summary.repeat(n_negatives, 1)
    ones, zeros = torch.ones(n, device=summary.device), torch.zeros(n * n_negatives, device=summary.device)
    losses = {}
    for group in model.param_groups:
        matching = model.log_ratio(summary, data, group)
        mismatched = model.log_ratio(repeated_summary, {p: data[p][shuffled] for p in group}, group)
        losses[group_name(group)] = (F.binary_cross_entropy_with_logits(matching, ones)
                                     + F.binary_cross_entropy_with_logits(mismatched, zeros))
    return losses


def _write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2) + '\n')


def train(model, training, validation, run_dir, *, augment=(), preprocess=(), batch_size=256,
          learning_rate=1e-3, max_epochs=50, patience=10, seed=0, device='cpu',
          n_negatives=1, save_every_epoch=False, notes=''):
    """Train the model and save everything needed to reload it in ``run_dir``.

    training, validation: raw simulation grids (dictionaries of tensors).
    augment, preprocess: lists of ``step(...)`` (see transforms.py).
    patience: stop when the validation loss has not improved for this many epochs.
    n_negatives: mismatched parameter sets per observation (see nre_loss).
    save_every_epoch: also keep a checkpoint of every epoch in run_dir/epochs/.

    The run folder holds config.json, history.json, best.pt (lowest validation
    loss) and last.pt. Returns the history.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
    augment, preprocess = list(augment), list(preprocess)
    _write_json(run_dir / 'config.json', {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'notes': notes,
        'network': model.settings,
        'augment': augment,
        'preprocess': preprocess,
        'training': {'batch_size': batch_size, 'learning_rate': learning_rate,
                     'max_epochs': max_epochs, 'patience': patience, 'seed': seed,
                     'n_negatives': n_negatives},
        'data': {'n_training': n_samples(training), 'n_validation': n_samples(validation),
                 'keys': sorted(training)},
    })

    torch.manual_seed(seed)
    model.to(device)
    training, validation = to_device(training, device), to_device(validation, device)
    prepare = lambda data: apply(preprocess, apply(augment, data))
    validation = prepare(validation)
    # Standardise inputs and parameters with statistics from (up to) 50,000 simulations.
    sample = torch.randperm(n_samples(training), device=device)[:50_000]
    model.fit_scaling(prepare(select(training, sample)))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=max(1, patience // 3))
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'learning_rate': [], 'seconds': [],
               'val_loss_groups': {group_name(g): [] for g in model.param_groups}}
    best, epochs_since_best = float('inf'), 0
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        total = 0.
        for batch in batches(training, batch_size, shuffle=True):
            losses = nre_loss(model, prepare(batch), n_negatives=n_negatives)
            loss = sum(losses.values()) / len(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * n_samples(batch)
        train_loss = total / n_samples(training)

        model.eval()
        group_totals = dict.fromkeys(history['val_loss_groups'], 0.)
        # Same shuffles every epoch. On a CPU generator except for CUDA (not every PyTorch supports MPS generators).
        generator = torch.Generator(device=device if torch.device(device).type == 'cuda' else 'cpu').manual_seed(seed + 1)
        with torch.no_grad():
            for batch in batches(validation, batch_size):
                for name, value in nre_loss(model, batch, generator, n_negatives).items():
                    group_totals[name] += value.item() * n_samples(batch)
        group_losses = {name: value / n_samples(validation) for name, value in group_totals.items()}
        val_loss = sum(group_losses.values()) / len(group_losses)
        scheduler.step(val_loss)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['seconds'].append(time.time() - start)
        for name, value in group_losses.items():
            history['val_loss_groups'][name].append(value)

        torch.save(model.state_dict(), run_dir / 'last.pt')
        if save_every_epoch:
            (run_dir / 'epochs').mkdir(exist_ok=True)
            torch.save(model.state_dict(), run_dir / 'epochs' / f'epoch_{epoch:03d}.pt')
        improved = val_loss < best
        if improved:
            best, epochs_since_best = val_loss, 0
            torch.save(model.state_dict(), run_dir / 'best.pt')
        else:
            epochs_since_best += 1
        _write_json(run_dir / 'history.json', history)
        print(f'epoch {epoch:3d}  train {train_loss:.4f}  validation {val_loss:.4f}'
              f'{"  *best" if improved else ""}  ({history["seconds"][-1]:.1f} s)', flush=True)
        if epochs_since_best >= patience:
            print(f'Stopping: no improvement for {patience} epochs.', flush=True)
            break
    return history
