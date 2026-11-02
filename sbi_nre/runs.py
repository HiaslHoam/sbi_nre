"""Load a saved training run."""
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from .network import RatioEstimator


@dataclass
class Run:
    """A trained network together with its settings and training history."""
    path: Path
    model: RatioEstimator
    config: dict
    history: dict

    @property
    def preprocess(self):
        return self.config['preprocess']

    @property
    def augment(self):
        return self.config['augment']


def load_run(run_dir, checkpoint='best', *, device='cpu'):
    """Load a run folder written by ``train``.

    checkpoint: 'best' (lowest validation loss), 'last', or a path to another
    checkpoint such as run_dir/epochs/epoch_012.pt.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    history_path = run_dir / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.is_file() else {}
    model = RatioEstimator(**config['network'])
    path = run_dir / f'{checkpoint}.pt' if checkpoint in ('best', 'last') else Path(checkpoint)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return Run(run_dir, model.to(device).eval(), config, history)
