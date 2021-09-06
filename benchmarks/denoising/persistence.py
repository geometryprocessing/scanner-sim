from pathlib import Path
from typing import Optional
import torch

class Checkpoint:
    def __init__(self, data: dict = {}):
        self.data = data

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, path)

    @classmethod
    def load(cls, path: Path, **kwargs):
        return cls(torch.load(path, **kwargs))

    @staticmethod
    def get_path(checkpoint_directory: Path, epoch: Optional[int] = None) -> Path:
        if epoch is None:
            return checkpoint_directory / 'last.ckpt'
        else:
            return checkpoint_directory  / f'{epoch}.ckpt'

    def get_path_best(checkpoint_directory: Path, epoch: Optional[int] = None) -> Path:
        if epoch is None:
            return checkpoint_directory / 'best.ckpt'
        else:
            return checkpoint_directory  / f'best_{epoch}.ckpt'

    def store_config(self, config: dict):
        self.data['config'] = config

    def store_model(self, model: torch.nn.Module):
        if isinstance(model, torch.nn.DataParallel):
            self.data['state_dict'] = model.module.state_dict()
        else:
            self.data['state_dict'] = model.state_dict()

    def store_optimizer(self, optimizer):
        self.data['optimizer_state_dict'] = optimizer.state_dict()

    def store_best_val_loss(self, loss):
        if loss is not None:
            self.data['best_val_loss'] = float(loss)

    def restore_config(self, config: dict):
        config.update(self.data['config'])

    def restore_model(self, model: torch.nn.Module):
        if isinstance(model, torch.nn.DataParallel):
            raise ValueError("Please restore the model before it is wrapped with DataParallel")
        else:
            model.load_state_dict(self.data['state_dict'])

    def restore_optimizer(self, optimizer):
        optimizer.load_state_dict(self.data['optimizer_state_dict'])

    def restore_best_val_loss(self):
        return self.data.get('best_val_loss', None)