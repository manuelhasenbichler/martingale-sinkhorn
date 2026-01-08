from typing import Callable, List, Tuple, Dict

import dataclasses
import jax
import jax.numpy as jnp



@dataclasses.dataclass
class DataSampler:
    data: jnp.ndarray
    batch_size: int
    key: jax.Array

    def __iter__(self):
        return self
    
    def _get_key(self):
        self.key, sub = jax.random.split(self.key)
        return sub
    
    def apply_fn(self, x, fn):
        self.data = fn(x)

    def __next__(self):
        n = self.data.shape[0]
        idx = jax.random.randint(self._get_key(), (self.batch_size,), 0, n)
        return self.data[idx]


class LossTracker:
    def __init__(self, monitor: List[str] = {"loss": None, "valid_loss": None}, alpha: Dict[str, float | None] = {"loss": 0.1, "valid_loss": 0.3}):
        self.alpha = alpha
        self.monitor = monitor
    
    @staticmethod
    def EMA(old: float | None, new: float, alpha):
        if alpha is None:
            return new
        if old is None:
            return new
        return alpha * new + (1 - alpha) * old
    def update(self, value: float, key: str) -> None:
        if key not in self.monitor:
            raise ValueError(f"Loss '{key}' not tracked. Available losses: {list(self.monitor.keys())}")
        self.monitor[key] = self.EMA(self.monitor[key], value, self.alpha[key])
    def reset(self) -> None:
        self.monitor = {k: None for k in self.monitor}