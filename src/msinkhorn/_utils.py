from typing import Callable, Optional, Union, Sequence, List, Dict, Optional

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


AxisSpec = Union[int, None, Sequence[Optional[int]]]
def microbatch(fn: Callable[..., jnp.ndarray], batch_size: int, in_axes: Optional[AxisSpec] = 0) -> Callable[..., jnp.ndarray]:
    """
    Wraps a function to process batched inputs in micro-batches of size batch_size.
    NOTE: For every positional argument of fn, with in_axis[i]=ax (ax is not None), arg must have a batch dimension of size n at axis ax. 
    NOTE: The wrapped function fn must return an array whose leading axis (axis 0) is the batch dimension, with size equal to batch_size when called on a chunk.

    Args:
        fn: The function to be wrapped.
        batch_size: The size of each micro-batch.
        in_axes: Specifies which axes of the input arguments to split into micro-batches. Can be an int, None, or a sequence of ints/None matching the number of positional arguments.
    Returns:
        A wrapped function that processes inputs in micro-batches.
    """    
    def pad_arg(arg, ax, pad_len):
        """Pads the argument arg along axis ax with pad_len zeros."""
        if ax is None or pad_len == 0:
            return arg
        pad_cfg = [(0, 0)] * arg.ndim
        pad_cfg[ax] = (0, pad_len)
        return jnp.pad(arg, pad_cfg)

    def get_chunk(i, args_tuple, ax_list):
        """Return the i-th chunk of each arg according to ax_list."""
        chunk_args = []
        start = i * batch_size

        for a, ax in zip(args_tuple, ax_list):
            if ax is None:
                chunk_args.append(a)
            else:
                chunk = jax.lax.dynamic_slice_in_dim(a, start, batch_size, axis=ax)
                chunk_args.append(chunk)

        return tuple(chunk_args)

    def wrapped(*args, **kwargs):
        """
        Processes inputs in micro-batches along specified axes, and concatenates the results.
        NOTE: For every positional argument of fn, with in_axis[i]=ax (ax is not None), arg must have a batch dimension of size n at axis ax.

        Args:
            *args: Positional arguments to fn.
            **kwargs: Keyword arguments to fn.
        Returns:
            Concatenated output from processing each micro-batch.
        """
        if isinstance(in_axes, (int, type(None))):
            ax_list = [in_axes] + [None] * (len(args) - 1)
        else:
            ax_list = list(in_axes)
            if len(ax_list) != len(args):
                raise ValueError("len(in_axes) must match number of positional args")

        n = None
        for a, ax in zip(args, ax_list):
            if ax is not None:
                n = a.shape[ax]
                break
        if n is None:
            return fn(*args, **kwargs)

        if batch_size >= n:
            return fn(*args, **kwargs)

        n_chunks = (n + batch_size - 1) // batch_size
        pad_len = n_chunks * batch_size - n
        padded_args = tuple(pad_arg(a, ax, pad_len) for a, ax in zip(args, ax_list))

        chunk0_args = get_chunk(0, padded_args, ax_list)
        y0 = fn(*chunk0_args, **kwargs)
        y_shape = y0.shape
        if y_shape[0] != batch_size:
            raise ValueError(f"fn must return batch along axis 0, got shape {y_shape}")

        out_init = jnp.zeros((n_chunks, batch_size) + y_shape[1:], dtype=y0.dtype)

        def body(i, out):
            chunk_args = get_chunk(i, padded_args, ax_list)
            y_chunk = fn(*chunk_args, **kwargs)
            return out.at[i].set(y_chunk)

        out = jax.lax.fori_loop(0, n_chunks, body, out_init)
        out_flat = out.reshape(n_chunks * batch_size, *y_shape[1:])
        return out_flat[:n]

    return wrapped