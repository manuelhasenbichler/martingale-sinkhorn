from .solver import ExpectileNeuralMOT, NMOTPotentials
from ._utils import DataSampler, LossTracker

try:
    from .viz2d import plot_mtransport_validation_2d, animate_bass_martingale
except ModuleNotFoundError:
    # viz2d extras not installed
    plot_mtransport_validation_2d = None
    animate_bass_martingale = None

__all__ = [
    "ExpectileNeuralMOT",
    "NMOTPotentials",
    "DataSampler",
    "LossTracker",
    "plot_mtransport_validation_2d",
    "animate_bass_martingale",
]