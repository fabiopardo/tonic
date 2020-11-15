from .utils import merge_first_two_dims
from .utils import tile

from .actors import ClippedRatio  # noqa
from .actors import DeterministicPolicyGradient
from .actors import DistributionalDeterministicPolicyGradient
from .actors import MaximumAPosterioriPolicyOptimization
from .actors import StochasticPolicyGradient
from .actors import TrustRegionPolicyGradient
from .actors import TwinCriticSoftDeterministicPolicyGradient

from .critics import DeterministicQLearning
from .critics import DistributionalDeterministicQLearning
from .critics import ExpectedSARSA
from .critics import QRegression
from .critics import TargetActionNoise
from .critics import TwinCriticDeterministicQLearning
from .critics import TwinCriticSoftQLearning
from .critics import VRegression

from .optimizers import ConjugateGradient


__all__ = [
    merge_first_two_dims, tile, ClippedRatio, DeterministicPolicyGradient,
    DistributionalDeterministicPolicyGradient,
    MaximumAPosterioriPolicyOptimization, StochasticPolicyGradient,
    TrustRegionPolicyGradient, TwinCriticSoftDeterministicPolicyGradient,
    DeterministicQLearning, DistributionalDeterministicQLearning,
    ExpectedSARSA, QRegression, TargetActionNoise,
    TwinCriticDeterministicQLearning, TwinCriticSoftQLearning, VRegression,
    ConjugateGradient]
