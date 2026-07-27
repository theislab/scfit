"""scfit.nn — paradigm-neutral torch building blocks (MLP, Resnet1d, AdaLNZero1d) + the activation resolver.

Pure torch, no jax and no flow/perturbation vocabulary — shared by every family (flow velocity fields,
foundation encoders, …). The ComponentRegistry spec-wrappers stay in their toolboxes for now.
"""

from scfit.nn._activation import ActivationId, resolve_activation
from scfit.nn._modules import MLP, AdaLNZero1d, FunctionalModule, Resnet1d

__all__ = [
    "FunctionalModule",
    "MLP",
    "Resnet1d",
    "AdaLNZero1d",
    "ActivationId",
    "resolve_activation",
]
