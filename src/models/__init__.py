"""Models module for HSI few-shot learning."""

from .backbone import HSI3DCNN
from .protonet import PrototypicalNetwork, build_prototypical_network

__all__ = [
    "HSI3DCNN",
    "PrototypicalNetwork",
    "build_prototypical_network",
]
