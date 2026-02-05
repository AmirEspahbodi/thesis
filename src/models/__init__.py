"""Models module for HSI few-shot learning."""

from .backbone import HSI3DCNN, Conv3DBlock
from .protonet import PrototypicalNetwork, build_prototypical_network

__all__ = [
    'HSI3DCNN',
    'Conv3DBlock',
    'PrototypicalNetwork',
    'build_prototypical_network'
]
