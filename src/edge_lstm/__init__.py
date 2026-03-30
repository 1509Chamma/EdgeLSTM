"""
Public compatibility namespace for EdgeLSTM.

The repository currently keeps its implementation under ``src/edgelstm`` on
disk, but consumers should import from package namespaces rather than from the
filesystem layout. New public-facing aliases can live under ``edge_lstm`` while
the existing ``edgelstm`` package remains supported internally.
"""

from edgelstm import (
    IO,
    Capabilities,
    DeviceRegistry,
    FPGADevice,
    Memory,
    Policies,
    Resources,
    __version__,
)

__all__ = [
    "FPGADevice",
    "Resources",
    "Memory",
    "IO",
    "Capabilities",
    "Policies",
    "DeviceRegistry",
    "__version__",
]
