"""Compatibility wrapper for the preferred public IR registry module."""

from edgelstm.ir.registry import (
    DuplicateOperatorError,
    OperatorRegistry,
    OperatorRegistryError,
    UnknownOperatorError,
    get_default_registry,
)

__all__ = [
    "DuplicateOperatorError",
    "OperatorRegistry",
    "OperatorRegistryError",
    "UnknownOperatorError",
    "get_default_registry",
]
