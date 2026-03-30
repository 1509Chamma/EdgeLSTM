"""Compatibility wrapper for the older public IR operator module."""

from edge_lstm.ir.op import (
    FPGACost,
    InvalidOperatorDefinitionError,
    InvalidOperatorInstanceError,
    Operator,
    OperatorError,
)

__all__ = [
    "FPGACost",
    "InvalidOperatorDefinitionError",
    "InvalidOperatorInstanceError",
    "Operator",
    "OperatorError",
]
