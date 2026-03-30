"""Compatibility wrapper for the preferred public IR operator module."""

from edgelstm.ir.op import (
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
