"""Compatibility wrapper for the preferred public IR validation module."""

from edgelstm.ir.validation import (
    GraphValidationError,
    IRValidationError,
    OperatorValidationError,
    TopologyValidationError,
    ValueValidationError,
    validate_fpga_constraints,
    validate_graph,
    validate_ir,
    validate_operators,
    validate_topology,
    validate_values,
)

__all__ = [
    "GraphValidationError",
    "IRValidationError",
    "OperatorValidationError",
    "TopologyValidationError",
    "ValueValidationError",
    "validate_fpga_constraints",
    "validate_graph",
    "validate_ir",
    "validate_operators",
    "validate_topology",
    "validate_values",
]
