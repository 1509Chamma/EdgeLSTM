from __future__ import annotations

import inspect
from typing import Dict, TypeVar

from .op import InvalidOperatorDefinitionError, Operator, OperatorError

OperatorT = TypeVar("OperatorT", bound=Operator)


class OperatorRegistryError(OperatorError):
    """Base exception for registry failures."""


class DuplicateOperatorError(OperatorRegistryError):
    """Raised when an operator type is registered more than once."""


class UnknownOperatorError(OperatorRegistryError):
    """Raised when an operator type is requested but not registered."""


class OperatorRegistry:
    """Runtime registry for operator classes and instance construction."""

    def __init__(self) -> None:
        self._operators: Dict[str, type[Operator]] = {}

    def register(self, operator_cls: type[OperatorT]) -> type[OperatorT]:
        if not inspect.isclass(operator_cls) or not issubclass(operator_cls, Operator):
            raise InvalidOperatorDefinitionError(
                "operator_cls must be a concrete Operator subclass"
            )
        if inspect.isabstract(operator_cls):
            raise InvalidOperatorDefinitionError(
                f"{operator_cls.__name__} must be concrete before registration"
            )

        op_type = operator_cls.operator_type()
        if op_type in self._operators:
            raise DuplicateOperatorError(
                f"operator type '{op_type}' is already registered"
            )

        self._operators[op_type] = operator_cls
        return operator_cls

    def get(self, op_type: str) -> type[Operator]:
        try:
            return self._operators[op_type]
        except KeyError as exc:
            raise UnknownOperatorError(
                f"operator type '{op_type}' is not registered"
            ) from exc

    def create(self, op_type: str, **node_kwargs: object) -> Operator:
        operator_cls = self.get(op_type)
        return operator_cls(**node_kwargs)

    def list_registered(self) -> list[str]:
        return sorted(self._operators)


default_registry = OperatorRegistry()
