from __future__ import annotations

from typing import Dict, List, Optional

from .op import Operator
from .registry import OperatorRegistry, default_registry
from .value import Value


class Graph:
    """
    Represents the entire intermediate representation (IR) graph.

    Args:
        values (Dict[str, Value]): A dictionary mapping value IDs to Value objects
        ops (Dict[str, Operator]): A dictionary mapping operation IDs to Operator
            objects
        graph_inputs (List[str]): A list of value IDs that are the inputs to the graph
        graph_outputs (List[str]): A list of value IDs that are the outputs of the graph
        states (Optional[Dict[str, Value]]): An optional dictionary of state values
        registry (Optional[OperatorRegistry]): Registry used to construct operators
            from op_type names
    """

    def __init__(
        self,
        values: Dict[str, Value],
        ops: Dict[str, Operator],
        graph_inputs: List[str],
        graph_outputs: List[str],
        states: Optional[Dict[str, Value]] = None,
        registry: Optional[OperatorRegistry] = None,
    ) -> None:
        self.values = dict(values)
        self.ops: Dict[str, Operator] = {}
        self.graph_inputs = list(graph_inputs)
        self.graph_outputs = list(graph_outputs)
        self.states = dict(states) if states is not None else {}
        self.registry = registry if registry is not None else default_registry

        for op_id, operator in ops.items():
            self._store_operator(op_id, operator)

    def add_operator(self, operator: Operator) -> Operator:
        """Insert an already-instantiated operator into the graph."""

        self._store_operator(operator.op_id, operator)
        return operator

    def create_operator(self, op_type: str, **node_kwargs: object) -> Operator:
        """
        Construct an operator from the graph registry and store it in the graph.
        """

        operator = self.registry.create(op_type, **node_kwargs)
        return self.add_operator(operator)

    def to_dict(self) -> Dict:
        """
        Convert the entire graph to a JSON-serializable dictionary.
        """
        return {
            "values": {vid: v.to_dict() for vid, v in self.values.items()},
            "ops": {oid: o.to_dict() for oid, o in self.ops.items()},
            "graph_inputs": self.graph_inputs,
            "graph_outputs": self.graph_outputs,
            "states": {sid: s.to_dict() for sid, s in self.states.items()},
        }

    @staticmethod
    def _validate_operator(op_id: str, operator: Operator) -> None:
        if not isinstance(operator, Operator):
            raise TypeError("ops must contain Operator instances")
        if operator.op_id != op_id:
            raise ValueError(
                f"operator key '{op_id}' does not match operator.op_id '{operator.op_id}'"
            )

    def _store_operator(self, op_id: str, operator: Operator) -> None:
        self._validate_operator(op_id, operator)
        self.ops[op_id] = operator
