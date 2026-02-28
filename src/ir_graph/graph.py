from typing import Dict, List, Optional
from .value import Value
from .op import Op

class Graph:
    """
    Represents the entire intermediate representation (IR) graph.
    Args:  
        values (Dict[str, Value]): A dictionary mapping value IDs to Value objects
        ops (Dict[str, Op]): A dictionary mapping operation IDs to Op objects
        graph_inputs (List[str]): A list of value IDs that are the inputs to the graph
        graph_outputs (List[str]): A list of value IDs that are the outputs of the graph
        states (Optional[Dict[str, Value]]): An optional dictionary of state values
    """
    def __init__(
        self,
        values: Dict[str, Value],
        ops: Dict[str, Op],
        graph_inputs: List[str],
        graph_outputs: List[str],
        states: Optional[Dict[str, Value]] = None,
    ) -> None:
        self.values = values
        self.ops = ops
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.states = states if states is not None else {}

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
