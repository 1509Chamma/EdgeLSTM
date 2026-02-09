from ir_graph.tensor import Tensor
from ir_graph.node import Node

class Graph:
    """
    Representing a computational graph, consisting of nodes and tensors.
    Args:
        nodes (list[Node]): A list of nodes in the graph.
        tensors (dict[str, Tensor]): A dictionary mapping tensor IDs to Tensor objects.
        input_tensors (list[str]): A list of input tensor IDs.
        output_tensors (list[str]): A list of output tensor IDs.
    """
    def __init__(
        self,
        nodes: list[Node],
        tensors: dict[str, Tensor],
        input_tensors: list[str],
        output_tensors: list[str]
    ) -> None:
        self.nodes = nodes
        self.tensors = tensors
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
    
    def to_dict(self) -> dict:
        return {
            "input_tensors": self.input_tensors,
            "output_tensors": self.output_tensors,
            "nodes": [node.to_dict() for node in self.nodes],
            "tensors": {
                tensor_id: tensor.to_dict()
                for tensor_id, tensor in self.tensors.items()
            }
        }