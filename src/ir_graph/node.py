from typing import Optional

class Node:
    """
    Represents a node in the intermediate representation (IR).
    Args:
        id (str): The unique identifier for the node.
        op_type (str): The type of operation represented by the node (e.g., "Add", "MatMul").
        inputs (list[str]): A list of input tensor identifiers for the node.
        outputs (list[str]): A list of output tensor identifiers produced by the node.
        attributes (Optional[dict[str, object]]): An optional dictionary of additional attributes for the node
    """
    def __init__(
        self,
        id: str,
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        attributes: Optional[dict[str, object]] = None
    ) -> None:
        self.id = id
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes if attributes is not None else {}
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes
        }
