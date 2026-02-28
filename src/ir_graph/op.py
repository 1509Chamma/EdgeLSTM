from typing import List, Dict, Optional

class Op:
    """
    Represents an operation (node) in the intermediate representation (IR) graph.
    Args:
        op_id (str): The unique identifier for the operation.
        op_type (str): The type of the operation (e.g., "Conv2D", "ReLU").
        inputs (List[str]): A list of input value IDs for this operation.
        outputs (List[str]): A list of output value IDs produced by this operation.
        attrs (Optional[Dict[str, object]]): A dictionary of attributes for the operation.
        name (Optional[str]): An optional human-readable name for the operation.
        source_span (Optional[str]): An optional source code span for debugging purposes.
    """
    def __init__(
        self,
        op_id: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attrs: Optional[Dict[str, object]] = None,
        name: Optional[str] = None,
        source_span: Optional[str] = None,
    ) -> None:
        self.op_id = op_id
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs if attrs is not None else {}
        self.name = name
        self.source_span = source_span

    def to_dict(self) -> Dict:
        """
        Convert the Op to a JSON-serializable dictionary.
        """
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
            "name": self.name,
            "source_span": self.source_span,
        }
