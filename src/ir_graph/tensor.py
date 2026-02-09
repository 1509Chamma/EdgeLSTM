class Tensor:
    """
    Represents a tensor in the intermediate representation (IR).
    Args:
        tensor_id (str): The unique identifier for the tensor.
        shape (list[int]): The shape of the tensor.
        dtype (str): The data type of the tensor.
    """
    def __init__(self, tensor_id: str, shape: list[int], dtype: str) -> None:
        self.tensor_id = tensor_id
        self.shape = shape
        self.dtype = dtype

    def to_dict(self) -> dict:
        return {
            "tensor_id": self.tensor_id,
            "shape": self.shape,
            "dtype": self.dtype
        }
