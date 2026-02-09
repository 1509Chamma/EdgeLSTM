class Tensor:
    """
    Represents a tensor in the intermediate representation (IR).
    Args:
        id (str): The unique identifier for the tensor.
        shape (list[int]): The shape of the tensor.
        dtype (str): The data type of the tensor.
    """
    def __init__(self, id: str, shape: list[int], dtype: str) -> None:
        self.id = id
        self.shape = shape
        self.dtype = dtype

    def to_dict(self) -> dict:
        return {
            "id": self.id,      
            "shape": self.shape,
            "dtype": self.dtype
        }