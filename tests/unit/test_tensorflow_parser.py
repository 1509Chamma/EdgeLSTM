import os
import pytest
from unittest.mock import MagicMock, patch

from edgelstm.parsers.tensorflow.parser import TensorFlowParser
from edgelstm.ir.graph import Graph

def test_tensorflow_parser_init():
    parser = TensorFlowParser()
    assert parser.onnx_parser is not None

@patch("tf2onnx.convert.from_keras")
@patch("os.path.exists")
def test_parse_model_calls_tf2onnx(mock_exists, mock_tf2onnx):
    # Setup mocks
    mock_exists.return_value = True
    mock_model = MagicMock()
    mock_model.__class__.__name__ = "Model"
    # Actually mock isinstance check
    with patch("tensorflow.keras.Model", MagicMock) as mock_keras:
        mock_keras.__instancecheck__.return_value = True
        
        parser = TensorFlowParser()
        # Mocked onnx parser
        parser.onnx_parser.parse = MagicMock(return_value=Graph())
        
        # Act
        graph = parser.parse_model(mock_model)
        
        # Assert
        assert isinstance(graph, Graph)
        mock_tf2onnx.assert_called_once()
