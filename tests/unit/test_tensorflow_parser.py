from unittest.mock import MagicMock, patch

from edgelstm.ir.graph import Graph
from edgelstm.parsers.tensorflow.parser import TensorFlowParser


def test_tensorflow_parser_init():
    parser = TensorFlowParser()
    assert parser.onnx_parser is not None


@patch("edgelstm.parsers.tensorflow.parser.convert.from_keras")
@patch("tensorflow.keras.Model")
@patch("os.path.exists")
def test_parse_model_calls_tf2onnx(mock_exists, mock_keras, mock_tf2onnx):
    # Setup mocks
    mock_exists.return_value = True
    mock_model = MagicMock(spec=mock_keras)

    parser = TensorFlowParser()
    # Mocked onnx parser
    parser.onnx_parser.parse = MagicMock(
        return_value=Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])
    )

    # Act
    graph = parser.parse_model(mock_model)

    # Assert
    assert isinstance(graph, Graph)
    mock_tf2onnx.assert_called_once()
