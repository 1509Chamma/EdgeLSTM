from __future__ import annotations

import torch
import torch.nn as nn

from edge_lstm.ir.graph import Graph
from edge_lstm.ir.op import FPGACost, Operator
from edge_lstm.ir.value import Value, ValueType
from edge_lstm.numerical_parity import (
    TorchQuantizedModelSimulator,
    run_numerical_parity_test,
)
from edge_lstm.quantization_config import (
    FixedPointSpec,
    QuantizationScheme,
    QuantizationSpec,
)


class MockOp(Operator):
    OP_TYPE = "Mock"

    def validate(self, values):
        del values

    def estimate_fpga_cost(self, values):
        del values
        return FPGACost(1)

    def hls_template_path(self):
        return ""

    def hls_context(self, values):
        del values
        return {}


class TinyParityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)

        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor([[0.5, -0.25], [0.75, 0.5], [-0.5, 0.125]])
            )
            self.fc1.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
            self.fc2.weight.copy_(torch.tensor([[0.25, -0.75, 0.5]]))
            self.fc2.bias.copy_(torch.tensor([0.125]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ConstantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        return self.identity(torch.zeros(1, dtype=torch.float32))


class NonFiniteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        values = torch.tensor([float("nan"), float("inf")], dtype=torch.float32)
        return self.identity(values)


def _fixed_point_spec(integer_bits: int, fractional_bits: int) -> QuantizationSpec:
    return QuantizationSpec(
        bit_width=integer_bits + fractional_bits,
        scheme=QuantizationScheme.SYMMETRIC,
        fixed_point=FixedPointSpec(
            integer_bits=integer_bits,
            fractional_bits=fractional_bits,
        ),
    )


def _make_graph(output_shape: list[int]) -> Graph:
    input_value = Value("input", ValueType.TENSOR, "float32", [1, 2], ["N", "C"])
    output_value = Value(
        "output",
        ValueType.TENSOR,
        "float32",
        output_shape,
        ["N", "C"] if len(output_shape) == 2 else ["N"],
        producer_op_id="op0",
    )
    op = MockOp("op0", ["input"], ["output"])
    return Graph(
        {"input": input_value, "output": output_value},
        {"op0": op},
        ["input"],
        ["output"],
    )


def test_numerical_parity_identical_models_have_zero_error() -> None:
    model = TinyParityModel().eval()

    def dataset():
        yield from (
            torch.tensor([0.2, -0.1], dtype=torch.float32),
            torch.tensor([1.0, 0.5], dtype=torch.float32),
            torch.tensor([-0.75, 0.25], dtype=torch.float32),
        )

    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=model,
        dataset=dataset(),
        config={
            "metrics": ["mae", "max_error", "relative_error"],
            "thresholds": {
                "mae": 0.0,
                "max_error": 0.0,
                "relative_error": 0.0,
            },
        },
    )

    assert result["pass"] is True
    assert result["violations"] == []
    assert result["metrics"]["global"]["mae"] == 0.0
    assert result["metrics"]["global"]["max_error"] == 0.0
    assert result["metrics"]["global"]["relative_error"] == 0.0
    assert "fc1" in result["metrics"]["layers"]
    assert result["diagnostics"]["top_k_worst_samples"][0]["score"] == 0.0


def test_numerical_parity_detects_quantization_noise() -> None:
    model = TinyParityModel().eval()
    simulator = TorchQuantizedModelSimulator(
        model,
        activation_spec=_fixed_point_spec(4, 4),
        weight_spec=_fixed_point_spec(4, 4),
        input_spec=_fixed_point_spec(4, 4),
        output_spec=_fixed_point_spec(4, 4),
    )
    dataset = [
        torch.tensor([0.2, -0.1], dtype=torch.float32),
        torch.tensor([0.7, 0.6], dtype=torch.float32),
        torch.tensor([-0.9, 0.4], dtype=torch.float32),
    ]

    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=simulator,
        dataset=dataset,
        config={
            "metrics": ["mae", "max_error", "relative_error", "sqnr"],
            "thresholds": {"max_error": 0.01},
        },
    )

    assert result["pass"] is False
    assert result["metrics"]["global"]["mae"] > 0.0
    assert result["metrics"]["global"]["max_error"] > 0.01
    assert result["metrics"]["global"]["max_error"] < 0.2
    assert result["diagnostics"]["highest_deviation_layer"] is not None
    assert result["diagnostics"]["quantization_reports"]


def test_numerical_parity_reports_clipping_diagnostics() -> None:
    model = TinyParityModel().eval()
    simulator = TorchQuantizedModelSimulator(
        model,
        activation_spec=_fixed_point_spec(2, 2),
        weight_spec=_fixed_point_spec(2, 2),
        input_spec=_fixed_point_spec(2, 2),
        output_spec=_fixed_point_spec(2, 2),
    )
    dataset = [torch.tensor([8.0, -8.0], dtype=torch.float32)]

    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=simulator,
        dataset=dataset,
        config={"metrics": ["max_error"], "thresholds": {"max_error": 0.5}},
    )

    assert result["pass"] is False
    assert result["diagnostics"]["quantization_reports"][0]["total_clipped_values"] > 0
    assert result["metrics"]["global"]["max_error"] > 0.5


def test_numerical_parity_handles_constant_and_nonfinite_outputs() -> None:
    constant_model = ConstantModel().eval()
    constant_result = run_numerical_parity_test(
        fp32_model=constant_model,
        quantized_model=constant_model,
        dataset=[torch.tensor([1.0], dtype=torch.float32)],
        config={"metrics": ["relative_error"], "thresholds": {"relative_error": 0.0}},
    )

    assert constant_result["pass"] is True
    assert constant_result["metrics"]["global"]["relative_error"] == 0.0

    nonfinite_result = run_numerical_parity_test(
        fp32_model=constant_model,
        quantized_model=NonFiniteModel().eval(),
        dataset=[torch.tensor([1.0], dtype=torch.float32)],
        config={"metrics": ["max_error"], "thresholds": {"max_error": 0.0}},
    )

    assert nonfinite_result["pass"] is False
    assert any(
        violation["metric"] == "nonfinite_count"
        for violation in nonfinite_result["violations"]
    )


def test_numerical_parity_surfaces_ir_mismatches() -> None:
    model = TinyParityModel().eval()
    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=model,
        dataset=[torch.tensor([0.1, 0.2], dtype=torch.float32)],
        config={
            "fp32_ir": _make_graph([1, 1]),
            "quantized_ir": _make_graph([1, 2]),
        },
    )

    assert result["pass"] is False
    assert result["diagnostics"]["ir"]["pass"] is False
    assert any(violation["scope"] == "ir" for violation in result["violations"])
