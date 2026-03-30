from __future__ import annotations

import copy
import heapq
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from edge_lstm.quantization_config import (
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    compute_quant_params,
)

if TYPE_CHECKING:
    from edge_lstm.ir.graph import Graph


DEFAULT_METRICS = ("mae", "mse", "max_error", "relative_error", "sqnr")


@dataclass
class NumericalParityConfig:
    metrics: tuple[str, ...] = DEFAULT_METRICS
    thresholds: dict[str, float] = field(default_factory=dict)
    relative_error_epsilon: float = 1e-8
    top_k_worst: int = 5
    histogram_bins: int | Sequence[float] | None = None
    capture_layers: bool = True
    layer_names: tuple[str, ...] | None = None
    sample_adapter: Any | None = None
    fail_on_nonfinite: bool = True
    enforce_eval_mode: bool = True
    compare_ir: bool = True
    fp32_ir: Graph | None = None
    quantized_ir: Graph | None = None
    ranking_metric: str = "max_error"

    @classmethod
    def from_input(
        cls, config: NumericalParityConfig | Mapping[str, object] | None
    ) -> NumericalParityConfig:
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config

        metrics_value = config.get("metrics", DEFAULT_METRICS)
        layer_names_value = config.get("layer_names")
        return cls(
            metrics=tuple(str(metric) for metric in metrics_value),
            thresholds={
                str(metric): float(threshold)
                for metric, threshold in dict(config.get("thresholds", {})).items()
            },
            relative_error_epsilon=float(config.get("relative_error_epsilon", 1e-8)),
            top_k_worst=int(config.get("top_k_worst", 5)),
            histogram_bins=config.get("histogram_bins"),
            capture_layers=bool(config.get("capture_layers", True)),
            layer_names=(
                tuple(str(name) for name in layer_names_value)
                if layer_names_value is not None
                else None
            ),
            sample_adapter=config.get("sample_adapter"),
            fail_on_nonfinite=bool(config.get("fail_on_nonfinite", True)),
            enforce_eval_mode=bool(config.get("enforce_eval_mode", True)),
            compare_ir=bool(config.get("compare_ir", True)),
            fp32_ir=config.get("fp32_ir"),
            quantized_ir=config.get("quantized_ir"),
            ranking_metric=str(config.get("ranking_metric", "max_error")),
        )


@dataclass
class QuantizationSimulationResult:
    dequantized: np.ndarray
    quantized: np.ndarray
    clipped_values: int
    scale: float
    zero_point: int


@dataclass
class _MetricAccumulator:
    metrics: tuple[str, ...]
    relative_error_epsilon: float
    histogram_bins: int | Sequence[float] | None
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0
    rel_error_sum: float = 0.0
    signal_sq_sum: float = 0.0
    noise_sq_sum: float = 0.0
    element_count: int = 0
    max_error: float = 0.0
    nonfinite_count: int = 0
    histogram_counts: np.ndarray | None = None
    histogram_edges: np.ndarray | None = None

    def update(self, reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
        reference_arr = _as_float_array(reference)
        candidate_arr = _as_float_array(candidate)
        if reference_arr.shape != candidate_arr.shape:
            raise ValueError(
                "reference and candidate tensors must have identical shapes, got "
                f"{reference_arr.shape} and {candidate_arr.shape}"
            )

        nonfinite_mask = ~np.isfinite(reference_arr) | ~np.isfinite(candidate_arr)
        self.nonfinite_count += int(nonfinite_mask.sum())

        sanitized_reference = np.nan_to_num(
            reference_arr, nan=0.0, posinf=0.0, neginf=0.0
        )
        sanitized_candidate = np.nan_to_num(
            candidate_arr, nan=0.0, posinf=0.0, neginf=0.0
        )

        diff = sanitized_reference - sanitized_candidate
        abs_diff = np.abs(diff)
        sq_diff = diff * diff
        rel_denominator = np.maximum(
            np.abs(sanitized_reference), self.relative_error_epsilon
        )
        rel_diff = abs_diff / rel_denominator

        self.abs_error_sum += float(abs_diff.sum())
        self.sq_error_sum += float(sq_diff.sum())
        self.rel_error_sum += float(rel_diff.sum())
        self.signal_sq_sum += float(np.square(sanitized_reference).sum())
        self.noise_sq_sum += float(sq_diff.sum())
        self.element_count += int(abs_diff.size)

        sample_max_error = float(abs_diff.max()) if abs_diff.size else 0.0
        self.max_error = max(self.max_error, sample_max_error)

        if self.histogram_bins is not None:
            counts, edges = np.histogram(abs_diff, bins=self.histogram_bins)
            if self.histogram_counts is None:
                self.histogram_counts = counts.astype(np.int64)
                self.histogram_edges = edges
            else:
                self.histogram_counts += counts

        return _compute_metric_values(
            abs_error_sum=float(abs_diff.sum()),
            sq_error_sum=float(sq_diff.sum()),
            rel_error_sum=float(rel_diff.sum()),
            signal_sq_sum=float(np.square(sanitized_reference).sum()),
            noise_sq_sum=float(sq_diff.sum()),
            element_count=int(abs_diff.size),
            max_error=sample_max_error,
            nonfinite_count=int(nonfinite_mask.sum()),
        )

    def finalize(self) -> dict[str, object]:
        metrics = _compute_metric_values(
            abs_error_sum=self.abs_error_sum,
            sq_error_sum=self.sq_error_sum,
            rel_error_sum=self.rel_error_sum,
            signal_sq_sum=self.signal_sq_sum,
            noise_sq_sum=self.noise_sq_sum,
            element_count=self.element_count,
            max_error=self.max_error,
            nonfinite_count=self.nonfinite_count,
        )
        metrics["num_elements"] = self.element_count
        metrics["nonfinite_count"] = self.nonfinite_count
        if self.histogram_counts is not None and self.histogram_edges is not None:
            metrics["abs_error_histogram"] = {
                "bins": self.histogram_edges.tolist(),
                "counts": self.histogram_counts.tolist(),
            }
        return metrics


class TorchQuantizedModelSimulator:
    """
    CPU-side quantization simulator for PyTorch modules.

    The simulator deep-copies the reference module, optionally quantizes weights,
    and quantizes inputs/activations/outputs back to dequantized floating-point
    tensors to approximate FPGA-facing numeric behavior during parity checks.
    """

    def __init__(
        self,
        module: Any,
        *,
        activation_spec: QuantizationSpec,
        weight_spec: QuantizationSpec | None = None,
        input_spec: QuantizationSpec | None = None,
        output_spec: QuantizationSpec | None = None,
        layer_specs: Mapping[str, QuantizationSpec] | None = None,
        quantize_inputs: bool = True,
        quantize_outputs: bool = True,
        quantize_weights: bool = True,
    ) -> None:
        try:
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TorchQuantizedModelSimulator requires PyTorch to be installed."
            ) from exc

        if not isinstance(module, nn.Module):
            raise TypeError("module must be a torch.nn.Module instance")

        self._module = copy.deepcopy(module)
        self._parity_layer_model = self._module
        self.activation_spec = activation_spec
        self.weight_spec = weight_spec or activation_spec
        self.input_spec = input_spec or activation_spec
        self.output_spec = output_spec or activation_spec
        self.layer_specs = dict(layer_specs or {})
        self.quantize_inputs = quantize_inputs
        self.quantize_outputs = quantize_outputs
        self.quantize_weights = quantize_weights
        self._last_quantization_report: dict[str, object] = {
            "total_clipped_values": 0,
            "layers": {},
        }

        if self.quantize_weights:
            self._quantize_module_parameters()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        total_clipped_values = 0
        layer_reports: dict[str, dict[str, int]] = {}
        hooks: list[Any] = []

        def make_activation_hook(name: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                nonlocal total_clipped_values
                spec = self.layer_specs.get(name, self.activation_spec)
                quantized_output, clipped = _quantize_value_like(output, spec)
                total_clipped_values += clipped
                layer_reports[name] = {"clipped_values": clipped}
                return quantized_output

            return hook

        for name, submodule in self._iter_capture_modules():
            hooks.append(submodule.register_forward_hook(make_activation_hook(name)))

        try:
            prepared_args = list(args)
            prepared_kwargs = dict(kwargs)
            if self.quantize_inputs:
                prepared_args, arg_clipped = _quantize_argument_sequence(
                    prepared_args, self.input_spec
                )
                prepared_kwargs, kwarg_clipped = _quantize_argument_mapping(
                    prepared_kwargs, self.input_spec
                )
                total_clipped_values += arg_clipped + kwarg_clipped

            with torch.no_grad():
                output = self._module(*prepared_args, **prepared_kwargs)

            if self.quantize_outputs:
                output, output_clipped = _quantize_value_like(output, self.output_spec)
                total_clipped_values += output_clipped

            self._last_quantization_report = {
                "total_clipped_values": total_clipped_values,
                "layers": layer_reports,
            }
            return output
        finally:
            for hook in hooks:
                hook.remove()

    def consume_last_quantization_report(self) -> dict[str, object]:
        report = self._last_quantization_report
        self._last_quantization_report = {
            "total_clipped_values": 0,
            "layers": {},
        }
        return report

    def eval(self) -> TorchQuantizedModelSimulator:
        self._module.eval()
        return self

    def train(self, mode: bool = True) -> TorchQuantizedModelSimulator:
        self._module.train(mode)
        return self

    @property
    def training(self) -> bool:
        return bool(getattr(self._module, "training", False))

    def _iter_capture_modules(self) -> list[tuple[str, Any]]:
        return [
            (name, module)
            for name, module in self._module.named_modules()
            if name and not any(True for _ in module.children())
        ]

    def _quantize_module_parameters(self) -> None:
        for parameter in self._module.parameters():
            quantized = quantize_array(
                parameter.detach().cpu().numpy(), self.weight_spec
            )
            parameter.data.copy_(_numpy_to_like(quantized.dequantized, parameter))

        for buffer in self._module.buffers():
            if bool(getattr(buffer, "is_floating_point", lambda: False)()):
                quantized = quantize_array(
                    buffer.detach().cpu().numpy(), self.weight_spec
                )
                buffer.data.copy_(_numpy_to_like(quantized.dequantized, buffer))


def quantize_array(
    values: Any,
    spec: QuantizationSpec,
) -> QuantizationSimulationResult:
    """
    Quantize values to an integer or fixed-point domain and dequantize back.
    """
    array = _as_float_array(values)
    if array.size == 0:
        return QuantizationSimulationResult(
            dequantized=array.copy(),
            quantized=array.astype(np.int64, copy=True),
            clipped_values=0,
            scale=1.0,
            zero_point=0,
        )

    if spec.qtype == QuantizationType.FIXED_POINT:
        if spec.fixed_point is None:
            raise ValueError("fixed-point quantization requires a fixed_point spec")
        scale = (
            float(spec.scale)
            if spec.scale is not None
            else 2.0 ** (-spec.fixed_point.fractional_bits)
        )
        zero_point = int(spec.zero_point or 0)
        qmin = -(2 ** (spec.bit_width - 1))
        qmax = (2 ** (spec.bit_width - 1)) - 1
    else:
        resolved_scale, resolved_zero_point = _resolve_integer_quant_params(array, spec)
        scale = resolved_scale
        zero_point = resolved_zero_point
        if spec.scheme == QuantizationScheme.SYMMETRIC:
            qmin = -(2 ** (spec.bit_width - 1))
            qmax = (2 ** (spec.bit_width - 1)) - 1
        else:
            qmin = 0
            qmax = (2**spec.bit_width) - 1

    scaled = array / scale + zero_point
    rounded = _round_half_away_from_zero(scaled)
    clipped = np.clip(rounded, qmin, qmax)
    clipped_values = int(np.count_nonzero(rounded != clipped))
    quantized = clipped.astype(np.int64, copy=False)
    dequantized = (quantized.astype(np.float64) - zero_point) * scale
    return QuantizationSimulationResult(
        dequantized=dequantized.astype(np.float64, copy=False),
        quantized=quantized,
        clipped_values=clipped_values,
        scale=scale,
        zero_point=zero_point,
    )


def compare_ir_graphs(
    fp32_ir: Graph | None,
    quantized_ir: Graph | None,
) -> dict[str, object]:
    violations: list[dict[str, object]] = []
    if fp32_ir is None or quantized_ir is None:
        return {
            "pass": True,
            "violations": violations,
            "summary": "IR comparison skipped.",
        }

    if fp32_ir.graph_inputs != quantized_ir.graph_inputs:
        violations.append(
            {
                "scope": "ir",
                "metric": "graph_inputs",
                "actual": list(quantized_ir.graph_inputs),
                "expected": list(fp32_ir.graph_inputs),
                "message": "Quantized IR graph inputs differ from the FP32 reference.",
            }
        )

    if fp32_ir.graph_outputs != quantized_ir.graph_outputs:
        violations.append(
            {
                "scope": "ir",
                "metric": "graph_outputs",
                "actual": list(quantized_ir.graph_outputs),
                "expected": list(fp32_ir.graph_outputs),
                "message": "Quantized IR graph outputs differ from the FP32 reference.",
            }
        )

    fp32_op_types = {op_id: op.op_type for op_id, op in fp32_ir.ops.items()}
    quantized_op_types = {op_id: op.op_type for op_id, op in quantized_ir.ops.items()}
    for op_id in sorted(set(fp32_op_types) | set(quantized_op_types)):
        if fp32_op_types.get(op_id) != quantized_op_types.get(op_id):
            violations.append(
                {
                    "scope": "ir",
                    "metric": "operator_type",
                    "item": op_id,
                    "actual": quantized_op_types.get(op_id),
                    "expected": fp32_op_types.get(op_id),
                    "message": (
                        f"Operator '{op_id}' differs between FP32 and quantized IR."
                    ),
                }
            )

    for value_id in sorted(set(fp32_ir.values) | set(quantized_ir.values)):
        fp32_value = fp32_ir.values.get(value_id)
        quantized_value = quantized_ir.values.get(value_id)
        if fp32_value is None or quantized_value is None:
            violations.append(
                {
                    "scope": "ir",
                    "metric": "value_presence",
                    "item": value_id,
                    "actual": value_id in quantized_ir.values,
                    "expected": value_id in fp32_ir.values,
                    "message": f"Value '{value_id}' is missing in one IR graph.",
                }
            )
            continue
        if fp32_value.shape != quantized_value.shape:
            violations.append(
                {
                    "scope": "ir",
                    "metric": "shape",
                    "item": value_id,
                    "actual": list(quantized_value.shape),
                    "expected": list(fp32_value.shape),
                    "message": f"Value '{value_id}' shape differs between IR graphs.",
                }
            )
        if fp32_value.dtype != quantized_value.dtype:
            violations.append(
                {
                    "scope": "ir",
                    "metric": "dtype",
                    "item": value_id,
                    "actual": quantized_value.dtype,
                    "expected": fp32_value.dtype,
                    "message": f"Value '{value_id}' dtype differs between IR graphs.",
                }
            )

    summary = (
        "IR graphs match."
        if not violations
        else f"IR comparison found {len(violations)} violation(s)."
    )
    return {"pass": not violations, "violations": violations, "summary": summary}


def run_numerical_parity_test(
    fp32_model: Any,
    quantized_model: Any,
    dataset: Iterable[Any],
    config: NumericalParityConfig | Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_config = NumericalParityConfig.from_input(config)
    violations: list[dict[str, object]] = []
    output_accumulators: dict[str, _MetricAccumulator] = {}
    layer_accumulators: dict[str, _MetricAccumulator] = {}
    global_accumulator = _MetricAccumulator(
        metrics=resolved_config.metrics,
        relative_error_epsilon=resolved_config.relative_error_epsilon,
        histogram_bins=resolved_config.histogram_bins,
    )
    top_k_heap: list[tuple[float, int, dict[str, object]]] = []
    failing_samples: set[int] = set()
    failing_layers: set[str] = set()
    quantization_reports: list[dict[str, object]] = []
    sample_count = 0

    ir_report = compare_ir_graphs(
        resolved_config.fp32_ir if resolved_config.compare_ir else None,
        resolved_config.quantized_ir if resolved_config.compare_ir else None,
    )
    violations.extend(ir_report["violations"])

    with ExitStack() as stack:
        _maybe_prepare_model(stack, fp32_model, resolved_config.enforce_eval_mode)
        _maybe_prepare_model(stack, quantized_model, resolved_config.enforce_eval_mode)

        for sample_index, sample in enumerate(dataset):
            sample_count += 1
            model_args, model_kwargs = _adapt_sample(
                sample, resolved_config.sample_adapter
            )

            fp32_outputs, fp32_layers = _run_model_with_optional_layer_capture(
                fp32_model, model_args, model_kwargs, resolved_config
            )
            (
                quantized_outputs,
                quantized_layers,
            ) = _run_model_with_optional_layer_capture(
                quantized_model, model_args, model_kwargs, resolved_config
            )

            sample_score = 0.0
            sample_failures = 0

            sample_score, sample_failures = _compare_scope_maps(
                reference_map=fp32_outputs,
                candidate_map=quantized_outputs,
                accumulators=output_accumulators,
                global_accumulator=global_accumulator,
                config=resolved_config,
                scope="output",
                sample_index=sample_index,
                violations=violations,
                failing_layers=failing_layers,
                current_score=sample_score,
                current_failures=sample_failures,
                update_global=True,
            )
            sample_score, sample_failures = _compare_scope_maps(
                reference_map=fp32_layers,
                candidate_map=quantized_layers,
                accumulators=layer_accumulators,
                global_accumulator=global_accumulator,
                config=resolved_config,
                scope="layer",
                sample_index=sample_index,
                violations=violations,
                failing_layers=failing_layers,
                current_score=sample_score,
                current_failures=sample_failures,
                update_global=False,
            )

            quantization_report = _consume_quantization_report(quantized_model)
            if quantization_report:
                quantization_reports.append(
                    {"sample_index": sample_index, **quantization_report}
                )

            if sample_failures:
                failing_samples.add(sample_index)

            sample_entry = {
                "sample_index": sample_index,
                "score": sample_score,
                "num_failures": sample_failures,
            }
            if len(top_k_heap) < resolved_config.top_k_worst:
                heapq.heappush(top_k_heap, (sample_score, sample_index, sample_entry))
            elif sample_score > top_k_heap[0][0]:
                heapq.heapreplace(
                    top_k_heap, (sample_score, sample_index, sample_entry)
                )

    output_metrics = {
        name: accumulator.finalize()
        for name, accumulator in sorted(output_accumulators.items())
    }
    layer_metrics = {
        name: accumulator.finalize()
        for name, accumulator in sorted(layer_accumulators.items())
    }
    global_metrics = global_accumulator.finalize()
    pass_flag = not violations

    worst_layer = None
    if layer_metrics:
        worst_layer = max(
            layer_metrics.items(),
            key=lambda item: float(item[1].get(resolved_config.ranking_metric, 0.0)),
        )[0]

    diagnostics: dict[str, object] = {
        "top_k_worst_samples": [
            item
            for _score, _sample_index, item in sorted(
                top_k_heap, key=lambda item: (item[0], item[1]), reverse=True
            )
        ],
        "failing_samples": sorted(failing_samples),
        "failing_layers": sorted(failing_layers),
        "highest_deviation_layer": worst_layer,
        "sample_count": sample_count,
        "quantization_reports": quantization_reports,
        "ir": ir_report,
    }

    summary = _build_summary(
        pass_flag=pass_flag,
        sample_count=sample_count,
        global_metrics=global_metrics,
        violations=violations,
        worst_layer=worst_layer,
        ranking_metric=resolved_config.ranking_metric,
    )

    return {
        "metrics": {
            "global": global_metrics,
            "outputs": output_metrics,
            "layers": layer_metrics,
        },
        "pass": pass_flag,
        "violations": violations,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def _compare_scope_maps(
    *,
    reference_map: Mapping[str, np.ndarray],
    candidate_map: Mapping[str, np.ndarray],
    accumulators: dict[str, _MetricAccumulator],
    global_accumulator: _MetricAccumulator,
    config: NumericalParityConfig,
    scope: str,
    sample_index: int,
    violations: list[dict[str, object]],
    failing_layers: set[str],
    current_score: float,
    current_failures: int,
    update_global: bool,
) -> tuple[float, int]:
    for missing_name in sorted(set(reference_map) - set(candidate_map)):
        violations.append(
            {
                "scope": scope,
                "scope_name": missing_name,
                "sample_index": sample_index,
                "metric": "missing_candidate",
                "message": (
                    f"{scope.title()} '{missing_name}' is missing in quantized outputs."
                ),
            }
        )
        if scope == "layer":
            failing_layers.add(missing_name)
        current_failures += 1

    for extra_name in sorted(set(candidate_map) - set(reference_map)):
        violations.append(
            {
                "scope": scope,
                "scope_name": extra_name,
                "sample_index": sample_index,
                "metric": "unexpected_candidate",
                "message": (
                    f"{scope.title()} '{extra_name}' is missing in FP32 outputs."
                ),
            }
        )
        if scope == "layer":
            failing_layers.add(extra_name)
        current_failures += 1

    for name in sorted(set(reference_map) & set(candidate_map)):
        accumulator = accumulators.setdefault(
            name,
            _MetricAccumulator(
                metrics=config.metrics,
                relative_error_epsilon=config.relative_error_epsilon,
                histogram_bins=config.histogram_bins,
            ),
        )
        try:
            sample_metrics = accumulator.update(
                reference_map[name], candidate_map[name]
            )
            if update_global:
                global_accumulator.update(reference_map[name], candidate_map[name])
        except ValueError as exc:
            violations.append(
                {
                    "scope": scope,
                    "scope_name": name,
                    "sample_index": sample_index,
                    "metric": "shape_mismatch",
                    "message": str(exc),
                }
            )
            current_failures += 1
            if scope == "layer":
                failing_layers.add(name)
            continue
        current_score = max(
            current_score, float(sample_metrics.get(config.ranking_metric, 0.0))
        )

        if config.fail_on_nonfinite and int(sample_metrics["nonfinite_count"]) > 0:
            violations.append(
                {
                    "scope": scope,
                    "scope_name": name,
                    "sample_index": sample_index,
                    "metric": "nonfinite_count",
                    "actual": int(sample_metrics["nonfinite_count"]),
                    "threshold": 0,
                    "message": f"{scope.title()} '{name}' produced NaN or Inf values.",
                }
            )
            current_failures += 1
            if scope == "layer":
                failing_layers.add(name)

        for metric_name, threshold in config.thresholds.items():
            actual_value = float(sample_metrics.get(metric_name, 0.0))
            if actual_value > threshold:
                violations.append(
                    {
                        "scope": scope,
                        "scope_name": name,
                        "sample_index": sample_index,
                        "metric": metric_name,
                        "actual": actual_value,
                        "threshold": threshold,
                        "message": (
                            f"{scope.title()} '{name}' exceeded {metric_name}: "
                            f"{actual_value:.6g} > {threshold:.6g}"
                        ),
                    }
                )
                current_failures += 1
                if scope == "layer":
                    failing_layers.add(name)

    return current_score, current_failures


def _run_model_with_optional_layer_capture(
    model: Any,
    model_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
    config: NumericalParityConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    capture_layers = config.capture_layers and _supports_torch_layer_capture(model)
    if not capture_layers:
        return _normalize_output_structure(model(*model_args, **model_kwargs)), {}

    layer_outputs: dict[str, np.ndarray] = {}
    hooks: list[Any] = []
    layer_model = _get_layer_capture_model(model)
    selected_names = set(config.layer_names or ())

    def make_hook(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if selected_names and name not in selected_names:
                return
            layer_outputs.update(
                {
                    f"{name}.{key}" if key != "output" else name: value
                    for key, value in _normalize_output_structure(output).items()
                }
            )

        return hook

    for name, submodule in layer_model.named_modules():
        if not name:
            continue
        if selected_names:
            if name not in selected_names:
                continue
        elif any(True for _ in submodule.children()):
            continue
        hooks.append(submodule.register_forward_hook(make_hook(name)))

    try:
        output = model(*model_args, **model_kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    return _normalize_output_structure(output), layer_outputs


def _normalize_output_structure(output: Any) -> dict[str, np.ndarray]:
    if isinstance(output, Mapping):
        normalized: dict[str, np.ndarray] = {}
        for key, value in output.items():
            child_map = _normalize_output_structure(value)
            normalized.update(
                {
                    (
                        f"{key}.{child_key}" if child_key != "output" else str(key)
                    ): child_value
                    for child_key, child_value in child_map.items()
                }
            )
        return normalized
    if isinstance(output, tuple):
        normalized = {}
        for index, value in enumerate(output):
            child_map = _normalize_output_structure(value)
            normalized.update(
                {
                    f"output.{index}.{child_key}"
                    if child_key != "output"
                    else f"output.{index}": child_value
                    for child_key, child_value in child_map.items()
                }
            )
        return normalized
    if isinstance(output, list):
        normalized = {}
        for index, value in enumerate(output):
            child_map = _normalize_output_structure(value)
            normalized.update(
                {
                    f"output.{index}.{child_key}"
                    if child_key != "output"
                    else f"output.{index}": child_value
                    for child_key, child_value in child_map.items()
                }
            )
        return normalized
    return {"output": _as_float_array(output)}


def _build_summary(
    *,
    pass_flag: bool,
    sample_count: int,
    global_metrics: Mapping[str, object],
    violations: Sequence[Mapping[str, object]],
    worst_layer: str | None,
    ranking_metric: str,
) -> str:
    if pass_flag:
        return (
            f"Numerical parity passed across {sample_count} sample(s); "
            "global "
            f"{ranking_metric}={float(global_metrics.get(ranking_metric, 0.0)):.6g}."
        )

    return (
        f"Numerical parity failed with {len(violations)} violation(s) across "
        f"{sample_count} sample(s); global {ranking_metric}="
        f"{float(global_metrics.get(ranking_metric, 0.0)):.6g}, "
        f"highest deviation layer={worst_layer or 'n/a'}."
    )


def _compute_metric_values(
    *,
    abs_error_sum: float,
    sq_error_sum: float,
    rel_error_sum: float,
    signal_sq_sum: float,
    noise_sq_sum: float,
    element_count: int,
    max_error: float,
    nonfinite_count: int,
) -> dict[str, float | int]:
    if element_count <= 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "max_error": 0.0,
            "relative_error": 0.0,
            "sqnr": math.inf,
            "nonfinite_count": nonfinite_count,
        }

    if noise_sq_sum == 0.0:
        sqnr = math.inf
    elif signal_sq_sum == 0.0:
        sqnr = -math.inf
    else:
        sqnr = 10.0 * math.log10(signal_sq_sum / noise_sq_sum)

    return {
        "mae": abs_error_sum / element_count,
        "mse": sq_error_sum / element_count,
        "max_error": max_error,
        "relative_error": rel_error_sum / element_count,
        "sqnr": sqnr,
        "nonfinite_count": nonfinite_count,
    }


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def _resolve_integer_quant_params(
    array: np.ndarray,
    spec: QuantizationSpec,
) -> tuple[float, int]:
    scale = float(spec.scale) if spec.scale is not None else None
    zero_point = int(spec.zero_point) if spec.zero_point is not None else None
    if scale is None or scale <= 0.0 or zero_point is None:
        scale, zero_point = compute_quant_params(array, spec)
    if scale <= 0.0:
        scale = 1.0
    return scale, zero_point


def _quantize_argument_sequence(
    arguments: Sequence[Any],
    spec: QuantizationSpec,
) -> tuple[list[Any], int]:
    quantized_arguments: list[Any] = []
    total_clipped = 0
    for value in arguments:
        quantized_value, clipped = _quantize_value_like(value, spec)
        quantized_arguments.append(quantized_value)
        total_clipped += clipped
    return quantized_arguments, total_clipped


def _quantize_argument_mapping(
    arguments: Mapping[str, Any],
    spec: QuantizationSpec,
) -> tuple[dict[str, Any], int]:
    quantized_arguments: dict[str, Any] = {}
    total_clipped = 0
    for key, value in arguments.items():
        quantized_value, clipped = _quantize_value_like(value, spec)
        quantized_arguments[str(key)] = quantized_value
        total_clipped += clipped
    return quantized_arguments, total_clipped


def _quantize_value_like(value: Any, spec: QuantizationSpec) -> tuple[Any, int]:
    if isinstance(value, tuple):
        quantized_items = []
        total_clipped = 0
        for item in value:
            quantized_item, clipped = _quantize_value_like(item, spec)
            quantized_items.append(quantized_item)
            total_clipped += clipped
        return tuple(quantized_items), total_clipped
    if isinstance(value, list):
        quantized_items = []
        total_clipped = 0
        for item in value:
            quantized_item, clipped = _quantize_value_like(item, spec)
            quantized_items.append(quantized_item)
            total_clipped += clipped
        return quantized_items, total_clipped
    if isinstance(value, Mapping):
        quantized_items: dict[str, Any] = {}
        total_clipped = 0
        for key, item in value.items():
            quantized_item, clipped = _quantize_value_like(item, spec)
            quantized_items[str(key)] = quantized_item
            total_clipped += clipped
        return quantized_items, total_clipped

    quantized = quantize_array(value, spec)
    if _is_torch_tensor(value):
        return _numpy_to_like(quantized.dequantized, value), quantized.clipped_values
    return quantized.dequantized, quantized.clipped_values


def _numpy_to_like(array: np.ndarray, like: Any) -> Any:
    if _is_torch_tensor(like):
        import torch

        tensor = torch.from_numpy(np.asarray(array))
        return tensor.to(device=like.device, dtype=like.dtype)
    return np.asarray(array)


def _adapt_sample(
    sample: Any,
    adapter: Any | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if adapter is not None:
        adapted = adapter(sample)
        if not isinstance(adapted, tuple) or len(adapted) != 2:
            raise ValueError("sample_adapter must return a tuple of (args, kwargs)")
        adapted_args, adapted_kwargs = adapted
        return tuple(adapted_args), dict(adapted_kwargs)
    if isinstance(sample, Mapping):
        return (), dict(sample)
    if isinstance(sample, tuple):
        return tuple(sample), {}
    return (sample,), {}


def _maybe_prepare_model(stack: ExitStack, model: Any, enforce_eval_mode: bool) -> None:
    if enforce_eval_mode and hasattr(model, "eval"):
        previous_training = getattr(model, "training", None)
        model.eval()
        if previous_training is not None and hasattr(model, "train"):
            stack.callback(model.train, previous_training)

    if _is_torch_module_or_wrapper(model):
        import torch

        stack.enter_context(torch.no_grad())


def _supports_torch_layer_capture(model: Any) -> bool:
    layer_model = _get_layer_capture_model(model)
    return hasattr(layer_model, "named_modules") and _is_torch_module_or_wrapper(
        layer_model
    )


def _get_layer_capture_model(model: Any) -> Any:
    return getattr(model, "_parity_layer_model", model)


def _consume_quantization_report(model: Any) -> dict[str, object] | None:
    reporter = getattr(model, "consume_last_quantization_report", None)
    if callable(reporter):
        return dict(reporter())
    return None


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")


def _is_torch_module_or_wrapper(value: Any) -> bool:
    try:
        import torch.nn as nn
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, nn.Module) or isinstance(
        getattr(value, "_parity_layer_model", None), nn.Module
    )


def _as_float_array(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


__all__ = [
    "NumericalParityConfig",
    "TorchQuantizedModelSimulator",
    "compare_ir_graphs",
    "quantize_array",
    "run_numerical_parity_test",
]
