from abc import abstractmethod

import pytest

from src.ir_graph.op import (
    FPGACost,
    InvalidOperatorDefinitionError,
    Operator,
)
from src.ir_graph.registry import (
    DuplicateOperatorError,
    OperatorRegistry,
    UnknownOperatorError,
    default_registry,
)


class ScaleOperator(Operator):
    OP_TYPE = "Scale"

    def validate(self, values):
        return None

    def estimate_fpga_cost(self, values):
        return FPGACost(latency_cycles=2, dsp=1)

    def hls_template_path(self):
        return "scale.cpp"

    def hls_context(self, values):
        return {"factor": self.attrs.get("factor", 1)}


class BiasOperator(Operator):
    OP_TYPE = "Bias"

    def validate(self, values):
        return None

    def estimate_fpga_cost(self, values):
        return FPGACost(latency_cycles=1, lut=4)

    def hls_template_path(self):
        return "bias.cpp"

    def hls_context(self, values):
        return {"bias": self.attrs.get("bias", 0)}


class AbstractRegisteredOperator(Operator):
    OP_TYPE = "AbstractRegistered"

    @abstractmethod
    def extra(self):
        """Keep the subclass abstract for registration validation."""

    def validate(self, values):
        return None

    def estimate_fpga_cost(self, values):
        return FPGACost(latency_cycles=1)

    def hls_template_path(self):
        return "abstract.cpp"

    def hls_context(self, values):
        return {}


def test_default_registry_starts_empty_until_builtins_are_registered():
    assert default_registry.list_registered() == []


def test_registry_register_and_get_returns_operator_class():
    registry = OperatorRegistry()

    returned = registry.register(ScaleOperator)

    assert returned is ScaleOperator
    assert registry.get("Scale") is ScaleOperator
    assert registry.list_registered() == ["Scale"]


def test_registry_create_instantiates_operator_from_type_name():
    registry = OperatorRegistry()
    registry.register(ScaleOperator)

    operator = registry.create(
        "Scale",
        op_id="scale_0",
        inputs=["x"],
        outputs=["y"],
        attrs={"factor": 4},
        name="scaler",
    )

    assert isinstance(operator, ScaleOperator)
    assert operator.to_dict() == {
        "op_id": "scale_0",
        "op_type": "Scale",
        "inputs": ["x"],
        "outputs": ["y"],
        "attrs": {"factor": 4},
        "name": "scaler",
        "source_span": None,
    }


def test_registry_rejects_duplicate_operator_type_registration():
    registry = OperatorRegistry()
    registry.register(ScaleOperator)

    with pytest.raises(
        DuplicateOperatorError, match="operator type 'Scale' is already registered"
    ):
        registry.register(ScaleOperator)


def test_registry_raises_clear_error_for_unknown_operator_lookup_and_create():
    registry = OperatorRegistry()

    with pytest.raises(
        UnknownOperatorError, match="operator type 'Missing' is not registered"
    ):
        registry.get("Missing")

    with pytest.raises(
        UnknownOperatorError, match="operator type 'Missing' is not registered"
    ):
        registry.create("Missing", op_id="missing_0", inputs=["x"], outputs=["y"])


def test_registry_rejects_non_operator_classes():
    registry = OperatorRegistry()

    with pytest.raises(
        InvalidOperatorDefinitionError,
        match="operator_cls must be a concrete Operator subclass",
    ):
        registry.register(dict)


def test_registry_rejects_abstract_operator_classes():
    registry = OperatorRegistry()

    with pytest.raises(
        InvalidOperatorDefinitionError,
        match="AbstractRegisteredOperator must be concrete before registration",
    ):
        registry.register(AbstractRegisteredOperator)


def test_registry_listing_is_sorted_for_deterministic_output():
    registry = OperatorRegistry()
    registry.register(ScaleOperator)
    registry.register(BiasOperator)

    assert registry.list_registered() == ["Bias", "Scale"]
