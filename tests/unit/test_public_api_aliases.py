from edge_lstm.ir import (
    FPGACost,
    Graph,
    Operator,
    OperatorRegistry,
    Value,
    ValueType,
    get_default_registry,
)
from edge_lstm.ir.graph import Graph as GraphModuleAlias
from edge_lstm.ir.op import Operator as OperatorModuleAlias
from edge_lstm.ir.registry import (
    OperatorRegistry as RegistryModuleAlias,
)
from edge_lstm.ir.registry import (
    get_default_registry as get_default_registry_module_alias,
)
from edge_lstm.ir.value import Value as ValueModuleAlias
from edge_lstm.ir_graph import Graph as GraphCompatibilityAlias


def test_edge_lstm_ir_namespace_reexports_core_types() -> None:
    assert Graph is GraphModuleAlias
    assert Operator is OperatorModuleAlias
    assert OperatorRegistry is RegistryModuleAlias
    assert Value is ValueModuleAlias
    assert ValueType.TENSOR.value == "tensor"
    assert FPGACost is not None


def test_edge_lstm_ir_registry_alias_matches_default_registry() -> None:
    assert get_default_registry is get_default_registry_module_alias
    assert "Add" in get_default_registry().list_registered()


def test_edge_lstm_ir_graph_remains_compatible_alias() -> None:
    assert GraphCompatibilityAlias is Graph
