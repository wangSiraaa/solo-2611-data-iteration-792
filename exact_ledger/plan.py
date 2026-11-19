"""Exact linear derivation plans.

A plan is a finite Directed Acyclic Graph of named ``Node`` objects.  Every
node is the exact rational linear combination

    node = constant + sum(coefficient_i * value(ref_i))

where ``ref_i`` is an input column name or another node's id.  Terms may
repeat, coefficients may be zero or negative, and a shared node may be
referenced from many places.  The graph must be acyclic.

The service is *independent of facts*: it validates, publishes immutable
plans, and derives -- purely algebraically -- a per-output contract

    output := constant + sum(final_coefficient * input)

The final coefficient of an input is the sum of the contributions of *all*
paths from that input to the output.  Paths are never added up naively at the
graph level (that would double count a shared node such as the ``u`` in
``(a+b)-b``); the coefficients are composed algebraically through the shared
DAG, so ``z = u - b`` with ``u = a + b`` has final ``b`` coefficient exactly
``0``.  Dependencies whose final coefficient is zero are dropped.

Nothing here reads or depends on facts, revisions or groups: that is the
replay warehouse's job.  The warehouse consumes the products this module
publishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .rational import as_fraction, to_str

__all__ = [
    "PlanError",
    "DuplicateNameError",
    "UnknownReferenceError",
    "EmptyIdentifierError",
    "CycleError",
    "InvalidPlanError",
    "Term",
    "Node",
    "Plan",
    "PlanService",
    "MAX_INPUTS",
    "MAX_NODES",
    "MAX_OUTPUTS",
]

# Documented scale limits.  These bound one plan; they are not value limits.
MAX_INPUTS = 8
MAX_NODES = 50
MAX_OUTPUTS = 10


class PlanError(ValueError):
    """Base class for all rejected plan definitions."""


class EmptyIdentifierError(PlanError):
    pass


class DuplicateNameError(PlanError):
    pass


class UnknownReferenceError(PlanError):
    pass


class CycleError(PlanError):
    pass


class InvalidPlanError(PlanError):
    pass


@dataclass(frozen=True)
class Term:
    ref: str
    coefficient: Fraction

    def as_dict(self) -> Dict[str, str]:
        return {"ref": self.ref, "coefficient": to_str(self.coefficient)}


@dataclass(frozen=True)
class Node:
    node_id: str
    terms: Tuple[Term, ...]
    constant: Fraction

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "terms": [term.as_dict() for term in self.terms],
            "constant": to_str(self.constant),
        }


@dataclass(frozen=True)
class Dependency:
    """One reduced, non-zero dependency of an output on an input column."""

    input: str
    coefficient: Fraction

    def as_dict(self) -> Dict[str, str]:
        return {"input": self.input, "coefficient": to_str(self.coefficient)}


@dataclass(frozen=True)
class OutputContract:
    """Exact algebraic contract for one output.

    ``input_coefficients`` is the *reduced* map: only inputs whose final
    coefficient is non-zero appear.  ``dependencies`` is the same data as an
    ordered tuple.  ``derived_from`` records the raw ``constant`` and
    ``terms`` plan product the contract was derived from, so a caller can
    verify the reduction rather than being asked to trust it.
    """

    output: str
    constant: Fraction
    input_coefficients: Mapping[str, Fraction]
    dependencies: Tuple[Dependency, ...]
    derived_from: Mapping[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "output": self.output,
            "constant": to_str(self.constant),
            "input_coefficients": {
                name: to_str(coefficient)
                for name, coefficient in self.input_coefficients.items()
            },
            "dependencies": [dep.as_dict() for dep in self.dependencies],
            "derived_from": dict(self.derived_from),
        }

    def coefficient_of(self, input_name: str) -> Fraction:
        return self.input_coefficients.get(input_name, Fraction(0))

    def depends_on(self, input_name: str) -> bool:
        return self.coefficient_of(input_name) != 0

    def inputs(self) -> Tuple[str, ...]:
        return tuple(self.input_coefficients.keys())


@dataclass(frozen=True)
class Plan:
    """An immutable, published plan plus everything a consumer needs.

    Instances are only produced by :class:`PlanService`; treat the fields as
    read-only.  ``digest`` is a content hash over everything semantic.
    """

    plan_id: str
    based_on: Optional[str]
    inputs: Tuple[str, ...]
    nodes: Tuple[Node, ...]
    outputs: Tuple[str, ...]
    node_products: Mapping[str, Mapping[str, object]]
    contracts: Mapping[str, OutputContract]
    digest: str
    topological_order: Tuple[str, ...] = ()

    def contract(self, output: str) -> OutputContract:
        try:
            return self.contracts[output]
        except KeyError as exc:
            raise InvalidPlanError(
                f"plan {self.plan_id!r} has no output named {output!r}"
            ) from exc

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise InvalidPlanError(f"plan {self.plan_id!r} has no node {node_id!r}")

    def node_ids(self) -> Tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    def input_set(self) -> frozenset:
        return frozenset(self.inputs)

    def as_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "based_on": self.based_on,
            "inputs": list(self.inputs),
            "nodes": [node.as_dict() for node in self.nodes],
            "outputs": list(self.outputs),
            "node_products": {
                node_id: {
                    name: to_str(coefficient)
                    for name, coefficient in product.items()
                }
                for node_id, product in self.node_products.items()
            },
            "contracts": {
                output: contract.as_dict()
                for output, contract in self.contracts.items()
            },
            "digest": self.digest,
        }


def _require_identifier(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmptyIdentifierError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _check_coefficient(value: object, what: str) -> Fraction:
    try:
        return as_fraction(value)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - re-raised as a plan error
        raise InvalidPlanError(f"{what}: {exc}") from exc


def _parse_terms(raw_terms: object, what: str) -> Tuple[Term, ...]:
    if raw_terms is None:
        return ()
    if not isinstance(raw_terms, (list, tuple)):
        raise InvalidPlanError(f"{what}: 'terms' must be a list")
    terms: List[Term] = []
    for index, raw in enumerate(raw_terms):
        if not isinstance(raw, Mapping):
            raise InvalidPlanError(f"{what}: term {index} must be an object")
        if "ref" not in raw:
            raise InvalidPlanError(f"{what}: term {index} is missing 'ref'")
        ref = _require_identifier(raw["ref"], f"{what} term {index} ref")
        coefficient = _check_coefficient(
            raw.get("coefficient", "1"), f"{what} term {index} coefficient"
        )
        terms.append(Term(ref=ref, coefficient=coefficient))
    return tuple(terms)


def _parse_node(raw: object, what: str) -> Node:
    if not isinstance(raw, Mapping):
        raise InvalidPlanError(f"{what} must be an object")
    if "node_id" not in raw:
        raise InvalidPlanError(f"{what} is missing 'node_id'")
    node_id = _require_identifier(raw["node_id"], f"{what} node_id")
    terms = _parse_terms(raw.get("terms", []), what)
    constant = _check_coefficient(raw.get("constant", "0"), f"{what} constant")
    return Node(node_id=node_id, terms=terms, constant=constant)


def _hashable_terms(terms: Sequence[Term]) -> tuple:
    return tuple((term.ref, term.coefficient) for term in terms)


class PlanService:
    """Publishes immutable plans and their exact algebraic contracts.

    The service is stateless with respect to facts.  Re-publishing the same
    ``plan_id`` with the same semantic content returns the previously stored
    :class:`Plan` object; publishing changed content under an existing
    ``plan_id`` is rejected (plans are immutable).
    """

    def __init__(self) -> None:
        self._plans: Dict[str, Plan] = {}

    # ------------------------------------------------------------------ API
    def publish(self, definition: Mapping[str, object]) -> Plan:
        """Validate ``definition`` and return its immutable :class:`Plan`."""
        if not isinstance(definition, Mapping):
            raise InvalidPlanError("plan definition must be an object")
        plan_id = _require_identifier(definition.get("plan_id"), "plan_id")

        raw_inputs = definition.get("inputs", [])
        if not isinstance(raw_inputs, (list, tuple)):
            raise InvalidPlanError("plan inputs must be a list")
        inputs: List[str] = []
        seen_inputs = set()
        for raw in raw_inputs:
            name = _require_identifier(raw, "input name")
            if name in seen_inputs:
                raise DuplicateNameError(f"duplicate input {name!r}")
            seen_inputs.add(name)
            inputs.append(name)
        if len(inputs) > MAX_INPUTS:
            raise InvalidPlanError(
                f"plan has {len(inputs)} inputs, limit is {MAX_INPUTS}"
            )

        raw_nodes = definition.get("nodes", [])
        if not isinstance(raw_nodes, (list, tuple)):
            raise InvalidPlanError("plan nodes must be a list")
        nodes: List[Node] = []
        seen_nodes: Dict[str, Node] = {}
        for index, raw in enumerate(raw_nodes):
            node = _parse_node(raw, f"node {index}")
            if node.node_id in seen_inputs:
                raise DuplicateNameError(
                    f"node id {node.node_id!r} collides with an input name"
                )
            if node.node_id in seen_nodes:
                raise DuplicateNameError(f"duplicate node id {node.node_id!r}")
            seen_nodes[node.node_id] = node
            nodes.append(node)
        if len(nodes) > MAX_NODES:
            raise InvalidPlanError(f"plan has {len(nodes)} nodes, limit is {MAX_NODES}")

        outputs = self._parse_outputs(definition.get("outputs", []), seen_nodes, seen_inputs)

        based_on = definition.get("based_on")
        if based_on is not None:
            based_on = _require_identifier(based_on, "based_on")

        order = self._topological_order(nodes, seen_nodes, seen_inputs)
        node_products = self._derive_node_products(order, seen_nodes, seen_inputs)
        node_constants = self._derive_node_constants(order, seen_nodes)
        contracts = self._derive_contracts(
            outputs, node_products, node_constants, seen_nodes
        )

        digest = self._digest(
            plan_id, based_on, inputs, nodes, outputs, node_products, contracts
        )

        existing = self._plans.get(plan_id)
        if existing is not None:
            if existing.digest != digest:
                raise InvalidPlanError(
                    f"plan_id {plan_id!r} is already published with different "
                    "content; published plans are immutable"
                )
            return existing

        plan = Plan(
            plan_id=plan_id,
            based_on=based_on,
            inputs=tuple(inputs),
            nodes=tuple(nodes),
            outputs=tuple(outputs),
            node_products=MappingProxyType(
                {
                    node_id: MappingProxyType(dict(product))
                    for node_id, product in node_products.items()
                }
            ),
            contracts=MappingProxyType(dict(contracts)),
            digest=digest,
            topological_order=tuple(order),
        )
        self._plans[plan_id] = plan
        return plan

    def publish_replacement(
        self, plan_id: str, based_on: str, replace_node: object
    ) -> Plan:
        """Publish ``plan_id`` as a copy of ``based_on`` with one node replaced.

        Inputs, outputs and every other node are preserved.  The replacement is
        a full node definition whose ``node_id`` must already exist in the base
        plan.
        """
        base = self.get(based_on)
        replacement = _parse_node(replace_node, "replacement node")
        if replacement.node_id not in base.node_products:
            raise UnknownReferenceError(
                f"replacement node {replacement.node_id!r} is not present in "
                f"plan {based_on!r}"
            )
        definition = base.as_dict()
        nodes = []
        for node in definition["nodes"]:
            if node["node_id"] == replacement.node_id:
                nodes.append(replacement.as_dict())
            else:
                nodes.append(node)
        return self.publish(
            {
                "plan_id": plan_id,
                "based_on": based_on,
                "inputs": definition["inputs"],
                "nodes": nodes,
                "outputs": definition["outputs"],
            }
        )

    def get(self, plan_id: str) -> Plan:
        try:
            return self._plans[plan_id]
        except KeyError as exc:
            raise UnknownReferenceError(f"unknown plan {plan_id!r}") from exc

    def has(self, plan_id: str) -> bool:
        return plan_id in self._plans

    def all_plans(self) -> Tuple[Plan, ...]:
        return tuple(self._plans.values())

    def explain(self, plan_id: str, output: Optional[str] = None) -> Dict[str, object]:
        """Return the published contract(s) and the raw products behind them."""
        plan = self.get(plan_id)
        if output is None:
            return {
                "plan_id": plan.plan_id,
                "digest": plan.digest,
                "outputs": {
                    name: plan.contracts[name].as_dict() for name in plan.outputs
                },
            }
        contract = plan.contract(output)
        return {
            "plan_id": plan.plan_id,
            "digest": plan.digest,
            "contract": contract.as_dict(),
            "node_ref_products": dict(plan.node_products[output]),
            "coefficient_explanations": {
                name: explain_coefficient(plan, output, name)
                for name in sorted(
                    set(plan.inputs) | set(contract.input_coefficients)
                )
            },
        }

    def explain_coefficient(
        self, plan_id: str, output: str, input_name: str
    ) -> Dict[str, object]:
        return explain_coefficient(self.get(plan_id), output, input_name)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _parse_outputs(
        raw_outputs: object, nodes: Mapping[str, Node], inputs: Iterable[str]
    ) -> Tuple[str, ...]:
        if raw_outputs is None:
            raw_outputs = []
        if not isinstance(raw_outputs, (list, tuple)):
            raise InvalidPlanError("plan outputs must be a list")
        outputs: List[str] = []
        known = set(nodes) | set(inputs)
        for raw in raw_outputs:
            name = _require_identifier(raw, "output name")
            if name in outputs:
                raise DuplicateNameError(f"duplicate output {name!r}")
            if name not in known:
                raise UnknownReferenceError(
                    f"output {name!r} is neither a node nor an input"
                )
            outputs.append(name)
        if len(outputs) > MAX_OUTPUTS:
            raise InvalidPlanError(
                f"plan has {len(outputs)} outputs, limit is {MAX_OUTPUTS}"
            )
        return tuple(outputs)

    @staticmethod
    def _topological_order(
        nodes: Sequence[Node],
        node_map: Mapping[str, Node],
        input_set: Iterable[str],
    ) -> Tuple[str, ...]:
        inputs = set(input_set)
        dependencies: Dict[str, set] = {}
        for node in nodes:
            node_deps = set()
            for term in node.terms:
                if term.ref in inputs:
                    continue
                if term.ref not in node_map:
                    raise UnknownReferenceError(
                        f"node {node.node_id!r} references unknown "
                        f"{term.ref!r} (not an input or node)"
                    )
                node_deps.add(term.ref)
            dependencies[node.node_id] = node_deps

        # Kahn's algorithm; deterministic order by original declaration.
        indegree = {node_id: len(deps) for node_id, deps in dependencies.items()}
        dependents: Dict[str, List[str]] = {node_id: [] for node_id in dependencies}
        for node_id, deps in dependencies.items():
            for dep in deps:
                dependents[dep].append(node_id)
        declared = [node.node_id for node in nodes]
        ready = [node_id for node_id in declared if indegree[node_id] == 0]
        order: List[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for dependent in dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if len(order) != len(nodes):
            cyclic = sorted(node_id for node_id in dependencies if node_id not in order)
            raise CycleError(
                "plan graph is not acyclic; nodes involved in (or blocked by) "
                f"a cycle: {cyclic}"
            )
        return tuple(order)

    @staticmethod
    def _derive_node_products(
        order: Sequence[str],
        node_map: Mapping[str, Node],
        input_set: Iterable[str],
    ) -> Dict[str, Dict[str, Fraction]]:
        """Exact input-coefficient vector for every node, via the shared DAG.

        Each node's vector is built by composing the *lifetime* vectors of its
        references, so a node shared by several paths contributes exactly once
        per distinct path, weighted by the coefficient on that edge.  Zero
        coefficients are dropped from the stored vector (but the node's own raw
        ``terms`` are preserved for auditing).
        """
        inputs = set(input_set)
        products: Dict[str, Dict[str, Fraction]] = {}
        for node_id in order:
            node = node_map[node_id]
            vector: Dict[str, Fraction] = {}
            for term in node.terms:
                if term.ref in inputs:
                    vector[term.ref] = vector.get(term.ref, Fraction(0)) + term.coefficient
                else:
                    for name, coefficient in products[term.ref].items():
                        vector[name] = vector.get(name, Fraction(0)) + term.coefficient * coefficient
            products[node_id] = {
                name: coefficient for name, coefficient in vector.items() if coefficient != 0
            }
        return products

    @staticmethod
    def _derive_node_constants(
        order: Sequence[str], node_map: Mapping[str, Node]
    ) -> Dict[str, Fraction]:
        """Exact constant folded through the shared DAG for every node."""
        constants: Dict[str, Fraction] = {}
        for node_id in order:
            node = node_map[node_id]
            total = node.constant
            for term in node.terms:
                if term.ref in node_map:
                    total += term.coefficient * constants[term.ref]
                # input references contribute no constant
            constants[node_id] = total
        return constants

    @staticmethod
    def _derive_contracts(
        outputs: Sequence[str],
        node_products: Mapping[str, Mapping[str, Fraction]],
        node_constants: Mapping[str, Fraction],
        node_map: Mapping[str, Node],
    ) -> Dict[str, OutputContract]:
        """Reduce each output to ``constant + sum(nonzero * input)``.

        For an output that is itself an input column the contract is simply
        ``that input * 1``.  For a node output, the published node product is
        the reduced non-zero input vector and the folded constant.  Zero
        dependencies are *removed*: an input whose final coefficient is zero is
        not part of the output's contract, and therefore cannot make the
        output unavailable later.
        """
        contracts: Dict[str, OutputContract] = {}
        for output in outputs:
            if output in node_map:
                node = node_map[output]
                raw_vector = dict(node_products[output])
                constant = node_constants[output]
                derived_from: Dict[str, object] = {
                    "kind": "node",
                    "node_id": node.node_id,
                    "constant": to_str(node.constant),
                    "terms": [term.as_dict() for term in node.terms],
                    "node_ref_products": {
                        term.ref: (
                            {
                                name: to_str(coef)
                                for name, coef in node_products[term.ref].items()
                            }
                            if term.ref in node_products
                            else {term.ref: to_str(term.coefficient)}
                        )
                        for term in node.terms
                    },
                    "folded_constant": to_str(constant),
                }
            else:
                # Output is an input column itself.
                raw_vector = {output: Fraction(1)}
                constant = Fraction(0)
                derived_from = {
                    "kind": "input",
                    "input": output,
                    "constant": "0",
                    "terms": [{"ref": output, "coefficient": "1"}],
                    "node_ref_products": {},
                    "folded_constant": "0",
                }

            reduced = {
                name: coefficient
                for name, coefficient in raw_vector.items()
                if coefficient != 0
            }
            dependencies = tuple(
                Dependency(input=name, coefficient=reduced[name]) for name in reduced
            )
            contracts[output] = OutputContract(
                output=output,
                constant=constant,
                input_coefficients=MappingProxyType(reduced),
                dependencies=dependencies,
                derived_from=MappingProxyType(derived_from),
            )
        return contracts

    # ------------------------------------------------------------- digest
    @staticmethod
    def _digest(
        plan_id: str,
        based_on: Optional[str],
        inputs: Sequence[str],
        nodes: Sequence[Node],
        outputs: Sequence[str],
        node_products: Mapping[str, Mapping[str, Fraction]],
        contracts: Mapping[str, OutputContract],
    ) -> str:
        import hashlib
        import json

        payload = {
            "plan_id": plan_id,
            "based_on": based_on,
            "inputs": list(inputs),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "terms": [[term.ref, to_str(term.coefficient)] for term in node.terms],
                    "constant": to_str(node.constant),
                }
                for node in nodes
            ],
            "outputs": list(outputs),
            "node_products": {
                node_id: {name: to_str(coef) for name, coef in product.items()}
                for node_id, product in node_products.items()
            },
            "contracts": {
                output: contract.as_dict() for output, contract in contracts.items()
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def explain_coefficient(plan: Plan, output: str, input_name: str) -> Dict[str, object]:
    """Trace why ``input_name`` has (or lacks) a final coefficient in ``output``.

    One exact reverse-topological pass over the shared DAG -- never path
    enumeration -- accumulates the weight flowing from ``output`` into each
    node and records the signed weight of every distinct edge into
    ``input_name``.  Each node is visited once, so a shared source is never
    double counted, and repeated terms on one edge are summed onto that edge.

    States:

    ``positive`` / ``negative``  a non-zero final coefficient;
    ``cancelled``                contributions exist but sum exactly to zero;
    ``absent``                   the input has no edge into this output at all.

    A zero that came from cancellation is never reported as missing.
    """
    contract = plan.contract(output)
    final = contract.coefficient_of(input_name)

    node_map = {node.node_id: node for node in plan.nodes}
    node_set = set(node_map)
    order = plan.topological_order or tuple(node_map)

    weight_into: Dict[str, Fraction] = {output: Fraction(1)}
    edge_weights: Dict[str, Fraction] = {}

    for node_id in reversed(order):
        weight = weight_into.get(node_id, Fraction(0))
        if weight == 0:
            continue
        for term in node_map[node_id].terms:
            if term.ref == input_name:
                edge = f"{node_id}->{input_name}"
                edge_weights[edge] = (
                    edge_weights.get(edge, Fraction(0))
                    + weight * term.coefficient
                )
            elif term.ref in node_set:
                weight_into[term.ref] = (
                    weight_into.get(term.ref, Fraction(0))
                    + weight * term.coefficient
                )

    if output == input_name:
        edge_weights[output] = edge_weights.get(output, Fraction(0)) + 1

    contributions = [
        {"weight": weight, "edge": edge}
        for edge, weight in edge_weights.items()
        if weight != 0
    ]
    raw_contributions = list(edge_weights.values())

    if final != 0:
        state = "positive" if final > 0 else "negative"
    elif raw_contributions:
        state = "cancelled"
    else:
        state = "absent"

    if state == "absent":
        explanation = (
            f"{output} has no dependency on {input_name}: no edge from the "
            "plan's published graph reaches it, so its final coefficient is 0"
        )
    elif state == "cancelled":
        explanation = (
            f"{output}'s dependency on {input_name} is fully cancelled: the "
            "signed edge contributions sum exactly to zero"
        )
    else:
        explanation = (
            f"{output} depends on {input_name} with final coefficient "
            f"{to_str(final)}, equal to the exact sum of its edge contributions"
        )

    return {
        "output": output,
        "input": input_name,
        "coefficient": to_str(final),
        "state": state,
        "depends": final != 0,
        "zero_reason": (
            None if final != 0 else ("cancelled" if state == "cancelled" else "no-path")
        ),
        "sum_of_contributions": to_str(sum(raw_contributions, Fraction(0))),
        "contributions": [
            {"weight": to_str(item["weight"]), "edge": item["edge"]}
            for item in sorted(contributions, key=lambda item: item["edge"])
        ],
        "explanation": explanation,
    }
