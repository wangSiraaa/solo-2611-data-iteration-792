"""Grouped fact replay warehouse.

Responsibilities (kept separate from :mod:`exact_ledger.plan`):

* own fact versions, groups and import rules (delegated to
  :class:`~exact_ledger.facts.FactStore`);
* decide, per requested output set, whether the *current* facts make that
  set publishable -- plan legality is never treated as "every output has a
  value";
* publish immutable results: per-record output values, per-group totals and
  provenance, all derived from the **published** plan products
  (reduced non-zero input coefficients and folded constants), never from a
  re-guessed display formula;
* keep old results pinned to the plan digest and fact snapshot they were
  produced from.

Eligibility rule: for every record and every requested output, every input in
that output's *reduced, non-zero* dependency map must be present in the
record's fact.  An input with coefficient zero for all requested outputs may
be absent.  A constant-only output is decidable with no inputs at all.

Rejection is atomic: if any requested output is blocked, no values or totals
are stored for the request, and a result with status ``rejected`` plus the
blocking reasons is recorded instead.

Evaluation is a sparse dot product over the reduced dependency map.  When
recomputing, either the whole snapshot is re-evaluated or records whose fact
snapshot and contract are unchanged are carried over; totals are always
rebuilt from the current record set so constants and group migrations cannot
be missed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import sqlite3
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .facts import FactStore, canonical_inputs
from .plan import (
    MAX_OUTPUTS,
    Plan,
    PlanService,
    explain_coefficient,
)
from .rational import as_fraction, to_str

__all__ = [
    "ReplayError",
    "UnknownOutputError",
    "InvalidRequestError",
    "DuplicateResultError",
    "UnknownResultError",
    "PublicationError",
    "EligibilityReport",
    "Result",
    "ReplayWarehouse",
]

CONSTANT_INPUT = ""  # provenance key used for the constant term


class ReplayError(ValueError):
    pass


class UnknownOutputError(ReplayError):
    pass


class InvalidRequestError(ReplayError):
    pass


class DuplicateResultError(ReplayError):
    pass


class UnknownResultError(ReplayError):
    pass


class PublicationError(ReplayError):
    """Raised by :meth:`ReplayWarehouse.require_publish` for a rejection."""

    def __init__(self, report: Mapping[str, object]):
        self.report = report
        blockers = report.get("blockers", [])
        detail = "; ".join(
            f"{b['record_id']}.{b['input']} blocks {b['output']}"
            for b in blockers  # type: ignore[union-attr]
        )
        super().__init__(f"request is not publishable: {detail or 'rejected'}")


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EligibilityReport:
    """Independent publishability decision for one plan and output set."""

    plan_id: str
    plan_digest: str
    requested_outputs: Tuple[str, ...]
    publishable: bool
    blockers: Tuple[Mapping[str, object], ...]
    per_output: Mapping[str, Mapping[str, object]]
    fact_snapshot: Mapping[str, object]
    contracts: Mapping[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "requested_outputs": list(self.requested_outputs),
            "publishable": self.publishable,
            "blockers": [dict(item) for item in self.blockers],
            "per_output": {name: dict(v) for name, v in self.per_output.items()},
            "fact_snapshot": dict(self.fact_snapshot),
            "contracts": dict(self.contracts),
        }


@dataclass(frozen=True)
class Result:
    """An immutable published or rejected replay result."""

    result_id: str
    plan_id: str
    plan_digest: str
    status: str
    requested_outputs: Tuple[str, ...]
    contracts: Mapping[str, object]
    plan_products: Mapping[str, object]
    fact_snapshot: Mapping[str, object]
    blockers: Tuple[Mapping[str, object], ...]
    record_values: Mapping[str, Mapping[str, Fraction]]
    group_totals: Mapping[str, Mapping[str, Fraction]]
    provenance: Tuple[Mapping[str, object], ...]
    group_sources: Tuple[Mapping[str, object], ...]
    parent_result_id: Optional[str]
    created_seq: int

    @property
    def published(self) -> bool:
        return self.status == "published"

    def value(self, record_id: str, output: str) -> Fraction:
        try:
            return self.record_values[record_id][output]
        except KeyError as exc:
            raise ReplayError(
                f"no value for record {record_id!r} output {output!r} in "
                f"result {self.result_id!r}"
            ) from exc

    def total(self, group_id: str, output: str) -> Fraction:
        try:
            return self.group_totals[group_id][output]
        except KeyError as exc:
            raise ReplayError(
                f"no total for group {group_id!r} output {output!r} in "
                f"result {self.result_id!r}"
            ) from exc

    def groups(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for entry in self.fact_snapshot.values():
            group = entry["group_id"]  # type: ignore[index]
            if group not in seen:
                seen.append(group)
        return tuple(seen)

    def records(self, group_id: Optional[str] = None) -> Tuple[str, ...]:
        names = []
        for record_id in self.record_values:
            if group_id is None or self.fact_snapshot[record_id]["group_id"] == group_id:
                names.append(record_id)
        return tuple(names)

    def provenance_for(
        self,
        record_id: Optional[str] = None,
        output: Optional[str] = None,
    ) -> Tuple[Mapping[str, object], ...]:
        return tuple(
            item
            for item in self.provenance
            if (record_id is None or item["record_id"] == record_id)
            and (output is None or item["output"] == output)
        )

    def trace(
        self,
        record_id: str,
        input_name: Optional[str] = None,
        output: Optional[str] = None,
    ) -> Dict[str, object]:
        """Trace a source fact revision to its actual contributions.

        Returns the fact's revision/group and every contribution it made to
        each requested output (including zero-valued contributions when the
        input's value is zero, flagged ``nonzero: false``).
        """
        snapshot = self.fact_snapshot.get(record_id)
        if snapshot is None:
            raise ReplayError(f"result {self.result_id!r} has no fact {record_id!r}")
        entries = [
            dict(item)
            for item in self.provenance
            if item["record_id"] == record_id
            and (input_name is None or item["input"] == input_name)
            and (output is None or item["output"] == output)
            and not item["is_constant"]
        ]
        nonzero = [item for item in entries if item["contribution"] != "0"]
        return {
            "result_id": self.result_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "record_id": record_id,
            "revision": snapshot["revision"],
            "group_id": snapshot["group_id"],
            "input": input_name,
            "output": output,
            "contributions": entries,
            "nonzero_contributions": nonzero,
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "result_id": self.result_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "status": self.status,
            "requested_outputs": list(self.requested_outputs),
            "contracts": dict(self.contracts),
            "plan_products": dict(self.plan_products),
            "fact_snapshot": dict(self.fact_snapshot),
            "blockers": [dict(item) for item in self.blockers],
            "record_values": {
                record_id: {out: to_str(val) for out, val in values.items()}
                for record_id, values in self.record_values.items()
            },
            "group_totals": {
                group: {out: to_str(val) for out, val in totals.items()}
                for group, totals in self.group_totals.items()
            },
            "provenance": [dict(item) for item in self.provenance],
            "group_sources": [dict(item) for item in self.group_sources],
            "parent_result_id": self.parent_result_id,
            "created_seq": self.created_seq,
        }


# --------------------------------------------------------------------------
class ReplayWarehouse:
    def __init__(self, connection: sqlite3.Connection, plans: PlanService) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self.plans = plans
        self.facts = FactStore(connection)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS results (
                result_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                plan_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_outputs TEXT NOT NULL,
                contracts TEXT NOT NULL,
                plan_products TEXT NOT NULL,
                fact_snapshot TEXT NOT NULL,
                blockers TEXT NOT NULL,
                parent_result_id TEXT,
                created_seq INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS result_values (
                result_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                output TEXT NOT NULL,
                num TEXT NOT NULL,
                den TEXT NOT NULL,
                PRIMARY KEY (result_id, record_id, output)
            );
            CREATE TABLE IF NOT EXISTS result_totals (
                result_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                output TEXT NOT NULL,
                num TEXT NOT NULL,
                den TEXT NOT NULL,
                PRIMARY KEY (result_id, group_id, output)
            );
            CREATE TABLE IF NOT EXISTS result_provenance (
                result_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                output TEXT NOT NULL,
                input TEXT NOT NULL,
                is_constant INTEGER NOT NULL,
                coef_num TEXT NOT NULL,
                coef_den TEXT NOT NULL,
                value_num TEXT NOT NULL,
                value_den TEXT NOT NULL,
                contrib_num TEXT NOT NULL,
                contrib_den TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS result_group_sources (
                result_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                output TEXT NOT NULL,
                input TEXT NOT NULL,
                is_constant INTEGER NOT NULL,
                contrib_num TEXT NOT NULL,
                contrib_den TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS result_provenance_idx
                ON result_provenance (result_id, record_id, output);
            CREATE INDEX IF NOT EXISTS result_values_record_idx
                ON result_values (result_id, record_id);
            """
        )
        self._conn.commit()

    # ---------------------------------------------------------- plan access
    def contracts(self, plan_id: str) -> Mapping[str, object]:
        plan = self.plans.get(plan_id)
        return {
            name: contract.as_dict() for name, contract in plan.contracts.items()
        }

    def explain_plan(
        self, plan_id: str, output: Optional[str] = None
    ) -> Dict[str, object]:
        return self.plans.explain(plan_id, output)

    def explain_coefficient(
        self, plan_id: str, output: str, input_name: str
    ) -> Dict[str, object]:
        return explain_coefficient(self.plans.get(plan_id), output, input_name)

    # ---------------------------------------------------------- fact import
    def import_facts(self, plan_id: str, raw_facts: Sequence[object]) -> int:
        """Import a fact batch for a plan; atomic and idempotent per revision."""
        plan = self.plans.get(plan_id)
        return self.facts.import_facts(raw_facts, plan.inputs)

    def import_facts_declared(
        self, declared_inputs: Iterable[str], raw_facts: Sequence[object]
    ) -> int:
        return self.facts.import_facts(raw_facts, declared_inputs)

    # ---------------------------------------------------------- eligibility
    def inspect(
        self,
        plan_id: str,
        requested_outputs: Iterable[str],
        *,
        plan: Optional[Plan] = None,
    ) -> EligibilityReport:
        plan = plan or self.plans.get(plan_id)
        outputs = self._normalise_outputs(plan, requested_outputs)
        effective = self.facts.effective()

        snapshot: Dict[str, object] = {}
        for record_id, fact in effective.items():
            snapshot[record_id] = {
                "revision": fact.revision,
                "group_id": fact.group_id,
                "complete": fact.complete,
                "inputs": canonical_inputs(fact.inputs),
            }

        blockers: List[Dict[str, object]] = []
        per_output: Dict[str, Dict[str, object]] = {}
        contracts: Dict[str, object] = {}
        for output in outputs:
            contract = plan.contract(output)
            contracts[output] = {
                "constant": to_str(contract.constant),
                "input_coefficients": {
                    name: to_str(coef)
                    for name, coef in contract.input_coefficients.items()
                },
            }
            required = tuple(contract.input_coefficients)
            blocking_records: Dict[str, List[str]] = {}
            for record_id in sorted(effective):
                fact = effective[record_id]
                missing = [name for name in required if name not in fact.inputs]
                if missing:
                    blocking_records[record_id] = missing
                    for name in missing:
                        blockers.append(
                            {
                                "record_id": record_id,
                                "group_id": fact.group_id,
                                "revision": fact.revision,
                                "input": name,
                                "output": output,
                                "coefficient": to_str(
                                    contract.coefficient_of(name)
                                ),
                                "reason": (
                                    f"record {record_id!r} (revision "
                                    f"{fact.revision}) does not supply input "
                                    f"{name!r}, which is a reduced non-zero "
                                    f"dependency of output {output!r}"
                                ),
                            }
                        )
            per_output[output] = {
                "constant": to_str(contract.constant),
                "required_inputs": list(required),
                "reduced_dependencies": {
                    name: to_str(coef)
                    for name, coef in contract.input_coefficients.items()
                },
                "available": not blocking_records,
                "blocking_records": blocking_records,
            }

        return EligibilityReport(
            plan_id=plan_id,
            plan_digest=plan.digest,
            requested_outputs=outputs,
            publishable=not blockers,
            blockers=tuple(blockers),
            per_output=per_output,
            fact_snapshot=snapshot,
            contracts=contracts,
        )

    # alias: the spec's "qualify a requested output set"
    decide = inspect
    can_publish = inspect

    def require_publish(
        self, plan_id: str, requested_outputs: Iterable[str]
    ) -> EligibilityReport:
        report = self.inspect(plan_id, requested_outputs)
        if not report.publishable:
            raise PublicationError(report.as_dict())
        return report

    # ---------------------------------------------------------- publishing
    def publish(
        self,
        plan_id: str,
        requested_outputs: Iterable[str],
        result_id: Optional[str] = None,
        parent_result_id: Optional[str] = None,
        plan: Optional[Plan] = None,
        reuse_from: Optional[Result] = None,
    ) -> Result:
        """Create a new immutable result (published or rejected).

        ``reuse_from`` enables a reduced recompute: records whose fact snapshot
        and contract are unchanged reuse their previous values; all totals and
        provenance are rebuilt from the current record set.
        """
        plan = plan or self.plans.get(plan_id)
        if plan.plan_id != plan_id:
            raise ReplayError(
                f"plan {plan.plan_id!r} does not match requested plan_id "
                f"{plan_id!r}"
            )
        outputs = self._normalise_outputs(plan, requested_outputs)
        report = self.inspect(plan_id, outputs, plan=plan)

        if result_id is None:
            result_id = self._next_result_id()
        elif self._result_exists(result_id):
            raise DuplicateResultError(
                f"result {result_id!r} already exists; results are immutable"
            )

        seq = self._next_seq()
        snapshot = dict(report.fact_snapshot)
        contracts = dict(report.contracts)
        # Persist the exact published plan graph + products this result was
        # computed from, so the result is self-describing even after a restart
        # or after the plan service is reloaded with a newer product.
        plan_products = plan.as_dict()

        if not report.publishable:
            return self._store(
                result_id=result_id,
                plan=plan,
                outputs=outputs,
                snapshot=snapshot,
                contracts=contracts,
                plan_products=plan_products,
                blockers=tuple(dict(item) for item in report.blockers),
                record_values={},
                group_totals={},
                provenance=[],
                group_sources=[],
                parent_result_id=parent_result_id,
                seq=seq,
                status="rejected",
            )

        record_values, provenance = self._evaluate(
            plan, outputs, snapshot, reuse_from
        )
        group_totals = self._totals(outputs, snapshot, record_values)
        group_sources = self._group_sources(
            outputs, snapshot, record_values, provenance
        )
        return self._store(
            result_id=result_id,
            plan=plan,
            outputs=outputs,
            snapshot=snapshot,
            contracts=contracts,
            plan_products=plan_products,
            blockers=(),
            record_values=record_values,
            group_totals=group_totals,
            provenance=provenance,
            group_sources=group_sources,
            parent_result_id=parent_result_id,
            seq=seq,
            status="published",
        )

    def build(
        self,
        plan_id: str,
        requested_outputs: Optional[Iterable[str]] = None,
        result_id: Optional[str] = None,
        parent_result_id: Optional[str] = None,
    ) -> Result:
        plan = self.plans.get(plan_id)
        outputs = plan.outputs if requested_outputs is None else requested_outputs
        return self.publish(plan_id, outputs, result_id=result_id,
                            parent_result_id=parent_result_id)

    def recompute(
        self,
        result_id: str,
        requested_outputs: Optional[Iterable[str]] = None,
    ) -> Result:
        """Full recompute as a new result pinned to current plan and facts."""
        base = self.read_result(result_id)
        plan = self.plans.get(base.plan_id)
        outputs = (
            base.requested_outputs
            if requested_outputs is None
            else requested_outputs
        )
        return self.publish(
            base.plan_id,
            outputs,
            parent_result_id=base.result_id,
            plan=plan,
        )

    def incremental_recompute(
        self,
        base_result_id: str,
        requested_outputs: Optional[Iterable[str]] = None,
        plan_id: Optional[str] = None,
    ) -> Result:
        """Reduced recompute reusing unchanged record values.

        Reuses a previous value only when the record's effective revision,
        content and group are unchanged *and* the output's reduced contract is
        identical.  Totals are always rebuilt across the current record set, so
        constants and group migrations are never dropped.
        """
        base = self.read_result(base_result_id)
        target_plan_id = plan_id or base.plan_id
        outputs = (
            base.requested_outputs
            if requested_outputs is None
            else requested_outputs
        )
        reuse = base if base.published else None
        return self.publish(
            target_plan_id,
            outputs,
            parent_result_id=base.result_id,
            reuse_from=reuse,
        )

    # ------------------------------------------------------------- reading
    def read_result(self, result_id: str) -> Result:
        row = self._conn.execute(
            "SELECT * FROM results WHERE result_id = ?", (result_id,)
        ).fetchone()
        if row is None:
            raise UnknownResultError(f"unknown result {result_id!r}")

        record_values: Dict[str, Dict[str, Fraction]] = {}
        groups: Dict[str, str] = {}
        for item in self._conn.execute(
            "SELECT record_id, group_id, output, num, den FROM result_values "
            "WHERE result_id = ? ORDER BY record_id, output",
            (result_id,),
        ):
            record_values.setdefault(item["record_id"], {})[item["output"]] = (
                Fraction(int(item["num"]), int(item["den"]))
            )
            groups[item["record_id"]] = item["group_id"]

        group_totals: Dict[str, Dict[str, Fraction]] = {}
        for item in self._conn.execute(
            "SELECT group_id, output, num, den FROM result_totals "
            "WHERE result_id = ? ORDER BY group_id, output",
            (result_id,),
        ):
            group_totals.setdefault(item["group_id"], {})[item["output"]] = (
                Fraction(int(item["num"]), int(item["den"]))
            )

        provenance: List[Dict[str, object]] = []
        for item in self._conn.execute(
            "SELECT * FROM result_provenance WHERE result_id = ? "
            "ORDER BY record_id, output, is_constant DESC, input",
            (result_id,),
        ):
            provenance.append(
                {
                    "record_id": item["record_id"],
                    "group_id": item["group_id"],
                    "revision": item["revision"],
                    "output": item["output"],
                    "input": item["input"],
                    "is_constant": bool(item["is_constant"]),
                    "coefficient": to_str(
                        Fraction(int(item["coef_num"]), int(item["coef_den"]))
                    ),
                    "value": to_str(
                        Fraction(int(item["value_num"]), int(item["value_den"]))
                    ),
                    "contribution": to_str(
                        Fraction(
                            int(item["contrib_num"]), int(item["contrib_den"])
                        )
                    ),
                    "nonzero": int(item["contrib_num"]) != 0,
                }
            )

        group_sources: List[Dict[str, object]] = []
        for item in self._conn.execute(
            "SELECT * FROM result_group_sources WHERE result_id = ? "
            "ORDER BY group_id, output, is_constant DESC, input",
            (result_id,),
        ):
            group_sources.append(
                {
                    "group_id": item["group_id"],
                    "output": item["output"],
                    "input": item["input"],
                    "is_constant": bool(item["is_constant"]),
                    "contribution": to_str(
                        Fraction(
                            int(item["contrib_num"]), int(item["contrib_den"])
                        )
                    ),
                }
            )

        return Result(
            result_id=row["result_id"],
            plan_id=row["plan_id"],
            plan_digest=row["plan_digest"],
            status=row["status"],
            requested_outputs=tuple(json.loads(row["requested_outputs"])),
            contracts=MappingProxyType(json.loads(row["contracts"])),
            plan_products=MappingProxyType(json.loads(row["plan_products"])),
            fact_snapshot=MappingProxyType(json.loads(row["fact_snapshot"])),
            blockers=tuple(json.loads(row["blockers"])),
            record_values=MappingProxyType(record_values),
            group_totals=MappingProxyType(group_totals),
            provenance=tuple(provenance),
            group_sources=tuple(group_sources),
            parent_result_id=row["parent_result_id"],
            created_seq=row["created_seq"],
        )

    # alias: "reopen" entry point
    reopen = read_result

    def list_results(self) -> List[Dict[str, object]]:
        rows = self._conn.execute(
            "SELECT result_id, plan_id, plan_digest, status, "
            "requested_outputs, parent_result_id, created_seq "
            "FROM results ORDER BY created_seq"
        ).fetchall()
        return [
            {
                "result_id": row["result_id"],
                "plan_id": row["plan_id"],
                "plan_digest": row["plan_digest"],
                "status": row["status"],
                "requested_outputs": json.loads(row["requested_outputs"]),
                "parent_result_id": row["parent_result_id"],
                "created_seq": row["created_seq"],
            }
            for row in rows
        ]

    def result_sources(
        self, result_id: str, group_id: Optional[str] = None
    ) -> Dict[str, object]:
        result = self.read_result(result_id)
        sources = [
            dict(item)
            for item in result.group_sources
            if group_id is None or item["group_id"] == group_id
        ]
        return {
            "result_id": result_id,
            "plan_id": result.plan_id,
            "plan_digest": result.plan_digest,
            "status": result.status,
            "group_sources": sources,
        }

    # ----------------------------------------------------------- internals
    def _normalise_outputs(
        self, plan: Plan, requested_outputs: Iterable[str]
    ) -> Tuple[str, ...]:
        outputs: List[str] = []
        for name in requested_outputs:
            if not isinstance(name, str) or not name:
                raise InvalidRequestError(
                    f"requested output must be a non-empty string, got {name!r}"
                )
            if name not in plan.contracts:
                raise UnknownOutputError(
                    f"output {name!r} is not published by plan {plan.plan_id!r}; "
                    f"known outputs: {list(plan.outputs)}"
                )
            if name not in outputs:
                outputs.append(name)
        if not outputs:
            raise InvalidRequestError("at least one output must be requested")
        if len(outputs) > MAX_OUTPUTS:
            raise InvalidRequestError(
                f"requested {len(outputs)} outputs, limit is {MAX_OUTPUTS}"
            )
        return tuple(outputs)

    def _evaluate(
        self,
        plan: Plan,
        outputs: Sequence[str],
        snapshot: Mapping[str, object],
        reuse_from: Optional[Result],
    ) -> Tuple[Dict[str, Dict[str, Fraction]], List[Dict[str, object]]]:
        record_values: Dict[str, Dict[str, Fraction]] = {}
        provenance: List[Dict[str, object]] = []
        reused_contracts = (
            reuse_from.contracts if reuse_from is not None else None
        )

        for record_id in sorted(snapshot):
            entry = snapshot[record_id]
            fact_inputs = {
                name: as_fraction(value)  # exact strings round-trip
                for name, value in entry["inputs"].items()  # type: ignore[union-attr]
            }
            values: Dict[str, Fraction] = {}
            for output in outputs:
                contract = plan.contract(output)
                contract_payload = {
                    "constant": to_str(contract.constant),
                    "input_coefficients": {
                        name: to_str(coef)
                        for name, coef in contract.input_coefficients.items()
                    },
                }
                reusable = (
                    reuse_from is not None
                    and reused_contracts is not None
                    and record_id in reuse_from.record_values
                    and reuse_from.fact_snapshot.get(record_id) == entry
                    and reused_contracts.get(output) == contract_payload
                    and output in reuse_from.record_values[record_id]
                )
                if reusable:
                    values[output] = reuse_from.record_values[record_id][output]
                    continue

                value = contract.constant
                for input_name, coefficient in contract.input_coefficients.items():
                    value += coefficient * fact_inputs[input_name]
                values[output] = value
            record_values[record_id] = values

        # Provenance is always rebuilt so it always describes this result.
        for record_id in sorted(snapshot):
            entry = snapshot[record_id]
            fact_inputs = {
                name: as_fraction(value)
                for name, value in entry["inputs"].items()  # type: ignore[union-attr]
            }
            for output in outputs:
                contract = plan.contract(output)
                provenance.append(
                    {
                        "record_id": record_id,
                        "group_id": entry["group_id"],
                        "revision": entry["revision"],
                        "output": output,
                        "input": CONSTANT_INPUT,
                        "is_constant": True,
                        "coefficient": "1",
                        "value": to_str(contract.constant),
                        "contribution": to_str(contract.constant),
                        "nonzero": contract.constant != 0,
                    }
                )
                for input_name, coefficient in contract.input_coefficients.items():
                    input_value = fact_inputs[input_name]
                    contribution = coefficient * input_value
                    provenance.append(
                        {
                            "record_id": record_id,
                            "group_id": entry["group_id"],
                            "revision": entry["revision"],
                            "output": output,
                            "input": input_name,
                            "is_constant": False,
                            "coefficient": to_str(coefficient),
                            "value": to_str(input_value),
                            "contribution": to_str(contribution),
                            "nonzero": contribution != 0,
                        }
                    )
        return record_values, provenance

    @staticmethod
    def _totals(
        outputs: Sequence[str],
        snapshot: Mapping[str, object],
        record_values: Mapping[str, Mapping[str, Fraction]],
    ) -> Dict[str, Dict[str, Fraction]]:
        totals: Dict[str, Dict[str, Fraction]] = {}
        for record_id in sorted(snapshot):
            group = snapshot[record_id]["group_id"]  # type: ignore[index]
            bucket = totals.setdefault(group, {})
            for output in outputs:
                bucket[output] = bucket.get(output, Fraction(0)) + record_values[
                    record_id
                ][output]
        return totals

    @staticmethod
    def _group_sources(
        outputs: Sequence[str],
        snapshot: Mapping[str, object],
        record_values: Mapping[str, Mapping[str, Fraction]],
        provenance: Sequence[Mapping[str, object]],
    ) -> List[Dict[str, object]]:
        sources: Dict[Tuple[str, str, str, bool], Fraction] = {}
        for item in provenance:
            key = (
                item["group_id"],  # type: ignore[arg-type]
                item["output"],  # type: ignore[arg-type]
                item["input"],  # type: ignore[arg-type]
                item["is_constant"],  # type: ignore[arg-type]
            )
            sources[key] = sources.get(key, Fraction(0)) + as_fraction(
                item["contribution"]  # type: ignore[arg-type]
            )
        return [
            {
                "group_id": group,
                "output": output,
                "input": input_name,
                "is_constant": is_constant,
                "contribution": to_str(contribution),
            }
            for (group, output, input_name, is_constant), contribution in sorted(
                sources.items()
            )
        ]

    def _store(
        self,
        *,
        result_id: str,
        plan: Plan,
        outputs: Sequence[str],
        snapshot: Mapping[str, object],
        contracts: Mapping[str, object],
        plan_products: Mapping[str, object],
        blockers: Sequence[Mapping[str, object]],
        record_values: Mapping[str, Mapping[str, Fraction]],
        group_totals: Mapping[str, Mapping[str, Fraction]],
        provenance: Sequence[Mapping[str, object]],
        group_sources: Sequence[Mapping[str, object]],
        parent_result_id: Optional[str],
        seq: int,
        status: str,
    ) -> Result:
        payload_outputs = json.dumps(list(outputs), sort_keys=True)
        payload_contracts = json.dumps(contracts, sort_keys=True)
        payload_plan_products = json.dumps(plan_products, sort_keys=True)
        payload_snapshot = json.dumps(snapshot, sort_keys=True)
        payload_blockers = json.dumps(list(blockers), sort_keys=True)

        with self._conn:
            self._conn.execute(
                "INSERT INTO results (result_id, plan_id, plan_digest, status, "
                "requested_outputs, contracts, plan_products, fact_snapshot, "
                "blockers, parent_result_id, created_seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result_id,
                    plan.plan_id,
                    plan.digest,
                    status,
                    payload_outputs,
                    payload_contracts,
                    payload_plan_products,
                    payload_snapshot,
                    payload_blockers,
                    parent_result_id,
                    seq,
                ),
            )
            for record_id in sorted(record_values):
                group = snapshot[record_id]["group_id"]  # type: ignore[index]
                for output in outputs:
                    value = record_values[record_id][output]
                    num, den = str(value.numerator), str(value.denominator)
                    self._conn.execute(
                        "INSERT INTO result_values (result_id, record_id, "
                        "group_id, output, num, den) VALUES (?, ?, ?, ?, ?, ?)",
                        (result_id, record_id, group, output, num, den),
                    )
            for group in sorted(group_totals):
                for output in outputs:
                    value = group_totals[group][output]
                    self._conn.execute(
                        "INSERT INTO result_totals (result_id, group_id, "
                        "output, num, den) VALUES (?, ?, ?, ?, ?)",
                        (
                            result_id,
                            group,
                            output,
                            str(value.numerator),
                            str(value.denominator),
                        ),
                    )
            for item in provenance:
                coef = as_fraction(item["coefficient"])  # type: ignore[arg-type]
                val = as_fraction(item["value"])  # type: ignore[arg-type]
                contrib = as_fraction(item["contribution"])  # type: ignore[arg-type]
                self._conn.execute(
                    "INSERT INTO result_provenance (result_id, record_id, "
                    "group_id, revision, output, input, is_constant, coef_num, "
                    "coef_den, value_num, value_den, contrib_num, contrib_den) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result_id,
                        item["record_id"],
                        item["group_id"],
                        item["revision"],
                        item["output"],
                        item["input"],
                        int(bool(item["is_constant"])),
                        str(coef.numerator),
                        str(coef.denominator),
                        str(val.numerator),
                        str(val.denominator),
                        str(contrib.numerator),
                        str(contrib.denominator),
                    ),
                )
            for item in group_sources:
                contrib = as_fraction(item["contribution"])  # type: ignore[arg-type]
                self._conn.execute(
                    "INSERT INTO result_group_sources (result_id, group_id, "
                    "output, input, is_constant, contrib_num, contrib_den) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        result_id,
                        item["group_id"],
                        item["output"],
                        item["input"],
                        int(bool(item["is_constant"])),
                        str(contrib.numerator),
                        str(contrib.denominator),
                    ),
                )
        return self.read_result(result_id)

    def _result_exists(self, result_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM results WHERE result_id = ?", (result_id,)
            ).fetchone()
            is not None
        )

    def _next_seq(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(created_seq), 0) + 1 FROM results"
        ).fetchone()
        return int(row[0])

    def _next_result_id(self) -> str:
        seq = self._next_seq()
        candidate = f"result-{seq:06d}"
        while self._result_exists(candidate):
            seq += 1
            candidate = f"result-{seq:06d}"
        return candidate
