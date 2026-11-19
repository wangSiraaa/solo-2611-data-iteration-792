#!/usr/bin/env python3.11
"""Replay the materials/746 scenario end to end and print the outcomes.

Usage:
    python3.11 run_scenario.py [path/to/input.json]

The driver interprets the input's ``operations`` with the public API of the
two independent components: the plan service publishes/looks up plans, and the
replay warehouse imports facts, decides eligibility and publishes immutable
results.  It also exercises the follow-up steps described by the task prose
(partial facts, version 3/4, request-set qualification).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from exact_ledger import PlanService, open_warehouse, to_str


DEFAULT_INPUT = Path(__file__).with_name("materials") / "746" / "input.json"


def _fmt(values):
    return {key: to_str(value) for key, value in values.items()}


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_INPUT
    payload = json.loads(path.read_text())

    plans = PlanService()
    warehouse = open_warehouse(plans)
    results = {}

    print("== plan service ==")
    for raw_plan in payload.get("plans", [payload.get("plan")]):
        if raw_plan:
            plan = plans.publish(raw_plan)
            print(f"published {plan.plan_id} digest={plan.digest[:12]}")
            for output in plan.outputs:
                contract = plan.contract(output)
                print(
                    f"  {output} = {to_str(contract.constant)} + "
                    + " + ".join(
                        f"{to_str(c)}*{n}"
                        for n, c in contract.input_coefficients.items()
                    )
                )

    print("\n== initial facts ==")
    if "facts" in payload:
        warehouse.import_facts(payload["plan"]["plan_id"], payload["facts"])

    def show(result_id, label=None):
        result = warehouse.reopen(result_id)
        results[result_id] = result
        header = label or result_id
        print(f"\n-- {header} ({result.status}) plan={result.plan_id} "
              f"digest={result.plan_digest[:12]}")
        if result.status != "published":
            for blocker in result.blockers:
                print(
                    f"   blocked: {blocker['record_id']}.{blocker['input']} "
                    f"-> {blocker['output']} "
                    f"(coefficient {blocker['coefficient']})"
                )
            return result
        for record_id in result.records():
            print(f"   {record_id}: " + ", ".join(
                f"{out}={to_str(value)}"
                for out, value in _fmt(result.record_values[record_id]).items()
            ))
        for group in result.groups():
            print(f"   total {group}: " + ", ".join(
                f"{out}={to_str(value)}"
                for out, value in _fmt(result.group_totals[group]).items()
            ))
        return result

    for operation in payload.get("operations", []):
        op = operation["op"]
        print(f"\n== op: {op} ==")
        if op == "publish_plan":
            plan = plans.publish_replacement(
                operation["plan_id"],
                operation["based_on"],
                operation["replace_node"],
            )
            print(f"published {plan.plan_id} digest={plan.digest[:12]}")
            for output in plan.outputs:
                contract = plan.contract(output)
                print(
                    f"  {output} = {to_str(contract.constant)} + "
                    + " + ".join(
                        f"{to_str(c)}*{n}"
                        for n, c in contract.input_coefficients.items()
                    )
                )
        elif op == "build":
            result = warehouse.build(
                operation["plan_id"],
                result_id=operation["result_id"],
            )
            show(operation["result_id"], "build " + operation["result_id"])
        elif op == "import_fact":
            warehouse.import_facts(
                operation.get("plan_id", "p1"), [operation["fact"]]
            )
            print("imported fact batch")
        elif op == "correct_fact":
            warehouse.import_facts(
                operation.get("plan_id", "p1"), [operation["fact"]]
            )
            print(f"corrected {operation['fact']['record_id']} -> "
                  f"revision {operation['fact']['revision']}")
        elif op == "read_result":
            show(operation["result_id"], "reopen " + operation["result_id"])
        else:  # pragma: no cover - unknown ops are a driver error
            raise SystemExit(f"unknown operation {op!r}")

    _followups(plans, warehouse, show)


def _followups(plans, warehouse, show):
    """The prose-specified continuation after the input's own operations."""
    records = warehouse.facts.effective()
    if "r1" not in records:
        return
    group = records["r1"].group_id

    print("\n\n== follow-up: revision 3 drops b (partial fact) ==")
    warehouse.import_facts(
        "p1",
        [
            {
                "record_id": "r1",
                "revision": 3,
                "group_id": group,
                "inputs": {"a": "3"},
            }
        ],
    )
    z_only = warehouse.publish("p1", ["z"], result_id="p1-z-only-r1v3")
    show(z_only.result_id, "p1 request {z} under r1 rev3")
    zw = warehouse.publish("p1", ["z", "w"], result_id="p1-zw-r1v3")
    show(zw.result_id, "p1 request {z,w} under r1 rev3 (must reject)")
    p2_z = warehouse.publish("p2", ["z"], result_id="p2-z-r1v3")
    show(p2_z.result_id, "p2 request {z} under r1 rev3 (must reject)")

    print("\n\n== follow-up: revision 4 restores b=10 ==")
    warehouse.import_facts(
        "p1",
        [
            {
                "record_id": "r1",
                "revision": 4,
                "group_id": group,
                "inputs": {"a": "3", "b": "10"},
            }
        ],
    )
    p2_z = warehouse.publish("p2", ["z"], result_id="p2-z-r1v4")
    show(p2_z.result_id, "p2 request {z} under r1 rev4")
    p1_all = warehouse.publish("p1", ["z", "w"], result_id="p1-zw-r1v4")
    show(p1_all.result_id, "p1 request {z,w} under r1 rev4")

    print("\n\n== explain why b is/must-be a dependency ==")
    for plan_id, output in (("p1", "z"), ("p2", "z"), ("p1", "w")):
        explanation = warehouse.explain_coefficient(plan_id, output, "b")
        print(
            f"  {plan_id}.{output} <- b: coeff={explanation['coefficient']} "
            f"state={explanation['state']} "
            f"edges={[(c['weight'], c['edge']) for c in explanation['contributions']]}"
        )

    print("\n\n== trace r1 revision 4 nonzero contribution ==")
    trace = warehouse.reopen("p2-z-r1v4").trace("r1")
    for contribution in trace["nonzero_contributions"]:
        if not contribution["is_constant"]:
            print(
                f"  r1 rev{trace['revision']} {contribution['input']}="
                f"{contribution['value']} * {contribution['coefficient']} "
                f"-> {contribution['contribution']} for {contribution['output']}"
            )

    print("\n\n== old results keep their old plan and facts ==")
    for result_id in ("initial", "fact-corrected", "plan-corrected"):
        try:
            result = warehouse.reopen(result_id)
        except Exception:  # noqa: BLE001
            continue
        summary = {
            group: _fmt(totals) for group, totals in result.group_totals.items()
        }
        print(f"  {result_id}: plan={result.plan_id} status={result.status} "
              f"totals={summary}")


if __name__ == "__main__":
    main(sys.argv)
