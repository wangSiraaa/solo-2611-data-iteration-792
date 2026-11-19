"""Persistence, reopen, scale and end-to-end scenario tests."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path

from exact_ledger import PlanService, open_warehouse
from exact_ledger.facts import GroupMigrationError
from exact_ledger.plan import MAX_NODES, InvalidPlanError

MATERIALS = Path(__file__).resolve().parent.parent / "materials" / "746" / "input.json"


def make_plan(plan_id="p1"):
    return {
        "plan_id": plan_id,
        "inputs": ["a", "b"],
        "nodes": [
            {"node_id": "u", "terms": [
                {"ref": "a", "coefficient": "1"},
                {"ref": "b", "coefficient": "1"},
            ]},
            {"node_id": "z", "terms": [
                {"ref": "u", "coefficient": "1"},
                {"ref": "b", "coefficient": "-1"},
            ]},
            {"node_id": "w", "terms": [
                {"ref": "u", "coefficient": "2"},
                {"ref": "a", "coefficient": "-1"},
            ]},
        ],
        "outputs": ["z", "w"],
    }


class PersistenceTests(unittest.TestCase):
    def test_results_survive_new_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "replay.sqlite")
            plans = PlanService()
            plans.publish(make_plan())
            first = open_warehouse(plans, path=path)
            first.import_facts("p1", [
                {"record_id": "r1", "revision": 1, "group_id": "g",
                 "inputs": {"a": "3", "b": "4"}},
            ])
            first.publish("p1", ["z", "w"], result_id="initial")

            # A fresh connection + fresh warehouse over the same file sees the
            # facts and the immutable result.
            reopened = open_warehouse(plans, path=path)
            result = reopened.reopen("initial")
            self.assertEqual(result.total("g", "z"), Fraction(3))
            self.assertEqual(result.total("g", "w"), Fraction(11))
            self.assertEqual(reopened.facts.effective()["r1"].revision, 1)
            self.assertEqual(len(reopened.list_results()), 1)
            # The result is self-describing: the published products it used
            # are stored with it (graph, reduced coefficients, constant).
            self.assertEqual(
                result.plan_products["node_products"]["z"], {"a": "1"}
            )
            self.assertEqual(
                result.contracts["z"]["input_coefficients"], {"a": "1"}
            )
            self.assertEqual(
                [node["node_id"] for node in result.plan_products["nodes"]],
                ["u", "z", "w"],
            )

    def test_new_result_does_not_overwrite_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "replay.sqlite")
            plans = PlanService()
            plans.publish(make_plan())
            warehouse = open_warehouse(plans, path=path)
            warehouse.import_facts("p1", [
                {"record_id": "r1", "revision": 1, "group_id": "g",
                 "inputs": {"a": "3", "b": "4"}},
            ])
            warehouse.publish("p1", ["w"], result_id="v1")
            warehouse.import_facts("p1", [
                {"record_id": "r1", "revision": 2, "group_id": "g",
                 "inputs": {"a": "3", "b": "10"}},
            ])
            warehouse.publish("p1", ["w"], result_id="v2")
            snapshot = {entry["result_id"]: entry for entry in warehouse.list_results()}
            self.assertEqual(snapshot["v1"]["plan_digest"], snapshot["v2"]["plan_digest"])
            self.assertEqual(warehouse.reopen("v1").total("g", "w"), Fraction(11))
            self.assertEqual(warehouse.reopen("v2").total("g", "w"), Fraction(23))
            # Each result carries the plan products it was built from.
            self.assertEqual(
                warehouse.reopen("v1").plan_products["plan_id"], "p1"
            )


class LimitTests(unittest.TestCase):
    def test_too_many_nodes(self):
        service = PlanService()
        nodes = [{"node_id": f"n{i}", "terms": [
            {"ref": "a", "coefficient": "1"}
        ]} for i in range(MAX_NODES + 1)]
        with self.assertRaises(InvalidPlanError):
            service.publish({"plan_id": "big", "inputs": ["a"], "nodes": nodes,
                             "outputs": ["n0"]})

    def test_too_many_inputs(self):
        service = PlanService()
        with self.assertRaises(InvalidPlanError):
            service.publish({
                "plan_id": "too-many-inputs",
                "inputs": [f"i{n}" for n in range(9)],
                "nodes": [],
                "outputs": [],
            })

    def test_too_many_outputs(self):
        service = PlanService()
        nodes = [{"node_id": f"n{i}", "terms": []} for i in range(11)]
        with self.assertRaises(InvalidPlanError):
            service.publish({"plan_id": "many-out", "inputs": [], "nodes": nodes,
                             "outputs": [node["node_id"] for node in nodes]})


class ScaleTests(unittest.TestCase):
    def test_5000_facts_50_nodes_10_outputs(self):
        inputs = [f"i{n}" for n in range(8)]
        nodes = []
        for index in range(40):
            refs = [inputs[index % 8], inputs[(index + 1) % 8]]
            nodes.append({"node_id": f"n{index}", "terms": [
                {"ref": ref, "coefficient": str(index + 1)} for ref in refs
            ]})
        outputs = []
        for out_index in range(10):
            terms = []
            for base in range(0, 40, 4):
                terms.append({"ref": f"n{base}", "coefficient": "1"})
                terms.append({"ref": f"n{base + 1}", "coefficient": "-1"})
            node_id = f"out{out_index}"
            nodes.append({"node_id": node_id, "terms": terms,
                          "constant": str(out_index)})
            outputs.append(node_id)
        # total nodes = 50
        self.assertEqual(len(nodes), 50)
        plans = PlanService()
        plan = plans.publish({
            "plan_id": "scale", "inputs": inputs, "nodes": nodes,
            "outputs": outputs,
        })
        warehouse = open_warehouse(plans)
        facts = []
        for index in range(5000):
            facts.append({
                "record_id": f"rec{index:05d}",
                "revision": 1,
                "group_id": f"g{index % 7}",
                "inputs": {name: f"{index}/{n + 1}" for n, name in enumerate(inputs)},
            })
        started = time.monotonic()
        warehouse.import_facts("scale", facts)
        result = warehouse.publish("scale", outputs, result_id="scale-result")
        elapsed = time.monotonic() - started
        self.assertTrue(result.published)
        self.assertEqual(len(result.record_values), 5000)
        self.assertEqual(len(result.group_totals), 7)
        # sanity: coefficients are exact and verifiable
        for output in outputs:
            sum_of = sum(
                (contract_coef for contract_coef in
                 plan.contract(output).input_coefficients.values()),
                Fraction(0),
            )
            self.assertIsInstance(sum_of, Fraction)
        self.assertLess(elapsed, 60, "scale run should complete promptly")


class ScenarioFileTests(unittest.TestCase):
    def test_materials_scenario_numbers(self):
        payload = json.loads(MATERIALS.read_text())
        plans = PlanService()
        plans.publish(payload["plan"])
        warehouse = open_warehouse(plans)
        warehouse.import_facts("p1", payload["facts"])

        initial = warehouse.publish("p1", ["z", "w"], result_id="initial")
        self.assertEqual(initial.total("g", "z"), Fraction(4))
        self.assertEqual(initial.total("g", "w"), Fraction(16))

        warehouse.import_facts("p1", [{
            "record_id": "r1", "revision": 2, "group_id": "g",
            "inputs": {"a": "3", "b": "10"},
        }])
        corrected = warehouse.publish("p1", ["z", "w"], result_id="fact-corrected")
        self.assertEqual(corrected.total("g", "z"), Fraction(4))
        self.assertEqual(corrected.total("g", "w"), Fraction(28))

        plans.publish_replacement("p2", "p1", payload["operations"][3]["replace_node"])
        plan_corrected = warehouse.publish("p2", ["z", "w"], result_id="plan-corrected")
        self.assertEqual(plan_corrected.total("g", "z"), Fraction(-8))
        self.assertEqual(plan_corrected.total("g", "w"), Fraction(28))

        # p1 original product still yields 4 for z on the same current facts
        replayed = warehouse.publish("p1", ["z"], result_id="p1-replay")
        self.assertEqual(replayed.total("g", "z"), Fraction(4))
        # and the originally published result is untouched
        self.assertEqual(warehouse.reopen("initial").total("g", "z"), Fraction(4))

    def test_partial_and_restore_sequence(self):
        payload = json.loads(MATERIALS.read_text())
        plans = PlanService()
        plans.publish(payload["plan"])
        plans.publish_replacement("p2", "p1", payload["operations"][3]["replace_node"])
        warehouse = open_warehouse(plans)
        warehouse.import_facts("p1", payload["facts"])
        warehouse.publish("p1", ["z", "w"], result_id="initial")

        warehouse.import_facts("p1", [{
            "record_id": "r1", "revision": 3, "group_id": "g",
            "inputs": {"a": "3"},
        }])
        self.assertTrue(warehouse.inspect("p1", ["z"]).publishable)
        z_only = warehouse.publish("p1", ["z"], result_id="z-r3")
        self.assertEqual(z_only.total("g", "z"), Fraction(4))
        self.assertFalse(warehouse.inspect("p1", ["z", "w"]).publishable)
        self.assertFalse(warehouse.inspect("p2", ["z"]).publishable)
        rejected = warehouse.publish("p1", ["z", "w"], result_id="zw-r3")
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(rejected.blockers[0]["input"], "b")
        self.assertEqual(rejected.blockers[0]["output"], "w")

        warehouse.import_facts("p1", [{
            "record_id": "r1", "revision": 4, "group_id": "g",
            "inputs": {"a": "3", "b": "10"},
        }])
        self.assertTrue(warehouse.inspect("p1", ["z", "w"]).publishable)
        restored = warehouse.publish("p2", ["z"], result_id="z-r4")
        self.assertEqual(restored.total("g", "z"), Fraction(-8))
        # identity preserved for all prior results
        self.assertEqual(warehouse.reopen("z-r3").status, "published")
        self.assertEqual(warehouse.reopen("z-r3").total("g", "z"), Fraction(4))
        self.assertEqual(warehouse.reopen("zw-r3").status, "rejected")
        self.assertEqual(warehouse.reopen("initial").total("g", "z"), Fraction(4))


if __name__ == "__main__":
    unittest.main()
