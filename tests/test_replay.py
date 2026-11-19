"""Tests for grouped fact import and replay eligibility/results."""

from __future__ import annotations

import unittest
from fractions import Fraction

from exact_ledger import (
    DuplicateResultError,
    GroupMigrationError,
    PlanService,
    PublicationError,
    ReplayWarehouse,
    RevisionConflictError,
    UnknownOutputError,
    open_warehouse,
)


P1 = {
    "plan_id": "p1",
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

INITIAL = [
    {"record_id": "r1", "revision": 1, "group_id": "g", "inputs": {"a": "3", "b": "4"}},
    {"record_id": "r2", "revision": 1, "group_id": "g", "inputs": {"a": "1", "b": "2"}},
]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.plans = PlanService()
        self.plans.publish(P1)
        self.warehouse = open_warehouse(self.plans)
        self.warehouse.import_facts("p1", INITIAL)

    def correct(self, revision, inputs, record="r1", group="g"):
        self.warehouse.import_facts(
            "p1",
            [{"record_id": record, "revision": revision, "group_id": group,
              "inputs": inputs}],
        )


class ScenarioTests(Fixture):
    def test_initial_totals(self):
        result = self.warehouse.publish("p1", ["z", "w"], result_id="initial")
        self.assertTrue(result.published)
        self.assertEqual(result.total("g", "z"), Fraction(4))
        self.assertEqual(result.total("g", "w"), Fraction(16))

    def test_b_correction_does_not_change_zero_dependency(self):
        result = self.warehouse.publish("p1", ["z", "w"], result_id="initial")
        self.correct(2, {"a": "3", "b": "10"})
        corrected = self.warehouse.publish(
            "p1", ["z", "w"], result_id="fact-corrected",
            parent_result_id="initial",
        )
        # z depends only on a; w moves.
        self.assertEqual(corrected.total("g", "z"), Fraction(4))
        self.assertEqual(corrected.total("g", "w"), Fraction(28))
        # old result pinned to old facts
        self.assertEqual(self.warehouse.reopen("initial").total("g", "w"), Fraction(16))

    def test_p2_z_is_a_minus_b_and_old_product_stays(self):
        self.plans.publish_replacement("p2", "p1", {
            "node_id": "z",
            "terms": [
                {"ref": "u", "coefficient": "1"},
                {"ref": "b", "coefficient": "-2"},
            ],
        })
        self.correct(2, {"a": "3", "b": "10"})
        new = self.warehouse.publish("p2", ["z", "w"], result_id="plan-corrected")
        self.assertEqual(new.total("g", "z"), Fraction(-8))
        self.assertEqual(new.total("g", "w"), Fraction(28))
        old = self.warehouse.publish("p1", ["z"], result_id="p1-again")
        self.assertEqual(old.total("g", "z"), Fraction(4))

    def test_partial_fact_keeps_z_publishable_but_blocks_w(self):
        self.correct(2, {"a": "3", "b": "10"})
        self.correct(3, {"a": "3"})  # b withdrawn
        z_only = self.warehouse.inspect("p1", ["z"])
        self.assertTrue(z_only.publishable)
        z = self.warehouse.publish("p1", ["z"], result_id="z-only")
        self.assertEqual(z.total("g", "z"), Fraction(4))
        policy = self.warehouse.inspect("p1", ["z", "w"])
        self.assertFalse(policy.publishable)
        self.assertEqual(len(policy.blockers), 1)
        blocker = policy.blockers[0]
        self.assertEqual((blocker["record_id"], blocker["input"], blocker["output"]),
                         ("r1", "b", "w"))
        rejected = self.warehouse.publish("p1", ["z", "w"], result_id="zw")
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(rejected.record_values, {})
        self.assertEqual(rejected.group_totals, {})
        with self.assertRaises(PublicationError):
            self.warehouse.require_publish("p1", ["z", "w"])

    def test_p2_z_blocks_on_b(self):
        self.plans.publish_replacement("p2", "p1", {
            "node_id": "z",
            "terms": [
                {"ref": "u", "coefficient": "1"},
                {"ref": "b", "coefficient": "-2"},
            ],
        })
        self.correct(3, {"a": "3"})
        report = self.warehouse.inspect("p2", ["z"])
        self.assertFalse(report.publishable)
        self.assertEqual(report.blockers[0]["input"], "b")
        self.assertEqual(report.blockers[0]["coefficient"], "-1")
        result = self.warehouse.publish("p2", ["z"], result_id="p2-z")
        self.assertEqual(result.status, "rejected")

    def test_restore_revision4(self):
        self.plans.publish_replacement("p2", "p1", {
            "node_id": "z",
            "terms": [
                {"ref": "u", "coefficient": "1"},
                {"ref": "b", "coefficient": "-2"},
            ],
        })
        self.correct(3, {"a": "3"})
        self.assertEqual(self.warehouse.publish("p2", ["z"], result_id="r3").status,
                         "rejected")
        self.correct(4, {"a": "3", "b": "10"})
        restored = self.warehouse.publish("p2", ["z"], result_id="r4")
        self.assertTrue(restored.published)
        self.assertEqual(restored.total("g", "z"), Fraction(-8))
        # rejected result is still there, with its own identity
        self.assertEqual(self.warehouse.reopen("r3").status, "rejected")

    def test_missing_zero_dependency_does_not_block(self):
        # z has no b dependency, so a fact with only a is publishable for z.
        self.correct(2, {"a": "5"})
        self.assertTrue(self.warehouse.inspect("p1", ["z"]).publishable)
        z = self.warehouse.publish("p1", ["z"])
        self.assertEqual(z.total("g", "z"), Fraction(6))

    def test_constant_output_publishable_without_any_inputs(self):
        plans = PlanService()
        plans.publish({
            "plan_id": "k",
            "inputs": ["a"],
            "nodes": [{"node_id": "c", "terms": [], "constant": "9/2"}],
            "outputs": ["c"],
        })
        warehouse = open_warehouse(plans)
        result = warehouse.publish("k", ["c"])
        self.assertTrue(result.published)
        self.assertEqual(result.record_values, {})
        # no records means grouped total is empty, not zero
        with self.assertRaises(Exception):
            result.total("g", "c")

    def test_constant_output_with_facts(self):
        plans = PlanService()
        plans.publish({
            "plan_id": "k",
            "inputs": ["a"],
            "nodes": [{"node_id": "c", "terms": [
                {"ref": "a", "coefficient": "0"},
            ], "constant": "9/2"}],
            "outputs": ["c"],
        })
        warehouse = open_warehouse(plans)
        warehouse.import_facts("k", [
            {"record_id": "r", "revision": 1, "group_id": "g", "inputs": {}},
        ])
        result = warehouse.publish("k", ["c"])
        self.assertTrue(result.published)
        self.assertEqual(result.total("g", "c"), Fraction(9, 2))

    def test_unknown_output_rejected(self):
        with self.assertRaises(UnknownOutputError):
            self.warehouse.inspect("p1", ["nope"])


class FactRuleTests(Fixture):
    def test_identical_replay_is_idempotent(self):
        added = self.warehouse.import_facts("p1", [INITIAL[0]])
        self.assertEqual(added, 0)
        self.assertEqual(len(self.warehouse.facts.all_revisions("r1")), 1)

    def test_conflicting_same_revision_rejects_batch(self):
        batch = [
            {"record_id": "r1", "revision": 1, "group_id": "g",
             "inputs": {"a": "3", "b": "99"}},
            {"record_id": "r3", "revision": 1, "group_id": "g",
             "inputs": {"a": "7", "b": "8"}},
        ]
        with self.assertRaises(RevisionConflictError):
            self.warehouse.import_facts("p1", batch)
        # nothing from the batch landed
        self.assertIsNone(self.warehouse.facts.get_revision("r3", 1))

    def test_missing_inputs_are_not_zero(self):
        self.correct(2, {"a": "3"})
        fact = self.warehouse.facts.effective()["r1"]
        self.assertNotIn("b", fact.inputs)
        self.assertFalse(fact.complete)

    def test_group_migration_requires_higher_complete(self):
        # Higher and covering every input, but not *explicitly* complete.
        with self.assertRaises(GroupMigrationError):
            self.correct(2, {"a": "3", "b": "4"}, group="h")
        # Same revision with different content is a revision conflict.
        with self.assertRaises(RevisionConflictError):
            self.warehouse.import_facts("p1", [
                {"record_id": "r1", "revision": 1, "group_id": "h",
                 "inputs": {"a": "3", "b": "4"}, "complete": True},
            ])
        # Explicitly complete but incomplete content is invalid.
        from exact_ledger import InvalidFactError
        with self.assertRaises(InvalidFactError):
            self.warehouse.import_facts("p1", [
                {"record_id": "r1", "revision": 2, "group_id": "h",
                 "inputs": {"a": "3"}, "complete": True},
            ])
        # Higher explicit complete fact may migrate.
        self.warehouse.import_facts("p1", [
            {"record_id": "r1", "revision": 5, "group_id": "h",
             "inputs": {"a": "3", "b": "4"}, "complete": True},
        ])
        self.assertEqual(self.warehouse.facts.effective()["r1"].group_id, "h")

    def test_complete_requires_all_declared_inputs(self):
        from exact_ledger import InvalidFactError
        with self.assertRaises(InvalidFactError):
            self.warehouse.import_facts("p1", [
                {"record_id": "r1", "revision": 5, "group_id": "h",
                 "inputs": {"a": "3"}, "complete": True},
            ])

    def test_facts_can_be_stored_incomplete(self):
        # Explicitly incomplete (subset) is legal to store, not coerced.
        self.warehouse.import_facts("p1", [
            {"record_id": "r9", "revision": 1, "group_id": "g",
             "inputs": {"a": "2"}, "complete": False},
        ])
        fact = self.warehouse.facts.effective()["r9"]
        self.assertEqual(dict(fact.inputs), {"a": Fraction(2)})
        self.assertFalse(fact.complete)

    def test_undeclared_input_rejected(self):
        with self.assertRaises(Exception):
            self.warehouse.import_facts("p1", [
                {"record_id": "rx", "revision": 1, "group_id": "g",
                 "inputs": {"zzz": "1"}},
            ])

    def test_previous_revision_ignored(self):
        self.correct(2, {"a": "3", "b": "10"})
        result = self.warehouse.publish("p1", ["w"])
        self.assertEqual(result.value("r1", "w"), Fraction(23))


class ResultIdentityTests(Fixture):
    def test_results_are_immutable_and_duplicate_ids_rejected(self):
        self.warehouse.publish("p1", ["z"], result_id="a")
        with self.assertRaises(DuplicateResultError):
            self.warehouse.publish("p1", ["z"], result_id="a")

    def test_old_results_keep_old_plan_and_facts(self):
        self.warehouse.publish("p1", ["z", "w"], result_id="initial")
        self.plans.publish_replacement("p2", "p1", {
            "node_id": "z",
            "terms": [
                {"ref": "u", "coefficient": "1"},
                {"ref": "b", "coefficient": "-2"},
            ],
        })
        self.correct(2, {"a": "3", "b": "10"})
        self.warehouse.publish("p2", ["z"], result_id="new")
        old = self.warehouse.reopen("initial")
        self.assertEqual(old.plan_id, "p1")
        self.assertEqual(old.total("g", "z"), Fraction(4))
        self.assertEqual(old.fact_snapshot["r1"]["revision"], 1)

    def test_incremental_recompute_reuses_and_rebuilds_totals(self):
        base = self.warehouse.publish("p1", ["z", "w"], result_id="base")
        self.correct(2, {"a": "3", "b": "10"})
        reduced = self.warehouse.incremental_recompute("base")
        self.assertEqual(reduced.parent_result_id, "base")
        self.assertEqual(reduced.total("g", "w"), Fraction(28))
        self.assertEqual(reduced.value("r2", "w"), base.value("r2", "w"))

    def test_incremental_recompute_picks_up_group_migration(self):
        self.warehouse.publish("p1", ["z"], result_id="base")
        self.warehouse.import_facts("p1", [
            {"record_id": "r1", "revision": 5, "group_id": "h",
             "inputs": {"a": "3", "b": "4"}, "complete": True},
        ])
        reduced = self.warehouse.incremental_recompute("base")
        self.assertEqual(reduced.total("g", "z"), Fraction(1))  # only r2 now
        self.assertEqual(reduced.total("h", "z"), Fraction(3))

    def test_rejected_result_is_distinct_identity(self):
        self.correct(3, {"a": "3"})
        result = self.warehouse.publish("p1", ["z", "w"], result_id="reject")
        self.assertEqual(result.status, "rejected")
        self.assertEqual(self.warehouse.reopen("reject").status, "rejected")
        self.assertEqual(
            self.warehouse.reopen("reject").plan_digest,
            self.plans.get("p1").digest,
        )


class ProvenanceTests(Fixture):
    def test_group_source_sums_and_zero_cancellation(self):
        result = self.warehouse.publish("p1", ["z"], result_id="z")
        self.assertEqual(result.total("g", "z"), Fraction(4))
        for record_id in ("r1", "r2"):
            entries = result.provenance_for(record_id=record_id, output="z")
            self.assertEqual(
                [(e["input"], e["is_constant"]) for e in entries],
                [("", True), ("a", False)],
            )
        # b never appears as a dependency of p1.z
        self.assertEqual(
            [e["input"] for e in result.trace("r1")["nonzero_contributions"]],
            ["a"],
        )

    def test_trace_follows_fact_revision_to_nonzero_contribution(self):
        self.correct(2, {"a": "3", "b": "10"})
        result = self.warehouse.publish("p1", ["w"], result_id="w")
        trace = result.trace("r1", input_name="b")
        self.assertEqual(trace["revision"], 2)
        nonzero = trace["nonzero_contributions"]
        self.assertEqual(len(nonzero), 1)
        self.assertEqual(nonzero[0]["value"], "10")
        self.assertEqual(nonzero[0]["coefficient"], "2")
        self.assertEqual(nonzero[0]["contribution"], "20")

    def test_group_sources_aggregate(self):
        result = self.warehouse.publish("p1", ["w"], result_id="w")
        sources = {
            (item["input"], item["is_constant"]): item["contribution"]
            for item in result.group_sources
        }
        # a: 1*(3+1)=4 ; b: 2*(4+2)=12 ; constant 0
        self.assertEqual(sources[("a", False)], "4")
        self.assertEqual(sources[("b", False)], "12")


if __name__ == "__main__":
    unittest.main()
