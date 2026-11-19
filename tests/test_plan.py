"""Tests for the exact linear plan service (no facts involved)."""

from __future__ import annotations

import unittest
from fractions import Fraction

from exact_ledger import (
    CycleError,
    DuplicateNameError,
    EmptyIdentifierError,
    PlanService,
    UnknownReferenceError,
    as_fraction,
    to_str,
)
from exact_ledger.plan import InvalidPlanError


def plan_def(**overrides):
    definition = {
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
    definition.update(overrides)
    return definition


class RationalTests(unittest.TestCase):
    def test_exact_parsing(self):
        self.assertEqual(as_fraction("3"), Fraction(3))
        self.assertEqual(as_fraction("-3/4"), Fraction(-3, 4))
        self.assertEqual(as_fraction("0.125"), Fraction(1, 8))
        self.assertEqual(as_fraction(" 3 / 4 "), Fraction(3, 4))
        self.assertEqual(to_str(Fraction(4, 2)), "2")
        self.assertEqual(to_str(Fraction(-3, 4)), "-3/4")

    def test_rejects_floats_and_junk(self):
        for bad in (1.5, "1e3", "nan", "inf", "0x10", True, "", "1/0"):
            with self.assertRaises(Exception):
                as_fraction(bad)


class PlanAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.service = PlanService()
        self.plan = self.service.publish(plan_def())

    def test_zero_dependency_is_eliminated(self):
        z = self.plan.contract("z")
        self.assertEqual(z.input_coefficients, {"a": Fraction(1)})
        self.assertFalse(z.depends_on("b"))
        self.assertEqual(z.coefficient_of("b"), 0)

    def test_shared_node_not_double_counted(self):
        w = self.plan.contract("w")
        self.assertEqual(w.input_coefficients, {"a": Fraction(1), "b": Fraction(2)})

    def test_contract_is_verifiable_from_raw_node(self):
        z = self.plan.contract("z")
        self.assertEqual(z.derived_from["node_id"], "z")
        self.assertEqual(
            z.derived_from["terms"],
            [{"ref": "u", "coefficient": "1"}, {"ref": "b", "coefficient": "-1"}],
        )
        # The published node product for z has b removed.
        self.assertEqual(dict(self.plan.node_products["z"]), {"a": Fraction(1)})

    def test_names_are_not_merged_by_text(self):
        service = PlanService()
        plan = service.publish({
            "plan_id": "text",
            "inputs": ["a", "aa", "a_a"],
            "nodes": [{"node_id": "o", "terms": [
                {"ref": "a", "coefficient": "1"},
                {"ref": "aa", "coefficient": "1"},
                {"ref": "a_a", "coefficient": "-1"},
            ]}],
            "outputs": ["o"],
        })
        self.assertEqual(
            plan.contract("o").input_coefficients,
            {"a": Fraction(1), "aa": Fraction(1), "a_a": Fraction(-1)},
        )

    def test_duplicate_terms_combine(self):
        service = PlanService()
        plan = service.publish({
            "plan_id": "dup",
            "inputs": ["a"],
            "nodes": [{"node_id": "o", "terms": [
                {"ref": "a", "coefficient": "2"},
                {"ref": "a", "coefficient": "-2"},
                {"ref": "a", "coefficient": "5/2"},
            ]}],
            "outputs": ["o"],
        })
        self.assertEqual(plan.contract("o").input_coefficients, {"a": Fraction(5, 2)})

    def test_constant_and_output_on_input(self):
        service = PlanService()
        plan = service.publish({
            "plan_id": "k",
            "inputs": ["a"],
            "nodes": [{"node_id": "c", "terms": [], "constant": "7/3"}],
            "outputs": ["c", "a"],
        })
        self.assertEqual(plan.contract("c").constant, Fraction(7, 3))
        self.assertEqual(plan.contract("c").input_coefficients, {})
        self.assertEqual(plan.contract("a").input_coefficients, {"a": Fraction(1)})

    def test_empty_ids_rejected(self):
        for bad in ("", "   "):
            with self.assertRaises(EmptyIdentifierError):
                self.service.publish(plan_def(inputs=["a", bad]))

    def test_unknown_reference_rejected(self):
        with self.assertRaises(UnknownReferenceError):
            self.service.publish(plan_def(nodes=[
                {"node_id": "o", "terms": [{"ref": "ghost", "coefficient": "1"}]},
            ], outputs=["o"]))

    def test_cycle_rejected(self):
        with self.assertRaises(CycleError):
            self.service.publish(plan_def(
                inputs=[],
                nodes=[
                    {"node_id": "x", "terms": [{"ref": "y", "coefficient": "1"}]},
                    {"node_id": "y", "terms": [{"ref": "x", "coefficient": "1"}]},
                ],
                outputs=["x"],
            ))

    def test_duplicate_node_and_input_collision(self):
        with self.assertRaises(DuplicateNameError):
            self.service.publish(plan_def(inputs=["a", "a"]))
        with self.assertRaises(DuplicateNameError):
            self.service.publish(plan_def(nodes=[
                {"node_id": "a", "terms": []},
                {"node_id": "a", "terms": []},
            ]))

    def test_immutable_plan(self):
        again = self.service.publish(plan_def())
        self.assertIs(again, self.plan)
        changed = plan_def(outputs=["z"])
        with self.assertRaises(InvalidPlanError):
            self.service.publish(changed)

    def test_replacement_preserves_other_nodes(self):
        updated = self.service.publish_replacement("p2", "p1", {
            "node_id": "z",
            "terms": [
                {"ref": "u", "coefficient": "1"},
                {"ref": "b", "coefficient": "-2"},
            ],
        })
        self.assertEqual(
            updated.contract("z").input_coefficients,
            {"a": Fraction(1), "b": Fraction(-1)},
        )
        self.assertEqual(
            updated.contract("w").input_coefficients,
            {"a": Fraction(1), "b": Fraction(2)},
        )
        # original unchanged
        self.assertEqual(
            self.plan.contract("z").input_coefficients, {"a": Fraction(1)}
        )
        self.assertNotEqual(updated.digest, self.plan.digest)

    def test_explain_coefficient_shows_cancellation(self):
        explanation = self.service.explain_coefficient("p1", "z", "b")
        self.assertEqual(explanation["coefficient"], "0")
        self.assertEqual(explanation["state"], "cancelled")
        self.assertEqual(
            explanation["sum_of_contributions"], explanation["coefficient"]
        )
        weights = [c["weight"] for c in explanation["contributions"]]
        self.assertEqual(sorted(weights), ["-1", "1"])
        self.assertEqual(
            sorted(c["edge"] for c in explanation["contributions"]),
            ["u->b", "z->b"],
        )

    def test_explain_absent_vs_cancelled(self):
        service = PlanService()
        plan = service.publish({
            "plan_id": "abs",
            "inputs": ["a", "b"],
            "nodes": [{"node_id": "o", "terms": [
                {"ref": "a", "coefficient": "1"},
            ]}],
            "outputs": ["o"],
        })
        absent = service.explain_coefficient("abs", "o", "b")
        self.assertEqual(absent["state"], "absent")
        self.assertEqual(absent["zero_reason"], "no-path")
        self.assertEqual(absent["contributions"], [])
        cancelled = self.service.explain_coefficient("p1", "z", "b")
        self.assertEqual(cancelled["zero_reason"], "cancelled")

    def test_explanation_is_polynomial_on_wide_dag(self):
        # 40 chained diamond shared nodes: a path-enumerating explainer would
        # blow up, the reverse topological pass must not.
        nodes = [{"node_id": "n0", "terms": [
            {"ref": "a", "coefficient": "1"},
            {"ref": "b", "coefficient": "1"},
        ]}]
        for index in range(1, 40):
            previous = f"n{index - 1}"
            nodes.append({"node_id": f"n{index}", "terms": [
                {"ref": previous, "coefficient": "1"},
                {"ref": previous, "coefficient": "-1"},
                {"ref": previous, "coefficient": "1"},
            ]})
        nodes.append({"node_id": "out", "terms": [
            {"ref": "n39", "coefficient": "1"},
            {"ref": "b", "coefficient": "-1"},
        ]})
        plan = self.service.publish({
            "plan_id": "wide",
            "inputs": ["a", "b"],
            "nodes": nodes,
            "outputs": ["out"],
        })
        self.assertEqual(plan.contract("out").input_coefficients, {"a": Fraction(1)})
        explanation = self.service.explain_coefficient("wide", "out", "b")
        self.assertEqual(explanation["coefficient"], "0")
        self.assertEqual(
            explanation["sum_of_contributions"], explanation["coefficient"]
        )

    def test_repeated_dependency_paths_are_counted_once_per_edge(self):
        """Diamond: a shared node referenced twice composes exactly."""
        service = PlanService()
        plan = service.publish({
            "plan_id": "diamond",
            "inputs": ["a"],
            "nodes": [
                {"node_id": "s", "terms": [{"ref": "a", "coefficient": "1"}]},
                {"node_id": "o", "terms": [
                    {"ref": "s", "coefficient": "3"},
                    {"ref": "s", "coefficient": "-3"},
                    {"ref": "a", "coefficient": "1"},
                ]},
            ],
            "outputs": ["o"],
        })
        self.assertEqual(plan.contract("o").input_coefficients, {"a": Fraction(1)})

    def test_large_exact_values_no_float(self):
        service = PlanService()
        huge = Fraction(999999, 1000000)
        plan = service.publish({
            "plan_id": "big",
            "inputs": ["a"],
            "nodes": [{"node_id": "o", "terms": [
                {"ref": "a", "coefficient": "999999/1000000"},
                {"ref": "a", "coefficient": "1/1000000"},
            ]}],
            "outputs": ["o"],
        })
        self.assertEqual(plan.contract("o").coefficient_of("a"), Fraction(1))


if __name__ == "__main__":
    unittest.main()
