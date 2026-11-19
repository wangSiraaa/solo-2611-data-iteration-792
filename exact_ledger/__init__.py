"""Independent exact linear derivation plan service and grouped fact replay
warehouse.

Two deliberately separate components live here:

* :mod:`exact_ledger.plan` -- immutable DAG plans, exact algebra, published
  contracts.  Knows nothing about facts.
* :mod:`exact_ledger.replay` (with :mod:`exact_ledger.facts`) -- fact
  versions, groups, eligibility and immutable replay results.  Consumes the
  plan service's published products.

Typical use::

    from exact_ledger import PlanService, ReplayWarehouse, open_warehouse

    plans = PlanService()
    plans.publish(plan_definition)
    warehouse = open_warehouse(plans)            # in-memory SQLite
    warehouse.import_facts("p1", facts)
    report = warehouse.inspect("p1", ["z", "w"])
    result = warehouse.publish("p1", ["z"], result_id="initial")
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from .facts import (
    FactError,
    FactStore,
    GroupMigrationError,
    InvalidFactError,
    RevisionConflictError,
    StoredFact,
)
from .plan import (
    CycleError,
    DuplicateNameError,
    EmptyIdentifierError,
    InvalidPlanError,
    Node,
    OutputContract,
    Plan,
    PlanError,
    PlanService,
    Term,
    UnknownReferenceError,
    explain_coefficient,
)
from .rational import RationalError, as_fraction, to_str
from .replay import (
    DuplicateResultError,
    EligibilityReport,
    InvalidRequestError,
    PublicationError,
    ReplayError,
    ReplayWarehouse,
    Result,
    UnknownOutputError,
    UnknownResultError,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "open_warehouse",
    "open_connection",
    # rational
    "RationalError",
    "as_fraction",
    "to_str",
    # plan
    "PlanService",
    "Plan",
    "Node",
    "Term",
    "OutputContract",
    "PlanError",
    "InvalidPlanError",
    "EmptyIdentifierError",
    "DuplicateNameError",
    "UnknownReferenceError",
    "CycleError",
    "explain_coefficient",
    # facts / replay
    "ReplayWarehouse",
    "FactStore",
    "StoredFact",
    "Result",
    "EligibilityReport",
    "ReplayError",
    "PublicationError",
    "DuplicateResultError",
    "UnknownResultError",
    "UnknownOutputError",
    "InvalidRequestError",
    "FactError",
    "InvalidFactError",
    "RevisionConflictError",
    "GroupMigrationError",
]


def open_connection(path: str = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def open_warehouse(
    plans: Optional[PlanService] = None,
    connection: Optional[sqlite3.Connection] = None,
    path: str = ":memory:",
) -> ReplayWarehouse:
    """Convenience constructor wiring a plan service to a SQLite warehouse."""
    plans = plans or PlanService()
    connection = connection or open_connection(path)
    return ReplayWarehouse(connection, plans)
