"""Grouped fact storage (SQLite).

Facts are keyed by ``record_id`` and carry a positive integer ``revision``, a
``group_id`` and a **subset** of the plan's declared inputs.  The store
enforces the replay rules:

* a missing input means *unavailable* -- it is stored as absent, never as 0;
* the highest revision of a record is the effective one;
* re-importing the same ``(record_id, revision)`` with the same canonical
  content is an idempotent replay;
* importing different content for an existing ``(record_id, revision)``
  rejects the whole batch and writes nothing;
* ``group_id`` may only change in a higher revision that is an explicit
  ``complete`` fact.

The store knows nothing about plan values or outputs -- it only owns fact
identity, versions and groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import sqlite3
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .rational import as_fraction, to_str

__all__ = [
    "FactError",
    "EmptyIdentifierError",
    "InvalidFactError",
    "RevisionConflictError",
    "GroupMigrationError",
    "StoredFact",
    "FactStore",
    "MAX_RECORDS",
    "content_hash",
]

# Documented scale limit for one fact view.
MAX_RECORDS = 5000


class FactError(ValueError):
    """Base class for rejected facts."""


class EmptyIdentifierError(FactError):
    pass


class InvalidFactError(FactError):
    pass


class RevisionConflictError(FactError):
    """Same ``(record_id, revision)`` with different canonical content."""


class GroupMigrationError(FactError):
    """Attempt to change a record's group without a higher complete fact."""


def frac_pair(value: Fraction) -> Tuple[str, str]:
    return str(value.numerator), str(value.denominator)


def frac_from_pair(num: str, den: str) -> Fraction:
    return Fraction(int(num), int(den))


def canonical_inputs(inputs: Mapping[str, Fraction]) -> Dict[str, str]:
    return {name: to_str(value) for name, value in sorted(inputs.items())}


def content_hash(
    group_id: str,
    inputs: Mapping[str, Fraction],
    complete: bool,
    explicit_complete: bool = False,
) -> str:
    """Hash of the *canonical* fact content used to detect replay conflicts."""
    payload = {
        "group_id": group_id,
        "complete": complete,
        "explicit_complete": explicit_complete,
        "inputs": canonical_inputs(inputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require_name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmptyIdentifierError(f"{what} must be a non-empty string, got {value!r}")
    return value


def require_revision(value: object) -> int:
    if isinstance(value, bool):
        raise InvalidFactError("revision must be a positive integer, not a boolean")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdigit():
            raise InvalidFactError(
                f"revision must be a positive integer, got {value!r}"
            )
        value = int(stripped)
    if not isinstance(value, int) or value < 1:
        raise InvalidFactError(f"revision must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class StoredFact:
    record_id: str
    revision: int
    group_id: str
    inputs: Mapping[str, Fraction]
    complete: bool
    explicit_complete: bool = False

    @property
    def content_hash(self) -> str:
        return content_hash(
            self.group_id, self.inputs, self.complete, self.explicit_complete
        )


class FactStore:
    """Append-only, SQLite-backed revision store with strict import rules."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                record_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                group_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                is_complete INTEGER NOT NULL,
                explicit_complete INTEGER NOT NULL DEFAULT 0,
                imported_seq INTEGER NOT NULL,
                PRIMARY KEY (record_id, revision)
            );
            CREATE TABLE IF NOT EXISTS fact_inputs (
                record_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                name TEXT NOT NULL,
                num TEXT NOT NULL,
                den TEXT NOT NULL,
                PRIMARY KEY (record_id, revision, name),
                FOREIGN KEY (record_id, revision)
                    REFERENCES facts (record_id, revision) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS facts_record_idx
                ON facts (record_id, revision);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------- reading
    def _load(self, row: sqlite3.Row) -> StoredFact:
        inputs = {
            item["name"]: frac_from_pair(item["num"], item["den"])
            for item in self._conn.execute(
                "SELECT name, num, den FROM fact_inputs "
                "WHERE record_id = ? AND revision = ? ORDER BY name",
                (row["record_id"], row["revision"]),
            )
        }
        return StoredFact(
            record_id=row["record_id"],
            revision=row["revision"],
            group_id=row["group_id"],
            inputs=inputs,
            complete=bool(row["is_complete"]),
            explicit_complete=bool(row["explicit_complete"]),
        )

    def all_revisions(self, record_id: str) -> List[StoredFact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE record_id = ? ORDER BY revision",
            (record_id,),
        ).fetchall()
        return [self._load(row) for row in rows]

    def get_revision(self, record_id: str, revision: int) -> Optional[StoredFact]:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE record_id = ? AND revision = ?",
            (record_id, revision),
        ).fetchone()
        return self._load(row) if row else None

    def effective(self) -> Dict[str, StoredFact]:
        rows = self._conn.execute(
            """
            SELECT f.* FROM facts f
            JOIN (
                SELECT record_id, MAX(revision) AS revision
                FROM facts GROUP BY record_id
            ) latest
              ON f.record_id = latest.record_id
             AND f.revision = latest.revision
            ORDER BY f.record_id
            """
        ).fetchall()
        return {row["record_id"]: self._load(row) for row in rows}

    def record_ids(self) -> List[str]:
        return [
            row["record_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT record_id FROM facts ORDER BY record_id"
            )
        ]

    def snapshot(self) -> Dict[str, object]:
        """A stable description of every effective fact, for result provenance."""
        snap: Dict[str, object] = {}
        for record_id, fact in self.effective().items():
            snap[record_id] = {
                "revision": fact.revision,
                "group_id": fact.group_id,
                "complete": fact.complete,
                "inputs": canonical_inputs(fact.inputs),
            }
        return snap

    # ------------------------------------------------------------- writing
    def prepare(
        self, raw_facts: Sequence[object], declared_inputs: Iterable[str]
    ) -> List[StoredFact]:
        """Validate an entire batch, returning rows ready to write.

        Nothing is written here.  Raises if any row is malformed, conflicts
        with an existing revision, or attempts an illegal group migration, so
        a rejected batch never partially lands.
        """
        declared = set(declared_inputs)
        prepared: List[StoredFact] = []
        batch_keys: Dict[Tuple[str, int], StoredFact] = {}

        for index, raw in enumerate(raw_facts):
            fact = self._parse(raw, declared, index)
            key = (fact.record_id, fact.revision)
            previous = batch_keys.get(key)
            if previous is not None:
                if previous.content_hash != fact.content_hash:
                    raise RevisionConflictError(
                        f"batch contains revision {fact.revision} of record "
                        f"{fact.record_id!r} with conflicting content"
                    )
                continue
            batch_keys[key] = fact
            prepared.append(fact)

        if len({fact.record_id for fact in prepared} | set(self.record_ids())) > MAX_RECORDS:
            raise InvalidFactError(
                f"fact view would exceed {MAX_RECORDS} records"
            )

        # Project effective state forward in revision order per record.
        projected: Dict[str, Tuple[int, str, bool]] = {
            record_id: (fact.revision, fact.group_id, fact.complete)
            for record_id, fact in self.effective().items()
        }
        for fact in sorted(prepared, key=lambda f: (f.record_id, f.revision)):
            existing = self.get_revision(fact.record_id, fact.revision)
            if existing is not None:
                if existing.content_hash != fact.content_hash:
                    raise RevisionConflictError(
                        f"record {fact.record_id!r} revision {fact.revision} "
                        "already exists with different canonical content; "
                        "the entire batch is rejected"
                    )
                continue  # idempotent replay of an identical revision
            current = projected.get(fact.record_id)
            if current is not None:
                current_revision, current_group, _ = current
                if fact.group_id != current_group:
                    # Group changes need an *explicitly* higher complete fact.
                    if not (
                        fact.revision > current_revision
                        and fact.explicit_complete
                    ):
                        raise GroupMigrationError(
                            f"record {fact.record_id!r} may change group from "
                            f"{current_group!r} to {fact.group_id!r} only in a "
                            "higher revision that is an explicit complete fact"
                        )
            if current is None or fact.revision > current[0]:
                projected[fact.record_id] = (
                    fact.revision,
                    fact.group_id,
                    fact.complete,
                )
        return prepared

    def _parse(self, raw: object, declared: set, index: int) -> StoredFact:
        if not isinstance(raw, Mapping):
            raise InvalidFactError(f"fact {index} must be an object")
        record_id = require_name(raw.get("record_id"), f"fact {index} record_id")
        revision = require_revision(raw.get("revision"))
        group_id = require_name(raw.get("group_id"), f"fact {index} group_id")
        raw_inputs = raw.get("inputs", {})
        if not isinstance(raw_inputs, Mapping):
            raise InvalidFactError(f"fact {index} inputs must be an object")
        inputs: Dict[str, Fraction] = {}
        for name, value in raw_inputs.items():
            name = require_name(name, f"fact {index} input name")
            if name not in declared:
                raise InvalidFactError(
                    f"fact {index} (record {record_id!r}) supplies input "
                    f"{name!r}, which is not declared by the plan"
                )
            try:
                inputs[name] = as_fraction(value)
            except Exception as exc:  # noqa: BLE001
                raise InvalidFactError(
                    f"fact {index} input {name!r}: {exc}"
                ) from exc
        explicit_complete = False
        if "complete" in raw:
            if not isinstance(raw["complete"], bool):
                raise InvalidFactError(
                    f"fact {index} 'complete' must be a boolean"
                )
            explicit_complete = raw["complete"]
            if explicit_complete and set(inputs) != declared:
                missing = sorted(declared - set(inputs))
                raise InvalidFactError(
                    f"fact {index} claims complete but omits declared input(s) "
                    f"{missing}"
                )
        # Missing inputs are legal and stored as absent (never zero); whether
        # the subset happens to cover every declared input is recorded too.
        complete = set(inputs) == declared
        return StoredFact(
            record_id=record_id,
            revision=revision,
            group_id=group_id,
            inputs=inputs,
            complete=complete,
            explicit_complete=explicit_complete,
        )

    def write(self, facts: Sequence[StoredFact]) -> int:
        """Write validated rows atomically; returns the number added."""
        added = 0
        with self._conn:
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(imported_seq), 0) FROM facts"
            ).fetchone()[0]
            for fact in facts:
                if self.get_revision(fact.record_id, fact.revision) is not None:
                    continue
                seq += 1
                self._conn.execute(
                    "INSERT INTO facts (record_id, revision, group_id, "
                    "content_hash, is_complete, explicit_complete, imported_seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        fact.record_id,
                        fact.revision,
                        fact.group_id,
                        fact.content_hash,
                        int(fact.complete),
                        int(fact.explicit_complete),
                        seq,
                    ),
                )
                for name, value in sorted(fact.inputs.items()):
                    num, den = frac_pair(value)
                    self._conn.execute(
                        "INSERT INTO fact_inputs "
                        "(record_id, revision, name, num, den) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (fact.record_id, fact.revision, name, num, den),
                    )
                added += 1
        return added

    def import_facts(
        self, raw_facts: Sequence[object], declared_inputs: Iterable[str]
    ) -> int:
        """Validate and atomically import a batch; returns rows added."""
        prepared = self.prepare(raw_facts, declared_inputs)
        return self.write(prepared)
