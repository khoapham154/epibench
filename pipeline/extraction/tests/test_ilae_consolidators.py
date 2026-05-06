"""Regression tests for ILAE consolidators (v4).

Locks in the 13 audit cases Codex flagged for ez_localization, plus
smoke coverage for the new status_epilepticus task.

Run:
    pytest pipeline/tests/test_ilae_consolidators.py -v
or
    python -m pytest pipeline/tests/test_ilae_consolidators.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make pipeline importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # type: ignore  # noqa: E402

from ilae_label_maps import (  # noqa: E402
    consolidate_ez_localization,
    consolidate_status_epilepticus,
    consolidate_surgery_outcome,
    consolidate_aed_response,
    consolidate_epilepsy_type,
    consolidate_seizure_type,
    LABELS,
)


# ---------- ez_localization: 13 audit cases ----------


@pytest.mark.parametrize("raw, expected", [
    ("bilateral temporal", "Multifocal"),
    ("Bilateral frontal", "Multifocal"),
    ("left fronto-temporal", "Multifocal"),
    ("right temporo-parietal", "Multifocal"),
    ("Temporal-parietal-occipital", "Multifocal"),
    ("left hemisphere", "Hemispheric"),
    ("right hemispheric", "Hemispheric"),
    ("mesial temporal", "Temporal"),
    ("left temporal", "Temporal"),
    ("left frontal", "Extratemporal"),
    ("right parietal", "Extratemporal"),
    ("Opercular", "Extratemporal"),
    ("Frontobasal", "Extratemporal"),
    ("cingulate", "Extratemporal"),
    ("Focal", "Unknown"),
    ("Left", "Unknown"),
    ("Right", "Unknown"),
    ("cryptogenic", "Unknown"),
    ("multifocal", "Multifocal"),
    ("", None),
    ("none", None),
])
def test_ez_localization(raw, expected):
    assert consolidate_ez_localization(raw) == expected


# ---------- status_epilepticus smoke coverage ----------


@pytest.mark.parametrize("raw, expected", [
    ("refractory status epilepticus", "Refractory SE"),
    ("RSE developed during admission", "Refractory SE"),
    ("super-refractory status", "Refractory SE"),
    ("non-convulsive status epilepticus", "Non-convulsive SE"),
    ("NCSE on EEG", "Non-convulsive SE"),
    ("absence status", "Non-convulsive SE"),
    ("convulsive status epilepticus", "Convulsive SE"),
    ("generalized tonic-clonic status", "Convulsive SE"),
    ("GCSE", "Convulsive SE"),
    ("no history of status epilepticus", "None"),
    ("without status epilepticus", "None"),
    ("status epilepticus reported", "Unknown"),
    ("", None),
    ("not mentioned", None),
])
def test_status_epilepticus(raw, expected):
    assert consolidate_status_epilepticus(raw) == expected


# ---------- schema sanity: survival is gone, status_epilepticus present ----------


def test_schema_replaced_survival_with_status_epilepticus():
    assert "survival" not in LABELS
    assert "status_epilepticus" in LABELS
    assert set(LABELS["status_epilepticus"]) == {
        "Convulsive SE", "Non-convulsive SE", "Refractory SE", "None", "Unknown",
    }


# ---------- basic smoke on other consolidators (unchanged but verify still pass) ----------


def test_aed_response_smoke():
    assert consolidate_aed_response("drug-resistant") == "drug-resistant"
    assert consolidate_aed_response("refractory epilepsy") == "drug-resistant"
    assert consolidate_aed_response("seizure-free for 2 years") == "drug-responsive"
    assert consolidate_aed_response("taking AED") == "unspecified"


def test_surgery_outcome_smoke():
    assert consolidate_surgery_outcome("Engel I") == "Seizure-free"
    assert consolidate_surgery_outcome("Engel class III") == "Improved"
    assert consolidate_surgery_outcome("Engel IV") == "No improvement"
    assert consolidate_surgery_outcome("no surgery") == "Not applicable"


def test_epilepsy_type_smoke():
    assert consolidate_epilepsy_type("focal") == "Focal"
    assert consolidate_epilepsy_type("generalized") == "Generalised"
    assert consolidate_epilepsy_type("JME") == "Generalised"
    assert consolidate_epilepsy_type("TLE") == "Focal"


def test_seizure_type_smoke():
    assert consolidate_seizure_type("focal onset") == "Focal"
    assert consolidate_seizure_type("tonic-clonic") == "Generalised"
    assert consolidate_seizure_type("unclassified") == "Unclassified"
