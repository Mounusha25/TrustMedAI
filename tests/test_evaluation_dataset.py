import json
from collections import Counter
from pathlib import Path


DATASET = Path(__file__).resolve().parents[1] / "TrustMed-AI" / "evaluation_dataset_100.json"


def load_dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_evaluation_dataset_matches_declared_scope():
    dataset = load_dataset()
    metadata = dataset["metadata"]
    cases = dataset["single_turn_queries"]

    assert metadata["total_queries"] == 100
    assert metadata["queries_per_domain"] == 20
    assert len(metadata["domains"]) == 5
    assert len(cases) == metadata["total_queries"]
    assert Counter(case["domain"] for case in cases) == {
        domain: metadata["queries_per_domain"] for domain in metadata["domains"]
    }


def test_evaluation_cases_have_unique_grounded_expectations():
    cases = load_dataset()["single_turn_queries"]
    queries = [case["query"] for case in cases]

    assert len(queries) == len(set(queries))
    for case in cases:
        assert case["query"].strip()
        assert case["topic"].strip()
        assert case["difficulty"] in {"easy", "medium", "hard"}
        assert case["expected_keywords"]
        assert all(keyword.strip() for keyword in case["expected_keywords"])
        assert case["expected_collections"]
