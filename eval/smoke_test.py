"""
CI smoke test for the retrieval pipeline.

Unlike run_eval.py, this does NOT measure retrieval quality (that requires
the real dataset, which is intentionally not published/available in CI). It
only proves legalis_api/main.py imports, starts, and returns correctly-shaped
results against tiny synthetic fixture data — catching import errors, crashes,
and schema breakage on every PR without exposing real data.

Usage:
    python eval/smoke_test.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)


def main():
    os.chdir(REPO_ROOT / "legalis_api")
    sys.path.insert(0, str(REPO_ROOT / "legalis_api"))
    import main as api  # noqa: E402  (import after path/cwd setup, deliberately)

    check(len(api.cases_data) > 0, "no fixture case data loaded")
    check(len(api.faq_data) > 0, "no fixture FAQ data loaded")

    case_results = api.find_relevant_cases("fixture test query", num_results=2)
    check(len(case_results) > 0, "find_relevant_cases returned no results")
    for key in (
        "case_id",
        "case_title",
        "case_link",
        "similarity_score",
        "sections",
        "strong_points",
        "weak_points",
    ):
        check(key in case_results[0], f"case result missing expected key: {key}")

    faq_results = api.find_relevant_faq("fixture test query", num_results=2)
    check(len(faq_results) > 0, "find_relevant_faq returned no results")
    for key in ("faq_prompt", "faq_completion", "similarity_score"):
        check(key in faq_results[0], f"FAQ result missing expected key: {key}")

    print(
        "Smoke test passed: retrieval pipeline imports, starts, and returns correctly-shaped results."
    )


if __name__ == "__main__":
    main()
