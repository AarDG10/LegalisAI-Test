"""
Retrieval quality eval for LegalisAI.

Loads eval/case_eval_set.json and eval/faq_eval_set.json (hand-labeled query ->
expected result mappings grounded in the real dataset), runs each query
through the retrieval functions in legalis_api/main.py, and reports
Hit@1/3/5 and MRR. Run this before and after any embedding/model change to
measure whether retrieval quality actually improved.

Usage:
    python eval/run_eval.py
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
K_VALUES = (1, 3, 5)
TOP_N = 10  # ask retrieval for more results than the largest K we score


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reciprocal_rank(ranked_ids, relevant_ids):
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def score(eval_set, get_ranked_ids, id_field_name):
    results = []
    for item in eval_set:
        query = item["query"]
        relevant = set(item[id_field_name])
        ranked_ids = get_ranked_ids(query)

        rr = reciprocal_rank(ranked_ids, relevant)
        row = {"query": query, "reciprocal_rank": rr}
        for k in K_VALUES:
            row[f"hit@{k}"] = bool(relevant & set(ranked_ids[:k]))
        results.append(row)
    return results


def summarize(results):
    n = len(results)
    summary = {"n_queries": n, "MRR": sum(r["reciprocal_rank"] for r in results) / n}
    for k in K_VALUES:
        summary[f"hit_rate@{k}"] = sum(1 for r in results if r[f"hit@{k}"]) / n
    return summary


def print_table(title, results):
    print(f"\n=== {title} ===")
    header = f"{'Query':<70} " + " ".join(f"H@{k:<4}" for k in K_VALUES) + "  RR"
    print(header)
    print("-" * len(header))
    for r in results:
        q = r["query"][:67] + "..." if len(r["query"]) > 70 else r["query"]
        hits = " ".join(f"{'Y' if r[f'hit@{k}'] else 'n':<5}" for k in K_VALUES)
        print(f"{q:<70} {hits}  {r['reciprocal_rank']:.2f}")


def print_summary(title, summary):
    print(f"\n--- {title} summary ---")
    print(f"n_queries: {summary['n_queries']}")
    print(f"MRR:       {summary['MRR']:.3f}")
    for k in K_VALUES:
        print(f"Hit@{k}:     {summary[f'hit_rate@{k}']:.3f}")


def main():
    # legalis_api/main.py uses relative paths ("../legalis_model", "../Data/...")
    # so it must be imported with legalis_api/ as the working directory.
    os.chdir(REPO_ROOT / "legalis_api")
    sys.path.insert(0, str(REPO_ROOT / "legalis_api"))
    import main as api  # noqa: E402  (import after path/cwd setup, deliberately)

    case_eval_set = load_json(REPO_ROOT / "eval" / "case_eval_set.json")
    faq_eval_set = load_json(REPO_ROOT / "eval" / "faq_eval_set.json")

    def get_case_ids(query):
        return [r["case_id"] for r in api.find_relevant_cases(query, num_results=TOP_N)]

    def get_faq_prompts(query):
        return [
            r["faq_prompt"] for r in api.find_relevant_faq(query, num_results=TOP_N)
        ]

    case_results = score(case_eval_set, get_case_ids, "relevant_case_ids")
    faq_results = score(faq_eval_set, get_faq_prompts, "relevant_prompts")

    case_summary = summarize(case_results)
    faq_summary = summarize(faq_results)

    print_table("Case retrieval", case_results)
    print_summary("Case retrieval", case_summary)

    print_table("FAQ retrieval", faq_results)
    print_summary("FAQ retrieval", faq_summary)

    out_path = REPO_ROOT / "eval" / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_retrieval": {"summary": case_summary, "per_query": case_results},
                "faq_retrieval": {"summary": faq_summary, "per_query": faq_results},
            },
            f,
            indent=2,
        )
    print(f"\nFull results written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
