
import json
import sys
from pathlib import Path

from app.agent import answer


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "agent_trajectory_dataset.json"
REPORT = ROOT / "reports" / "trajectory_report.json"


def contains_expected(actual, expected):
    """Recursively check expected fields in actual data."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False

        return all(
            key in actual
            and contains_expected(actual[key], value)
            for key, value in expected.items()
        )

    if isinstance(expected, list):
        return actual == expected

    return actual == expected


def evaluate_case(case):
    response = answer(case["question"])
    trace = response.get("execution_trace", [])

    actual_tools = [step["tool"] for step in trace]
    expected_tools = case["expected_tools"]

    # Compare names without requiring an execution order.
    # Counter also detects duplicate or missing calls.
    from collections import Counter

    selection_pass = (
        Counter(actual_tools) == Counter(expected_tools)
    )

    # Ensure every tool request has a successful result.
    execution_pass = (
        bool(trace)
        and all(
            step["status"] == "SUCCESS"
            for step in trace
        )
    )

    arguments_pass = True
    results_pass = True

    expected_arguments = case.get(
        "expected_arguments", {}
    )

    expected_results = case.get(
        "expected_results", {}
    )

    for tool_name, expected in expected_arguments.items():
        matching_steps = [
            step for step in trace
            if step["tool"] == tool_name
        ]

        if len(matching_steps) != 1:
            arguments_pass = False
            continue

        if not contains_expected(
            matching_steps[0]["arguments"],
            expected
        ):
            arguments_pass = False

    for tool_name, expected in expected_results.items():
        matching_steps = [
            step for step in trace
            if step["tool"] == tool_name
        ]

        if len(matching_steps) != 1:
            results_pass = False
            continue

        if not contains_expected(
            matching_steps[0]["result"],
            expected
        ):
            results_pass = False

    checks = {
        "tool_selection": selection_pass,
        "tool_execution": execution_pass,
        "argument_validation": arguments_pass,
        "result_validation": results_pass,
    }

    return {
        "id": case["id"],
        "question": case["question"],
        "passed": all(checks.values()),
        "checks": checks,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "execution_trace": trace,
        "answer": response.get("answer", ""),
    }


def main():
    cases = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    results = []

    for case in cases:
        print(f"\nEvaluating {case['id']}")

        try:
            result = evaluate_case(case)

        except Exception as error:
            result = {
                "id": case["id"],
                "question": case["question"],
                "passed": False,
                "error": str(error),
            }

        results.append(result)

        print("PASS" if result["passed"] else "FAIL")

        for name, value in result.get("checks", {}).items():
            print(f"  {name}: {'PASS' if value else 'FAIL'}")

        if "error" in result:
            print("  Error:", result["error"])

    total = len(results)
    passed = sum(r["passed"] for r in results)
    failed = total - passed

    accuracy = (
        round(passed / total * 100, 2)
        if total else 0
    )

    report = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "results": results,
    }

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("AGENT TRAJECTORY EVALUATION")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Accuracy: {accuracy}%")
    print(f"Report: {REPORT}")

    if failed or not total:
        sys.exit(1)


if __name__ == "__main__":
    main()
