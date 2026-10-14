
import json
import sys
from pathlib import Path

from app.agent import answer


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR / "data" / "agent_tool_dataset.json"
)


def load_dataset():
    """Load golden test cases."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_tool_selection():
    """
    Evaluate LangGraph agent tool selection
    against expected tool names.
    """

    dataset = load_dataset()

    total = len(dataset)
    passed = 0
    failed = 0
    results = []

    for index, test_case in enumerate(dataset, start=1):

        question = test_case["question"]
        expected_tools = set(
            test_case["expected_tools"]
        )

        print("\n" + "=" * 70)
        print(f"TEST CASE: {index}")
        print(f"QUESTION: {question}")

        try:
            # Execute the LangGraph agent
            response = answer(question)

            # Extract tool calls
            tool_calls = response.get("tool_calls", [])

            actual_tools = {
                call["tool"]
                for call in tool_calls
            }

            # Compare actual vs expected
            is_passed = actual_tools == expected_tools

            if is_passed:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            result = {
                "question": question,
                "expected_tools": sorted(expected_tools),
                "actual_tools": sorted(actual_tools),
                "status": status,
                "answer": response.get("answer", ""),
            }

        except Exception as error:
            failed += 1
            status = "ERROR"

            result = {
                "question": question,
                "expected_tools": sorted(expected_tools),
                "actual_tools": [],
                "status": status,
                "error": str(error),
            }

        results.append(result)

        print(f"EXPECTED TOOLS: {result['expected_tools']}")
        print(f"ACTUAL TOOLS:   {result['actual_tools']}")
        print(f"STATUS:         {status}")

        if "error" in result:
            print(f"ERROR: {result['error']}")

    # Calculate overall accuracy
    accuracy = (
        (passed / total) * 100
        if total > 0
        else 0
    )

    print("\n" + "=" * 70)
    print("LANGGRAPH TOOL SELECTION EVALUATION")
    print("=" * 70)

    print(f"Total Tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Accuracy    : {accuracy:.2f}%")

    # Save evaluation report
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_path = (
        reports_dir / "tool_selection_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "accuracy": accuracy,
                "results": results,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nReport saved: {report_path}")

    # Fail CI/CD pipeline when tests fail
    if failed > 0 or total == 0:
        sys.exit(1)

    return results


if __name__ == "__main__":
    evaluate_tool_selection()
