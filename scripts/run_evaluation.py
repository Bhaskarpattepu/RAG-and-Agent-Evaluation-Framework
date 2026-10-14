import json
from pathlib import Path
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from app.agent import answer
from app.rag import retrieve

cases = []
for row in json.loads(Path("data/golden_dataset.json").read_text()):
    retrieved = retrieve(row["input"])
    response = answer(row["input"])
    cases.append(LLMTestCase(
        input=row["input"],
        actual_output=response["answer"],
        expected_output=row["expected_output"],
        retrieval_context=[item["text"] for item in retrieved] or ["No documents retrieved"],
    ))
    print(json.dumps({"question": row["input"], "sources": response["sources"],
                      "used_tool": response["used_tool"]}))

evaluate(test_cases=cases, metrics=[FaithfulnessMetric(threshold=0.7),
                                    AnswerRelevancyMetric(threshold=0.7)])
