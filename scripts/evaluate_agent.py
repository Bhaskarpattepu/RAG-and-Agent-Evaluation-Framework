import json
from pathlib import Path
from app.agent import answer

rows = json.loads(Path("data/golden_dataset.json").read_text())
passed = 0
for row in rows:
    result = answer(row["input"])
    ok = result["used_tool"] and row["expected_source"] in result["sources"]
    passed += ok
    print(json.dumps({"question": row["input"], "tool_and_source_pass": ok}))
print(f"Tool/source checks: {passed}/{len(rows)}")
if passed != len(rows):
    raise SystemExit(1)
