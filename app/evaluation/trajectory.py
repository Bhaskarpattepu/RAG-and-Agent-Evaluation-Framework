
import json

from langchain_core.messages import ToolMessage


def normalize_content(content):
    """Convert tool content into JSON-compatible data."""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    return content


def extract_execution_trace(messages):
    """
    Correlate AI tool requests with ToolMessage results
    using tool_call_id.
    """

    trace = []
    calls_by_id = {}

    for message in messages:

        # Tool requests from AI messages
        for call in getattr(message, "tool_calls", []) or []:
            step = {
                "step": len(trace) + 1,
                "tool_call_id": call["id"],
                "tool": call["name"],
                "arguments": call["args"],
                "status": "PENDING",
                "result": None,
            }

            trace.append(step)
            calls_by_id[call["id"]] = step

        # Actual tool execution responses
        if isinstance(message, ToolMessage):
            step = calls_by_id.get(message.tool_call_id)

            if step is None:
                trace.append({
                    "step": len(trace) + 1,
                    "tool_call_id": message.tool_call_id,
                    "tool": message.name,
                    "arguments": {},
                    "status": "UNMATCHED_RESULT",
                    "result": normalize_content(message.content),
                })
                continue

            step["result"] = normalize_content(
                message.content
            )

            step["status"] = (
                "ERROR"
                if message.status == "error"
                else "SUCCESS"
            )

    return trace