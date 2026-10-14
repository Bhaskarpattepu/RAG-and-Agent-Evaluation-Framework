
import json

from langchain_core.messages import AIMessage, ToolMessage

from app.tools.calculator import calculate_refund
from app.tools.tickets import get_ticket_status
from app.agent import answer, extract_loop_trace


def test_calculate_refund():
    result = calculate_refund.invoke({
        "amount": 2000,
        "deduction_percent": 10
    })

    assert result["refund_amount"] == 1800


def test_invalid_deduction():
    result = calculate_refund.invoke({
        "amount": 2000,
        "deduction_percent": 150
    })

    assert "error" in result


def test_ticket_status():
    result = get_ticket_status.invoke({
        "ticket_id": "TKT-1001"
    })

    assert result["found"] is True
    assert result["status"] == "OPEN"


def test_unknown_ticket():
    result = get_ticket_status.invoke({
        "ticket_id": "TKT-9999"
    })

    assert result["found"] is False


def test_agent_selects_ticket_tool():
    result = answer(
        "What is the status of ticket TKT-1001?"
    )

    called_tools = [
        call["tool"]
        for call in result["tool_calls"]
    ]

    assert "get_ticket_status" in called_tools


def test_agent_selects_retrieval_tool():
    result = answer(
        "Are digital gift cards refundable?"
    )

    called_tools = [
        call["tool"]
        for call in result["tool_calls"]
    ]

    assert "search_knowledge_base" in called_tools

    assert "refund_policy.txt" in result["sources"]


def test_agent_stops_at_model_turn_budget(monkeypatch):
    class ToolRequestModel:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_ticket_status",
                    "args": {"ticket_id": "TKT-1001"},
                    "id": "call_budget_test",
                    "type": "tool_call",
                }],
            )

    monkeypatch.setattr("app.agent.llm", ToolRequestModel())

    result = answer("Check ticket TKT-1001", max_steps=1)

    assert result["model_turns"] == 1
    assert result["stop_reason"] == "max_steps"
    assert result["loop_trace"][0]["actions"][0]["tool"] == "get_ticket_status"
    assert result["loop_trace"][0]["observations"] == []
    assert "maximum of 1 reasoning steps" in result["answer"]


def test_extract_loop_trace_correlates_tool_observation():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_ticket_status",
                "args": {"ticket_id": "TKT-1001"},
                "id": "call_trace_test",
                "type": "tool_call",
            }],
        ),
        ToolMessage(
            content='{"found": true}',
            name="get_ticket_status",
            tool_call_id="call_trace_test",
        ),
    ]

    trace = extract_loop_trace(messages)

    assert trace[0]["turn"] == 1
    assert trace[0]["actions"][0]["arguments"] == {"ticket_id": "TKT-1001"}
    assert trace[0]["observations"][0]["result"] == '{"found": true}'
