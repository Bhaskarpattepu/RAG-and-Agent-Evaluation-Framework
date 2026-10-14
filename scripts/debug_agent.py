
import argparse

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    HumanMessage,
)

from app.agent import DEFAULT_MAX_STEPS, agent_graph


parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-steps",
    type=int,
    default=DEFAULT_MAX_STEPS,
    help="maximum number of model turns in the ReAct loop",
)
parser.add_argument(
    "question",
    nargs="*",
    default=["What is the status of ticket TKT-1001?"],
)
args = parser.parse_args()
question = " ".join(args.question)

if args.max_steps < 1:
    parser.error("--max-steps must be at least 1")

print(f"\nUSER: {question}\n")
print(f"MAX MODEL TURNS: {args.max_steps}\n")

model_turn = 0

for event in agent_graph.stream(
    {
        "messages": [
            HumanMessage(content=question)
        ],
        "model_turns": 0,
        "max_steps": args.max_steps,
        "stop_reason": "completed",
    },
    config={"recursion_limit": (2 * args.max_steps) + 2},
    stream_mode="updates",
):
    for node_name, update in event.items():

        print(f"\nNODE: {node_name}")

        for message in update.get("messages", []):

            if isinstance(message, AIMessage):
                model_turn += 1
                print(f"MODEL TURN: {model_turn}")

                if message.tool_calls:
                    for call in message.tool_calls:
                        print("TOOL REQUEST:", call["name"])
                        print("ARGUMENTS:", call["args"])
                else:
                    print("AI ANSWER:", message.content)

            elif isinstance(message, ToolMessage):
                print("TOOL EXECUTED:", message.name)
                print("RESULT:", message.content)

        if node_name == "step_limit":
            print("STOP REASON: max_steps")
