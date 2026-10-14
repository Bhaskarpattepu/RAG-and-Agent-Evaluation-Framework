
import json
from pathlib import Path

from langchain_core.tools import tool

TICKET_FILE = Path("data/tickets.json")


@tool
def get_ticket_status(ticket_id: str) -> dict:
    """
    Retrieve the current status of a support ticket.

    Use this tool when the user provides a ticket ID
    and asks about its status.
    """
    tickets = json.loads(
        TICKET_FILE.read_text(encoding="utf-8")
    )

    ticket_id = ticket_id.strip().upper()

    if ticket_id not in tickets:
        return {
            "ticket_id": ticket_id,
            "found": False,
            "message": "Ticket not found"
        }

    return {
        "ticket_id": ticket_id,
        "found": True,
        **tickets[ticket_id]
    }


@tool
def draft_support_ticket(issue: str) -> dict:
    """
    Prepare a draft support ticket for the user.

    This does not create or submit a real ticket.
    """
    return {
        "status": "DRAFT",
        "issue": issue,
        "message": (
            "Ticket draft prepared. "
            "It has not been submitted."
        )
    }
