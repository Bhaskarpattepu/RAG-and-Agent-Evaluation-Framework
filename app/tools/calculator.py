
from langchain_core.tools import tool


@tool
def calculate_refund(
    amount: float,
    deduction_percent: float = 0
) -> dict:
    """
    Calculate a hypothetical refund amount.

    Only use a deduction percentage explicitly
    provided by the user or trusted policy evidence.
    This tool does not determine refund eligibility.
    """
    if amount < 0:
        return {"error": "Amount cannot be negative"}

    if not 0 <= deduction_percent <= 100:
        return {"error": "Invalid deduction percentage"}

    deduction = amount * deduction_percent / 100

    return {
        "original_amount": round(amount, 2),
        "deduction_percent": deduction_percent,
        "deduction": round(deduction, 2),
        "refund_amount": round(amount - deduction, 2),
        "currency": "INR"
    }
