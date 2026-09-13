import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(override=True)

PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def push(message: str):
    print(f"Push: {message}")

    payload = {
        "user": PUSHOVER_USER,
        "token": PUSHOVER_TOKEN,
        "message": message,
    }

    requests.post(PUSHOVER_URL, data=payload, timeout=10)


@tool
def record_user_details(
    email: str,
    name: str = "Name not provided",
    notes: str = "not provided",
) -> str:
    """Record a visitor who is interested in getting in touch."""

    push(
        f"Recording interest from {name} "
        f"with email {email} "
        f"and notes {notes}"
    )

    return "OK"


@tool
def record_unknown_question(question: str) -> str:
    """Record a question that could not be answered from the knowledge base."""

    push(
        f"Recording unknown question: {question}"
    )

    return "OK"


tools = [
    record_user_details,
    record_unknown_question,
]
