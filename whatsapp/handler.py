"""
WhatsApp message handler.

Receives a parsed incoming message, runs the RAG pipeline,
formats the answer for WhatsApp, and sends the reply.
"""

import logging
import pipeline
from whatsapp.formatter import format_for_whatsapp
from whatsapp.sender import send_message

logger = logging.getLogger(__name__)

# Shown when the RAG pipeline finds no relevant documents
NO_ANSWER_REPLY = (
    "Sorry, I couldn't find relevant information in the documentation "
    "for your question. Please try rephrasing it."
)

# Shown when something goes wrong internally
ERROR_REPLY = (
    "Something went wrong on my end. Please try again in a moment."
)


def handle_message(sender_number: str, user_text: str) -> None:
    """
    Core handler: query → RAG pipeline → format → send.

    Args:
        sender_number — E.164 phone number of the WhatsApp user (e.g. "919876543210")
        user_text     — the raw message text typed by the user
    """
    logger.info("Incoming message from %s: %r", sender_number, user_text[:120])

    user_text = user_text.strip()
    if not user_text:
        return

    try:
        raw_answer, results = pipeline.answer(user_text)

        if not results:
            send_message(sender_number, NO_ANSWER_REPLY)
            return

        formatted = format_for_whatsapp(raw_answer)
        send_message(sender_number, formatted)

    except Exception as e:
        logger.exception("RAG pipeline error for query %r: %s", user_text, e)
        send_message(sender_number, ERROR_REPLY)
