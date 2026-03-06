"""
WhatsApp message sender via Meta Cloud API.

Env vars required:
  WHATSAPP_TOKEN      — permanent access token from Meta Developer Console
  PHONE_NUMBER_ID     — the Meta-assigned ID for your WhatsApp business number
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

META_API_VERSION = "v19.0"
META_API_URL = "https://graph.facebook.com/{version}/{phone_number_id}/messages"


def send_message(to: str, text: str) -> bool:
    """
    Send a plain-text WhatsApp message to a recipient.

    Args:
        to   — recipient phone number in E.164 format e.g. "919876543210"
               (country code + number, no + prefix, no spaces)
        text — formatted message body (WhatsApp markdown safe)

    Returns:
        True if Meta API accepted the message, False otherwise.
    """
    token = os.getenv("WHATSAPP_TOKEN")
    phone_number_id = os.getenv("PHONE_NUMBER_ID")

    if not token or not phone_number_id:
        logger.error(
            "Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID environment variables."
        )
        return False

    url = META_API_URL.format(
        version=META_API_VERSION,
        phone_number_id=phone_number_id,
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Message sent to %s | status=%s", to, response.status_code)
        return True
    except requests.HTTPError as e:
        logger.error(
            "Meta API HTTP error: %s | response: %s",
            e,
            e.response.text if e.response else "no response",
        )
        return False
    except requests.RequestException as e:
        logger.error("Meta API request failed: %s", e)
        return False
