"""
WhatsApp webhook server (FastAPI).

Two endpoints:
  GET  /webhook  — Meta one-time verification handshake
  POST /webhook  — receives every incoming WhatsApp message

Env vars required:
  VERIFY_TOKEN        — any string you chose when registering the webhook on Meta
  WHATSAPP_TOKEN      — permanent access token from Meta Developer Console
  PHONE_NUMBER_ID     — Meta-assigned phone number ID

Run locally:
  uvicorn whatsapp.webhook:app --host 0.0.0.0 --port 8000 --reload

Expose publicly for Meta (dev):
  ngrok http 8000
  → set https://<ngrok-url>/webhook as the webhook URL in Meta Developer Console
"""

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import PlainTextResponse

from whatsapp.handler import handle_message

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Climate — WhatsApp Webhook")


# ---------------------------------------------------------------------------
# GET /webhook  —  Meta verification handshake
# ---------------------------------------------------------------------------
@app.get("/webhook")
async def verify(request: Request):
    """
    Meta sends a GET with three query params to verify your webhook URL.
    We must echo back hub.challenge if hub.verify_token matches ours.
    """
    params = request.query_params
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    expected_token = os.getenv("VERIFY_TOKEN", "")

    if mode == "subscribe" and token == expected_token:
        logger.info("Webhook verified by Meta.")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("Webhook verification failed. Token mismatch or wrong mode.")
    return Response(status_code=403)


# ---------------------------------------------------------------------------
# POST /webhook  —  incoming messages
# ---------------------------------------------------------------------------
@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """
    Meta posts every event (messages, status updates, etc.) here.
    We filter for text messages only and process them in the background
    so we can return 200 immediately (Meta requires < 20s response).
    """
    try:
        body = await request.json()
    except Exception:
        logger.warning("Received non-JSON payload.")
        return Response(status_code=400)

    # Walk the nested Meta webhook payload structure
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            # Ignore delivery receipts and read receipts — only handle messages
            if "messages" not in value:
                continue

            for message in value["messages"]:
                msg_type = message.get("type")
                if msg_type != "text":
                    logger.info("Ignoring non-text message type: %s", msg_type)
                    continue

                sender = message.get("from")          # E.164 number, e.g. "919876543210"
                text   = message.get("text", {}).get("body", "").strip()

                if sender and text:
                    # Run RAG in background — keeps the 200 response fast
                    background_tasks.add_task(handle_message, sender, text)

    # Always return 200 quickly; Meta will retry if we return anything else
    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
