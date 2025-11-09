# backend/llm_client.py
import os
import requests
from typing import Any, Dict, Optional
import json
import streamlit as st

from dotenv import load_dotenv

# Load environment variables from .env at project root
load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format_json: bool = False,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Thin wrapper around OpenRouter chat/completions.
    Returns either parsed JSON (if response_format_json=True) or raw string.
    """

    
    # 🔍 DEBUG: see exactly what we got from the environment
    api_key = st.secrets["OPENROUTER_API_KEY"] #ADD YOUR OWN API KEY
    print("[DEBUG] OPENROUTER_API_KEY from env:",(api_key))

    if not api_key:
        # Fail early with a clear message instead of 401 later
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Create a .env file with OPENROUTER_API_KEY=... in the project root."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 🔍 DEBUG: check what we’re actually sending in the header (masked)
    print("[DEBUG] Auth header prefix:", headers["Authorization"][:25])
    print("[DEBUG] Using model:", model)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)

    # 🔍 DEBUG: log status + first part of body before raising
    print("[DEBUG] OpenRouter status:", resp.status_code)
    print("[DEBUG] OpenRouter raw response (start):", resp.text[:300])

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Print server message to your terminal for easier debugging
        print("[OpenRouter ERROR]", resp.status_code, resp.text)
        raise

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    if response_format_json:
        # The model is instructed to return JSON only
        return json.loads(content)

    return {"raw": content}
