# backend/classification.py

import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional

from .pii_detection import find_pii
from .safety import naive_unsafe_check, profanity_pages  # ✅ includes profanity check
from .llm_client import call_openrouter_chat

# === Prompt library paths ===
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_CONFIG_PATH = os.path.join(PROMPTS_DIR, "prompt_config.json")

PRIMARY_MODEL = "meta-llama/llama-3.1-8b-instruct"
VALIDATOR_MODEL = "meta-llama/llama-3.1-70b-instruct"
VALIDATION_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def load_prompt_config() -> Dict[str, Any]:
    """Load prompt_config.json with simple rule mapping."""
    try:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback: at minimum use base_classification.txt
        return {"default": ["base_classification.txt"]}


def load_prompt_template(filename: str) -> str:
    """Load a single prompt template from the prompts folder."""
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_system_prompt(context_flags: Dict[str, bool]) -> str:
    """
    Compose the system prompt dynamically from the prompt library,
    based on simple rule mapping in prompt_config.json.

    context_flags can contain:
      - unsafe_keyword_flag: bool
      - has_ssn: bool
      - has_pii: bool
    """
    cfg = load_prompt_config()

    # Very simple rule tree (can be expanded later)
    if context_flags.get("unsafe_keyword_flag"):
        key = "unsafe"
    elif context_flags.get("has_ssn") or context_flags.get("has_pii"):
        key = "sensitive"
    else:
        key = "public"

    template_list = cfg.get(key, cfg.get("default", ["base_classification.txt"]))

    pieces: List[str] = []
    for fname in template_list:
        try:
            pieces.append(load_prompt_template(fname))
        except FileNotFoundError:
            # Skip missing templates so the app doesn't crash
            continue

    # Absolute fallback: use base_classification if everything else fails
    if not pieces:
        pieces.append(load_prompt_template("base_classification.txt"))

    return "\n\n".join(pieces)


def run_llm_classification(
    payload: Dict[str, Any],
    model: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """
    Makes a single OpenRouter API call to the given model.
    Returns a normalized JSON structure with category, unsafe, etc.
    """
    user_prompt = json.dumps(payload, ensure_ascii=False)

    print(f"[LLM] Calling OpenRouter model={model}")

    llm_raw = call_openrouter_chat(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format_json=True,
        temperature=0.1,
    )

    category = llm_raw.get("category", "Public")
    unsafe = bool(llm_raw.get("unsafe", False))
    kid_safe = bool(llm_raw.get("kid_safe", not unsafe))

    try:
        confidence = float(llm_raw.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    reasoning = llm_raw.get("reasoning", "No reasoning provided.")
    citations = llm_raw.get("citations", []) or []

    return {
        "category": category,
        "unsafe": unsafe,
        "kid_safe": kid_safe,
        "confidence": confidence,
        "reasoning": reasoning,
        "citations": citations,
    }


def classify_document(doc_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates full classification:
    - Extract heuristic features (PII, profanity, unsafe)
    - Builds a dynamic system prompt from the prompt library
    - Calls LLaMA models
    - Merges LLM + rules for final classification
    """
    pages = doc_info["pages"]
    num_pages = doc_info["num_pages"]
    num_images = doc_info["num_images"]

    # === Heuristic extraction ===
    pii_raw = find_pii(pages)
    # Filter out public-facing business contact info (brochures, websites)
    pii = [
        f for f in pii_raw
        if not (f.get("type") == "email" and f.get("is_business"))
    ]

    unsafe_flag_heuristic = naive_unsafe_check(pages)
    prof_pages = profanity_pages(pages)
    has_ssn = any(f["type"] == "ssn" for f in pii)
    has_pii = bool(pii)

    # === Build dynamic system prompt from prompt library ===
    context_flags = {
        "unsafe_keyword_flag": unsafe_flag_heuristic,
        "has_ssn": has_ssn,
        "has_pii": has_pii,
    }
    system_prompt = build_system_prompt(context_flags)

    # === Compact summaries ===
    page_summaries: List[Dict[str, Any]] = []
    for p in pages:
        text = (p.get("text") or "").strip()
        if len(text) > 800:
            text = text[:800] + "..."
        page_summaries.append({
            "page": p["page_num"],
            "text": text,
        })

    # === LLM payload ===
    user_payload: Dict[str, Any] = {
        "num_pages": num_pages,
        "num_images": num_images,
        "pii_findings": pii,
        "unsafe_keyword_flag": unsafe_flag_heuristic,
        "profanity_pages": prof_pages,
        "page_summaries": page_summaries,
    }

    # === 1) Primary model ===
    primary = run_llm_classification(user_payload, PRIMARY_MODEL, system_prompt)

    # === 2) Validator (optional) ===
    validator: Optional[Dict[str, Any]] = None
    if primary["confidence"] < VALIDATION_THRESHOLD:
        validator = run_llm_classification(user_payload, VALIDATOR_MODEL, system_prompt)

    # === Combine results ===
    category = primary["category"]
    unsafe_flag_llm = primary["unsafe"]
    kid_safe = primary["kid_safe"]
    confidence = primary["confidence"]
    reasoning = primary["reasoning"]
    citations: List[Dict[str, Any]] = list(primary["citations"] or [])

    if validator is not None:
        disagreement = (
            validator["category"] != primary["category"]
            or validator["unsafe"] != primary["unsafe"]
        )

        if disagreement:
            print("[Validator] 70B disagreed with 8B.")
            category = validator["category"]
            unsafe_flag_llm = validator["unsafe"]
            kid_safe = validator["kid_safe"]
            confidence = min(primary["confidence"], validator["confidence"], 0.7)
            reasoning = (
                "Validator cross-check: the 70B model disagreed with 8B.\n"
                f"Primary said '{primary['category']}', validator said '{validator['category']}'.\n"
                "Validator chosen for higher precision.\n\n"
                f"Primary reasoning: {primary['reasoning']}\n\n"
                f"Validator reasoning: {validator['reasoning']}"
            )
            citations += (validator.get("citations") or [])

    # === Merge with deterministic rules ===
    unsafe_flag = unsafe_flag_llm or unsafe_flag_heuristic

    # Unsafe always overrides category
    if unsafe_flag and "Unsafe" not in category:
        if category and category != "Unsafe":
            category = f"{category} and Unsafe"
        else:
            category = "Unsafe"

    # SSN rule: override to Highly Sensitive
    if has_ssn and "Highly Sensitive" not in category:
        if "Unsafe" in category:
            category = "Highly Sensitive and Unsafe"
        else:
            category = "Highly Sensitive"
        confidence = max(confidence, 0.9)

    # === Citations ===
    for f in pii:
        citations.append({
            "page": f["page"],
            "reason": f"Detected {f['type'].upper()}: {f['value']}",
        })

    for pnum in prof_pages:
        citations.append({
            "page": pnum,
            "reason": "Strong profanity detected on this page (not kid-safe).",
        })

    # === Kid-safe normalization ===
    kid_safe_final = kid_safe and not unsafe_flag and not prof_pages

    # === Optional note for business contact info ===
    if not unsafe_flag and any(f.get("is_business") for f in pii_raw):
        reasoning += "\nNote: Detected only public business contact info (e.g., info@company.com) — does not increase sensitivity."

    # === Final output ===
    return {
        "category": category,
        "unsafe": unsafe_flag,
        "kid_safe": kid_safe_final,
        "confidence": confidence,
        "reasoning": reasoning,
        "citations": citations,
    }
