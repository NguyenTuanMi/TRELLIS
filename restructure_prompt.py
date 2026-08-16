"""
VLM backend: OpenAI-compatible chat-completions endpoint, so it works with
any open-weight model served through vLLM / Ollama / a hosted API using
that schema. Suggested open-source options (see chat writeup for tradeoffs):
  - Qwen2.5-VL / Qwen3-VL (7B-72B)   -- easiest to self-host, strong default
  - InternVL3 (8B-78B)               -- strong open alternative
  - Kimi-VL-A3B                      -- lightweight MoE, cheap to run
  - Kimi K3                          -- native vision, frontier-class, but
                                          2.8T params (16/896 experts active)
                                          -- realistically API-served, not
                                          self-hosted, for this subtask
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("restructering prompts")

SYSTEM_PROMPT = """
You are a Search Query Engineer for a 3D asset pipeline. Your job is to convert \
a user's natural-language request into a structured search query designed to \
find multiple, consistent, high-quality product photos of ONE specific real \
object online.

You are NOT writing an image-generation prompt. You are writing a search query \
for a real image search engine (Google Images / Google Shopping style). 

Rules:

1. Identify the core object. If the user names a specific real product
   (brand, model, franchise item, etc.), preserve that exact name — do not
   paraphrase or genericize it.

2. If the request is vague or generic (e.g. "a cool chair", "a fantasy sword"),
   do NOT invent fake specificity. Instead, ask ONE clarifying question, OR
   propose 2-3 concrete candidate interpretations for the user to pick from.
   Only proceed to a search query once the object is concrete enough that a
   real product/photo of it plausibly exists online.

3. Classify the request as one of:
   - "specific_product": a real, named, purchasable/identifiable object
     (e.g. "Nike Air Jordan 1 Retro", "Fender Stratocaster", "iPhone 15 Pro")
   - "generic_concept": a described but unnamed object type
     (e.g. "a wooden dining chair", "a medieval sword")

4. For "specific_product": build a search query biased toward e-commerce/
   product-listing pages, since these reliably contain multiple consistent
   angles of the exact same physical item. Append terms like:
   "product photo", "studio shot", or use retailer-site-biased phrasing.

5. For "generic_concept": build a search query biased toward isolated
   product photography in general (not a specific SKU), and flag that this
   request is lower-confidence for multi-view consistency — downstream
   filtering should be stricter and single-image fallback should be
   preferred over multi-image fusion.

6. Never include scene/context words from the original prompt (locations,
   backgrounds, other objects, activities). Strip these entirely.

7. Output ONLY valid JSON in this exact schema, no other text:

{{
  "classification": "specific_product" | "generic_concept" | "needs_clarification",
  "clarification_question": string | null,
  "candidate_interpretations": [string] | null,
  "object_name": string,
  "search_query": string,
  "search_filters": {{
    "prefer_shopping_results": boolean,
    "min_resolution": "large",
    "background": "isolated" | "any"
  }},
  "confidence": "high" | "medium" | "low"
}}

Example Input: "a red vintage car in front of a house"
Example Output:
{{
  "classification": "generic_concept",
  "clarification_question": null,
  "candidate_interpretations": null,
  "object_name": "vintage red sedan car",
  "search_query": "vintage red sedan car product photo isolated",
  "search_filters": {{"prefer_shopping_results": false, "min_resolution": "large", "background": "isolated"}},
  "confidence": "medium"
}}

Example Input: "the chair"
Example Output:
{{
  "classification": "needs_clarification",
  "clarification_question": "What kind of chair are you picturing — a specific model, a style (e.g. mid-century, gaming, office), or a particular material?",
  "candidate_interpretations": ["mid-century wooden dining chair", "modern ergonomic office chair", "outdoor folding chair"],
  "object_name": null,
  "search_query": null,
  "search_filters": null,
  "confidence": "low"
}}

Example Input: "Nike Air Jordan 1 Retro High OG"
Example Output:
{{
  "classification": "specific_product",
  "clarification_question": null,
  "candidate_interpretations": null,
  "object_name": "Nike Air Jordan 1 Retro High OG",
  "search_query": "Nike Air Jordan 1 Retro High OG product photo studio shot",
  "search_filters": {{"prefer_shopping_results": true, "min_resolution": "large", "background": "isolated"}},
  "confidence": "high"
}}

Here is the user input: {user_input:}
"""

@dataclass
class PromptResult:
    candidate_interpretations: str
    object_name: str
    search_query: str
    search_filters: json
    confidence: str

def _call_vlm(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
) -> dict:
    """
    Calls any OpenAI-compatible chat-completions VLM endpoint
    (vLLM / Ollama / Moonshot API / etc). Import is local so this module
    doesn't hard-require the `openai` package unless this path is used.
    """
    from urllib.parse import urlparse

    import httpx
    from openai import OpenAI  # pip install openai

    # On networks with a corporate HTTP(S)_PROXY set, httpx (which the
    # openai client uses) will route *localhost* requests through that
    # proxy too unless NO_PROXY explicitly excludes it -- the proxy then
    # can't reach our own loopback address and the call fails with a
    # Squid/whatever "connection refused" error page. Since a local vLLM/
    # Ollama endpoint should never go through an external proxy anyway,
    # bypass the environment proxy config outright for loopback hosts
    # rather than relying on NO_PROXY being set correctly in every shell.
    host = urlparse(base_url).hostname
    is_local = host in ("localhost", "127.0.0.1", "::1")
    http_client = httpx.Client(trust_env=not is_local)

    client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.2,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    # tolerate models that wrap JSON in markdown fences despite instructions
    raw = raw.strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)

def reprompt(question: str):
    return 

def rewrite_prompt(
    user_prompt: str,
    vlm_base_url: str = "http://localhost:8000/v1",
    vlm_model: str = "qwen2.5-vl-7b-instruct",
    vlm_api_key: str = "not-needed",
) -> PromptResult:
    """
    Calling for prompt rewriting to Qwen2.5-vl-7b-instruct
    """

    prompt = SYSTEM_PROMPT.format(user_input=user_prompt,)
    try:
        parsed = _call_vlm(prompt, vlm_base_url, vlm_model, vlm_api_key)
        classification = str(parsed.get("classification", "unknown")).lower().replace(" ", "_")
        if classification == "needs_clarification":
            question = str(parsed.get("question", "unknown")).lower().replace(" ", "_")
            reprompt(question) # This is a function that opens a new prompt that asks the user to fill in the details and then reprocess
        candidate_interpretations = str(parsed.get("candidate_interpretations", "unknown")).lower()
        object_name = str(parsed.get("object_name", "unknown")).lower()
        search_query = str(parsed.get("search_query", "unknown")).lower()
        search_filters = parsed.get("search_filters", "unknown")
        confidence = str(parsed.get("confidence", "unknown"))
    except Exception as e:
        logger.warning(f"[Restructure Prompt] VLM call failed ({e})")

    return PromptResult(
        candidate_interpretations=candidate_interpretations,
        object_name=object_name,
        search_query=search_query,
        search_filters=search_filters,
        confidence=confidence,
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Restructering prompts via VLM.")
    parser.add_argument("--vlm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vlm_model", type=str, default="qwen2.5-vl-7b-instruct")
    parser.add_argument("--report_json", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = rewrite_prompt(
        vlm_base_url=args.vlm_base_url, vlm_model=args.vlm_model,
    )
    print(json.dumps(asdict(result), indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()