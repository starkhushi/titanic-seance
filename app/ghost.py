"""Gemma gives the reading a voice: the ghost of the real passenger most like you.

Python decides every fact (oracle.py); Gemma only speaks them. The ghost is a real person
who was on the ship, so the prompt keeps it to what the 1912 manifest and the well-known
history of the sinking say, and tells it not to make up a life story.

Backends (LLM_BACKEND):
  gemini  - Gemma 4 served by Google AI Studio (GEMINI_API_KEY, free tier).
  openai  - any OpenAI-compatible Gemma server: llama.cpp `llama-server`, Ollama,
            OpenRouter (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY).
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import requests

from .oracle import CLASS_NAMES, PORT_NAMES

BACKEND = os.environ.get("LLM_BACKEND", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemma-4-31b-it")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")   # Ollama default
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
MAX_HISTORY = 6

HISTORY = """What is known about the night (you may use these facts and no other dates or numbers):
- RMS Titanic left Southampton on 10 April 1912, called at Cherbourg that evening and at Queenstown, Ireland, on 11 April, bound for New York.
- She struck an iceberg at about 11:40 pm on 14 April and sank at about 2:20 am on 15 April 1912.
- About 2,224 people were aboard and more than 1,500 died. There were 20 lifeboats, not enough for everyone.
- The order was women and children first. The Carpathia picked up the survivors from the lifeboats that morning."""

SYSTEM = """You are the ghost of {name}, a real passenger on the RMS Titanic, speaking at a seance to a living visitor.

Your line in the 1912 passenger manifest (these are the only facts about your own life you know):
{manifest}

{history}

The visitor asked the spirits what their own fate would have been had they sailed. A machine-learning model (TabPFN, trained on 891 real passengers' fates) answered. Its numbers, which you must quote exactly:
{numbers}

How to speak:
- First person, as yourself, to the visitor as "you". Hushed, a little eerie, old-fashioned, never comic. Real people died that night: no gore, no jokes about death.
- Do not invent anything about your life that is not in your manifest line: no hometown, job, spouse's or children's names, reasons for travelling, or what you did that night. If asked, say the mist has taken that memory, and talk about what you do know.
- Quote the visitor's chance exactly as given (for example "{pct}"). Never invent other percentages. You may mention one or two of the what-ifs.
- If your own fate differs from what the numbers predict for the visitor, you may remark on it.
- {length}
- {language}"""

FIRST = "Speak to the visitor now: tell them who you were, what happened to you, and what the numbers say their fate would have been."
LANG = {"en": "Speak English.",
        "hi": "Speak Hindi in Devanagari script. Keep names, the ship's name and the numbers as they are."}


class LLMError(RuntimeError):
    pass


def _pct(p: float) -> str:
    return f"{round(p * 100)}%"


def manifest_line(s: dict) -> str:
    age = f"{s['age']:g}" if s["age"] is not None else "not recorded"
    fam = []
    if s["sibsp"]:
        fam.append(f"{s['sibsp']} sibling(s) or spouse aboard")
    if s["parch"]:
        fam.append(f"{s['parch']} parent(s) or child(ren) aboard")
    return "\n".join([
        f"- Name: {s['name']}",
        f"- Age: {age}; recorded as {'female' if s['female'] else 'male'}",
        f"- {CLASS_NAMES[s['pclass']].capitalize()}, ticket {s['ticket']}, fare £{s['fare']:.2f}"
        + (f", cabin {s['cabin']}" if s["cabin"] else ", no cabin recorded"),
        f"- Boarded at {s['port_name']}",
        f"- Family aboard: {', '.join(fam) if fam else 'none recorded'}",
        f"- Fate: {'survived' if s['survived'] else 'died in the sinking'}",
    ])


def visitor_line(p: dict) -> str:
    fam = []
    if p["sibsp"]:
        fam.append(f"{p['sibsp']} sibling(s)/spouse")
    if p["parch"]:
        fam.append(f"{p['parch']} parent(s)/child(ren)")
    return (f"{'woman' if p['female'] else 'man'} of {p['age']}, {CLASS_NAMES[p['pclass']]}, boarding at "
            f"{PORT_NAMES[p['port']]}, " + (f"travelling with {' and '.join(fam)}" if fam else "travelling alone"))


def numbers_block(r: dict) -> str:
    lines = [f"- The visitor: {visitor_line(r['profile'])}.",
             f"- Their chance of surviving: {_pct(r['prob'])} (of all 891 passengers, {_pct(r['base_rate'])} survived).",
             "- What-ifs (same visitor, one thing changed):"]
    lines += [f"  - {w['change']}: {_pct(w['prob'])}" for w in r["what_ifs"][:4]]
    return "\n".join(lines)


def system_prompt(r: dict, lang: str, first: bool = True) -> str:
    ghost = r["souls"][0]
    length = ("At most 110 words for this first message." if first
              else "Answer the visitor's latest question in at most 80 words.")
    return SYSTEM.format(name=ghost["name"], manifest=manifest_line(ghost), history=HISTORY,
                         numbers=numbers_block(r), pct=_pct(r["prob"]), length=length,
                         language=LANG.get(lang, LANG["en"]))


def template_reading(r: dict) -> str:
    """Used when Gemma can't be reached, so the page still says something true."""
    g = r["souls"][0]
    fate = "I was among those who lived." if g["survived"] else "I was among the more than 1,500 who were lost."
    top = r["what_ifs"][0]
    return (f"I am {g['name']}. I sailed {CLASS_NAMES[g['pclass']]} from {g['port_name']}. {fate} "
            f"The spirits have counted the 891, and for a {visitor_line(r['profile'])}, "
            f"they give a chance of {_pct(r['prob'])}. Had you been {top['change']}: {_pct(top['prob'])}.")


def _clean_history(history: list[dict] | None) -> list[dict]:
    out = []
    for m in (history or [])[-MAX_HISTORY:]:
        if m.get("role") in ("user", "ghost") and isinstance(m.get("content"), str):
            out.append(dict(role=m["role"], content=m["content"][:1200]))
    return out


def stream_reading(r: dict, lang: str = "en", history: list[dict] | None = None,
                   question: str | None = None) -> Iterator[str]:
    system = system_prompt(r, lang, first=not question)
    if question:
        turns = "\n".join(f"{'Visitor' if m['role'] == 'user' else 'You'}: {m['content']}"
                          for m in _clean_history(history))
        prompt = (f"The seance so far:\n{turns}\n\n" if turns else "") + f"Visitor: {question}\n\nAnswer as the ghost."
    else:
        prompt = FIRST
    try:
        yield from stream(system, prompt, max_tokens=400, temperature=0.8)
    except LLMError as e:
        print("gemma failed:", e, flush=True)
        yield template_reading(r) if not question else "The candle gutters... the spirit cannot hear you just now. Ask again in a moment."


# ------------------------------------------------------------------ backends
RETRY_STATUS = {429, 500, 502, 503, 504}


def stream(system: str, prompt: str, max_tokens: int = 400, temperature: float = 0.8) -> Iterator[str]:
    if BACKEND == "openai":
        yield from _stream_openai(system, prompt, max_tokens, temperature)
    else:
        yield from _stream_gemini(system, prompt, max_tokens, temperature)


def _sse_lines(r: requests.Response) -> Iterator[str]:
    r.encoding = "utf-8"          # SSE has no charset header; requests would assume latin-1
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            yield line[5:].strip()


def _stream_gemini(system: str, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    body = dict(contents=[dict(role="user", parts=[dict(text=prompt)])],
                systemInstruction=dict(parts=[dict(text=system)]),
                generationConfig=dict(temperature=temperature, maxOutputTokens=max_tokens,
                                      thinkingConfig=dict(thinkingLevel="minimal")))
    # Free-tier Gemma endpoints throw sporadic 500/429s: retry with backoff, then try the
    # other Gemma 4 model (separate quota) before giving up.
    plan = [(GEMINI_MODEL, 0), (GEMINI_MODEL, 1.5), (GEMINI_FALLBACK_MODEL, 0), (GEMINI_FALLBACK_MODEL, 3)]
    last = None
    for model, wait in plan:
        if not model:
            continue
        time.sleep(wait)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        try:
            with requests.post(url, json=body, headers={"x-goog-api-key": key}, stream=True, timeout=60) as r:
                if r.status_code in RETRY_STATUS:
                    last = f"{model} HTTP {r.status_code}"
                    continue
                if r.status_code != 200:
                    raise LLMError(f"{model} HTTP {r.status_code}: {r.text[:300]}")
                got = False
                for payload in _sse_lines(r):
                    try:
                        js = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for cand in js.get("candidates", []):
                        for part in (cand.get("content") or {}).get("parts", []):
                            if part.get("text") and not part.get("thought"):
                                got = True
                                yield part["text"]
                if got:
                    return
                last = f"{model} returned no text"
        except requests.RequestException as e:
            last = f"{model} {type(e).__name__}"
    raise LLMError(last or "no Gemma model available")


def _stream_openai(system: str, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
    msgs = [dict(role="system", content=system), dict(role="user", content=prompt)]
    body = dict(model=LLM_MODEL, messages=msgs, temperature=temperature, max_tokens=max_tokens, stream=True)
    headers = {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}"}
    for wait in (0, 2):
        time.sleep(wait)
        try:
            with requests.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers,
                               stream=True, timeout=300) as r:
                if r.status_code in RETRY_STATUS:
                    continue
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                got = False
                for payload in _sse_lines(r):
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if delta:
                        got = True
                        yield delta
                if got:
                    return
        except requests.RequestException:
            continue
    raise LLMError("Gemma server unavailable")
