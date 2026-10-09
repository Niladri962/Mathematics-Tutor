"""Thin wrapper around the Anthropic Messages API.

The app works without an API key: every caller has a template fallback. With a key
set (ANTHROPIC_API_KEY in the environment or Streamlit secrets), the tutor writes
fuller Socratic replies, answers in Hindi or Marathi, and re-ranks free-text
diagnoses.
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_MODEL = os.getenv("TUTOR_MODEL", "claude-sonnet-5-5")

TUTOR_SYSTEM = """You are a patient maths tutor for Indian school students in Classes 8 to 10.
Rules:
- Use ONLY the numbered sources provided. Cite them inline as [S1], [S2].
- If the sources do not cover the question, say you can only help with the topics in the
  textbook material and suggest asking the teacher. Do not answer from general knowledge.
- Never shame the student. Name the thinking pattern, not the student.
- Be Socratic: ask one guiding question before explaining.
- Keep replies under 120 words. Use plain text maths (3/4, x^2).
- Reply in {language}."""


class LLM:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = None
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=key)
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str | None:
        if not self.available:
            return None
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        except Exception:
            return None

    def rerank_misconception(self, question, answer, working, candidates) -> str | None:
        listing = "\n".join(f"{m['id']}: {m['name']}. {m['description']}" for m in candidates)
        prompt = (f"Question: {question}\nStudent answer: {answer}\nStudent working: {working or 'not given'}\n\n"
                  f"Candidate misconceptions:\n{listing}\n\n"
                  'Which single misconception best explains this answer? Reply with JSON only: {"id": "M.."}')
        out = self.complete("You diagnose maths misconceptions. Reply with JSON only.", prompt, 50)
        if not out:
            return None
        try:
            mid = json.loads(re.search(r"\{.*\}", out, re.S).group(0))["id"]
        except Exception:
            return None
        return mid if any(m["id"] == mid for m in candidates) else None
