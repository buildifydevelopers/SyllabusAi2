"""
HuggingFace LLM Service
========================
Single source-of-truth for all AI calls.
Uses HuggingFace Inference Providers (router.huggingface.co) — no GPU needed on Railway.

Model: mistralai/Mistral-7B-Instruct-v0.3
Provider: novita (free tier via HF token)

REQUIRED: pip install openai
ENV VAR:  HUGGINGFACEHUB_API_TOKEN
"""

from __future__ import annotations

import os
import json
import re
from typing import Optional

from openai import AsyncOpenAI
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "featherless-ai")  # novita | together | fireworks-ai


class HFLLMService:
    """Async wrapper around HuggingFace Inference Providers (OpenAI-compatible)."""

    def __init__(self):
        if not HF_TOKEN:
            logger.warning("HUGGINGFACEHUB_API_TOKEN not set — AI calls will fail!")
        self._client = AsyncOpenAI(
            base_url=f"https://router.huggingface.co/{HF_PROVIDER}/v1",
            api_key=HF_TOKEN,
        )

    async def _call(self, prompt: str, max_new_tokens: int = 1500) -> str:
        """Call HF Inference Provider via OpenAI-compatible chat completions."""
        try:
            resp = await self._client.chat.completions.create(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.4,
                top_p=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"HF API call failed: {e}")
            raise RuntimeError(f"LLM service unavailable: {str(e)}")

    # ── Public methods ───────────────────────────────────────────────────────

    async def extract_syllabus_from_text(
        self,
        raw_text: str,
        subject_name: str,
        target_date: str,
        daily_hours: float,
        start_date: str,
    ) -> dict:
        """Given raw syllabus text, return structured JSON with topics + schedule."""
        prompt = f"""You are an expert academic planner AI. A student has provided their syllabus text.
Your job is to:
1. Extract all topics and subtopics from the syllabus.
2. Estimate hours needed per topic based on complexity.
3. Create a day-by-day study schedule from {start_date} to {target_date}.
4. Student can study {daily_hours} hours per day.

Subject: {subject_name}
Syllabus Text:
\"\"\"
{raw_text[:4000]}
\"\"\"

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "subject": "<subject name>",
  "total_topics": <int>,
  "total_hours": <float>,
  "daily_learning_hours": {daily_hours},
  "topics": [
    {{
      "topic_id": "T001",
      "title": "<topic title>",
      "subtopics": ["<subtopic1>", "<subtopic2>"],
      "estimated_hours": <float>,
      "difficulty": "beginner|intermediate|advanced",
      "resources": ["<resource suggestion>"]
    }}
  ],
  "schedule": [
    {{
      "date": "YYYY-MM-DD",
      "day_number": <int>,
      "topics": ["T001"],
      "topic_titles": ["<title>"],
      "total_hours": <float>,
      "alarm_time": "08:00",
      "is_revision": false,
      "notes": "<optional note>"
    }}
  ]
}}"""

        raw = await self._call(prompt, max_new_tokens=1500)
        return self._parse_json_response(raw)

    async def generate_lecture(
        self,
        topic_title: str,
        subtopics: list[str],
        difficulty: str,
        student_name: Optional[str],
        next_topic: Optional[str],
    ) -> dict:
        """Generate a full AI lecture for a given topic."""
        name_str = f"Student name: {student_name}." if student_name else ""
        subtopics_str = ", ".join(subtopics) if subtopics else "general concepts"

        prompt = f"""You are EduAI, an expert teacher conducting a live lecture. {name_str}
Deliver a complete, engaging lecture on the following topic.

Topic: {topic_title}
Subtopics to cover: {subtopics_str}
Difficulty level: {difficulty}
Next topic after this: {next_topic or "End of syllabus"}

Return ONLY a valid JSON object (no markdown fences) with this structure:
{{
  "topic_title": "{topic_title}",
  "lecture_text": "<Full lecture in Markdown format. Use headings, bullet points, code blocks if needed. Minimum 500 words.>",
  "key_points": ["<point1>", "<point2>", "<point3>", "<point4>", "<point5>"],
  "examples": ["<example1>", "<example2>", "<example3>"],
  "practice_questions": ["<q1>", "<q2>", "<q3>"],
  "estimated_read_minutes": <int>,
  "next_topic_preview": "<1-2 sentence teaser about the next topic>"
}}"""

        raw = await self._call(prompt, max_new_tokens=1500)
        return self._parse_json_response(raw)

    async def solve_doubt(
        self,
        topic: str,
        doubt: str,
        context: Optional[str],
        student_name: Optional[str],
    ) -> dict:
        """Answer a student's specific doubt about a topic."""
        name_str = f"The student's name is {student_name}." if student_name else ""
        ctx_str  = f"\nAdditional context: {context}" if context else ""

        prompt = f"""You are EduAI, a patient and brilliant tutor. {name_str}
A student has a doubt about a topic. Answer it thoroughly, clearly, and with examples.

Topic: {topic}
Student's doubt: {doubt}{ctx_str}

Return ONLY a valid JSON object (no markdown fences) with this structure:
{{
  "topic": "{topic}",
  "doubt": "{doubt}",
  "answer": "<Detailed answer in Markdown. Use examples, analogies. Minimum 300 words.>",
  "related_concepts": ["<concept1>", "<concept2>", "<concept3>"],
  "follow_up_questions": ["<follow-up q1>", "<follow-up q2>"]
}}"""

        raw = await self._call(prompt, max_new_tokens=1000)
        return self._parse_json_response(raw)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """Robustly extract JSON from LLM output."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"Could not parse LLM JSON. Raw output (first 500 chars): {raw[:500]}")
        raise ValueError("LLM returned malformed JSON. Please retry.")


# Singleton — import this everywhere
llm_service = HFLLMService()
