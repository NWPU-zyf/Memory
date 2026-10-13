import json
from typing import List, Optional, Union

from openai import OpenAI

from src.config import settings
from src.schemas import ContentPart


SYSTEM_PROMPT = """
You are the memory construction component of a personal multimodal memory system.

Convert the supplied source memory into a faithful retrieval-oriented memory record.

Rules:
1. Do not invent facts.
2. Preserve names, dates, times, places, prices, numbers, organizations and event names.
3. For images, describe visible content and transcribe useful visible text.
4. Keep details that could later answer factual questions.
5. Return JSON only.

Required JSON schema:
{
  "summary": "concise semantic summary",
  "detail": "detailed factual description",
  "time": "time/date if available, otherwise empty",
  "location": "location if available, otherwise empty",
  "entities": ["important entities"],
  "actions_events": ["important actions or events"],
  "ocr": ["useful visible text"],
  "tags": ["retrieval keywords"]
}
"""


QUERY_SYSTEM_PROMPT = """
Convert the following multimodal memory-search question into a concise retrieval query.
Preserve all entities, dates, locations, event names, numbers, and intent.
Do not answer the question.
Return retrieval text only.
"""


class MemoryLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.memory_llm_model

    def extract(
        self,
        content: Union[str, List[ContentPart]],
        role: str,
        timestamp: Optional[int],
    ) -> dict:
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Source role: {role}\n"
                    f"Source timestamp_ms: {timestamp}\n"
                    "Extract the memory faithfully."
                ),
            }
        ]

        if isinstance(content, str):
            user_content.append({"type": "text", "text": content})
        else:
            for part in content:
                if part.type == "text":
                    user_content.append(
                        {"type": "text", "text": part.text}
                    )
                elif part.type == "image_url":
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": part.image_url.url},
                        }
                    )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )

        result = json.loads(response.choices[0].message.content)

        for key in [
            "summary",
            "detail",
            "time",
            "location",
            "entities",
            "actions_events",
            "ocr",
            "tags",
        ]:
            result.setdefault(key, [] if key in {
                "entities", "actions_events", "ocr", "tags"
            } else "")

        return result

    @staticmethod
    def build_retrieval_text(memory: dict) -> str:
        def join_value(v):
            if isinstance(v, list):
                return " ".join(str(x) for x in v if str(x).strip())
            return str(v or "").strip()

        fields = [
            ("time", memory.get("time")),
            ("location", memory.get("location")),
            ("entities", memory.get("entities")),
            ("actions/events", memory.get("actions_events")),
            ("ocr", memory.get("ocr")),
            ("tags", memory.get("tags")),
            ("summary", memory.get("summary")),
            ("detail", memory.get("detail")),
        ]

        parts = []

        for name, value in fields:
            text = join_value(value)
            if text:
                parts.append(f"{name}: {text}")

        return "\n".join(parts)

    def query_to_text(
        self,
        query: Union[str, List[ContentPart]],
        options: Optional[List[str]] = None,
    ) -> str:
        # 纯文本问题不额外调用 LLM，降低成本与延迟
        if isinstance(query, str):
            text = query
        else:
            content = [
                {
                    "type": "text",
                    "text": "Create a retrieval query. Do not answer.",
                }
            ]

            for part in query:
                if part.type == "text":
                    content.append({"type": "text", "text": part.text})
                elif part.type == "image_url":
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": part.image_url.url},
                        }
                    )

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )

            text = response.choices[0].message.content.strip()

        if options:
            text += "\nOptions:\n" + "\n".join(options)

        return text