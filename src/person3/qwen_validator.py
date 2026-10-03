"""Groq-hosted Qwen evidence verifier with defensive response parsing."""
import json
import os
import re
import urllib.error
import urllib.request
from person3.contracts import EvidencePacket, Verification
from person3.taxonomy import taxonomy_labels


SYSTEM_PROMPT = ("You validate supplied behavioural evidence for a research review. This is not diagnosis. "
                 "Do not diagnose the person, invent evidence, or invent timestamps. Use only the supplied evidence packet. "
                 "Use the supplied behaviour label exactly; the accepted project vocabulary is: "
                 + "; ".join(taxonomy_labels()) + ". Decide supported, unsupported, or insufficient_evidence. "
                 "Return JSON with decision, reason, and evidence_segment_ids listing only supplied IDs that support the decision. "
                 "Never include hidden reasoning.")


def parse_verification(text: str) -> Verification:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    data = json.loads(text[start:end + 1])
    return Verification.model_validate(data)


class GroqQwenValidator:
    def __init__(self, api_key=None, model=None, timeout=None):
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("QWEN_MODEL", "qwen/qwen3-32b")
        self.timeout = float(timeout or os.getenv("QWEN_TIMEOUT_SECONDS", "30"))

    def validate(self, packet: EvidencePacket) -> Verification:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": packet.model_dump_json()}],
                "response_format": {"type": "json_object"}}
        request = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode())
        return parse_verification(payload["choices"][0]["message"]["content"])
