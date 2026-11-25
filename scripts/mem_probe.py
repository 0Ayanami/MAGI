"""One real multimodal request. Run only at the start of the online phase."""

import base64
import io
import json
import os

from openai import APIError, OpenAI
from PIL import Image

buf = io.BytesIO()
Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
client = OpenAI(api_key=os.environ["VLM_API_KEY"], base_url=os.environ["VLM_BASE_URL"], max_retries=0)
try:
    result = client.chat.completions.create(model=os.environ["VLM_MODEL"], temperature=0,
                                            extra_body={"enable_thinking": False}, messages=[{
        "role": "user", "content": [{"type": "text", "text": 'Return JSON with a "color" field in English.'},
                                    {"type": "image_url", "image_url": {"url": url}}]}])
except APIError as exc:
    raise SystemExit(f"Probe failed: {type(exc).__name__} HTTP {getattr(exc, 'status_code', None)}")
from vlm_orchestrator.vlm import parse_json

assert parse_json(result.choices[0].message.content)["color"].lower() == "red"
print(json.dumps({"model": result.model, "tokens": getattr(result.usage, "total_tokens", None)}))
