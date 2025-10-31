from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import json
import re

load_dotenv()

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["pos", "neg"]},
        "pros": {"type": ["array", "null"], "items": {"type": "string"}},
        "cons": {"type": ["array", "null"], "items": {"type": "string"}},
        "name": {"type": ["string", "null"]}
    },
    "required": ["key_themes", "summary", "sentiment"]
}

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation",
    max_new_tokens=250  # ✅ Increased tokens for JSON output
)

model = ChatHuggingFace(llm=llm)

prompt = f"""
You MUST output ONLY a JSON object matching this schema:

{json.dumps(json_schema, indent=2)}

Review text:
Not only that, but Black Friday 2025 is fast approaching...
"""

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            print("⚠️ Model returned invalid JSON")
            print("Raw Output:", text)
    return None

raw_response = model.invoke(prompt)
raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

data = extract_json(raw_text)

if data:
    print("\n✅ Extracted JSON Output:\n")
    print(json.dumps(data, indent=2))
else:
    print("\n❌ No JSON Found in Response\n")
    print("Raw model response:\n", raw_text)
