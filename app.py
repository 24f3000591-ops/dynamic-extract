import os
import json
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str
    schema: Dict[str, str]


SUPPORTED_TYPES = {
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "array[string]",
    "array[integer]"
}


def validate_type(value, expected):

    if value is None:
        return None

    try:

        if expected == "string":
            return str(value)

        elif expected == "integer":
            return int(value)

        elif expected == "float":
            return float(value)

        elif expected == "boolean":
            return bool(value)

        elif expected == "date":
            if isinstance(value, str):

                # Already ISO
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                    return value
                except:
                    pass

                # Try parsing natural dates
                for fmt in ("%d %B %Y", "%d %b %Y"):
                    try:
                        dt = datetime.strptime(value, fmt)
                        return dt.strftime("%Y-%m-%d")
                    except:
                        pass

            return None

        elif expected == "array[string]":
            if isinstance(value, list):
                return [str(v) for v in value]
            return None

        elif expected == "array[integer]":
            if isinstance(value, list):
                return [int(v) for v in value]
            return None

    except:
        return None

    return None


@app.post("/dynamic-extract")
def dynamic_extract(req: ExtractRequest):

    # Validate schema types
    for t in req.schema.values():
        if t not in SUPPORTED_TYPES:
            return {"error": f"Unsupported type: {t}"}

    prompt = f"""
Extract structured information.

TEXT:
{req.text}

Return ONLY valid JSON.

Schema:
{json.dumps(req.schema)}

Rules:
- Return exactly these keys.
- No extra keys.
- Missing values -> null.
- Dates -> YYYY-MM-DD.
- Numbers should be JSON numbers.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)

    output = {}

    for key, expected_type in req.schema.items():
        value = data.get(key)
        output[key] = validate_type(value, expected_type)

    return output
