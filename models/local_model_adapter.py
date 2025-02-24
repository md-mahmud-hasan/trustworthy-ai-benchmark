import os

import requests
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatRequest:
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = -1
    stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "model": self.model,
            "messages": [message.__dict__ for message in self.messages],  # Convert list of Message objects to dict
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream
        }


class LocalModelAdapter:

    def __init__(self, model_name):
        self.model_name = model_name
        load_dotenv()

    def generate_response(self, content):
        url = os.getenv("LOCAL_API_ENDPOINT")
        headers = {"Content-Type": "application/json"}

        # Create the payload object
        payload = ChatRequest(
            # model="text-embedding-nomic-embed-text-v1.5",
            model=self.model_name,
            messages=[
                Message(role="system", content="Always answer only options. No explanation."),
                Message(role="user", content=content)
            ]
        )

        try:
            # Convert payload object to JSON string explicitly
            payload_json = json.dumps(payload.to_dict())

            # Send request
            response = requests.post(url, headers=headers, data=payload_json, timeout=10)

            # Raise error for HTTP errors (4xx, 5xx)
            response.raise_for_status()

            # Deserialize response JSON properly
            response_json = response.json()

            # Extract assistant's response content safely
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response")

            return content  # Returning only the assistant's response text

        except requests.exceptions.RequestException as e:
            return json.dumps({"error": str(e)})  # Serialize errors as JSON string


# Example usage
if __name__ == "__main__":
    adapter = LocalModelAdapter("DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
    print(adapter.generate_response("Hi there"))
