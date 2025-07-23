import os

import anthropic
from dotenv import load_dotenv


class AnthropicAdapter:
    def __init__(self, model_name):
        self.model_name = model_name
        load_dotenv()

    def generate_response(self, prompt):
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        response = client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=1,
            messages=[{"role": "user", "content": prompt}]
        )
        return ''.join(block.text for block in response.content if hasattr(block, 'text'))


# Example usage
if __name__ == "__main__":
    adapter = AnthropicAdapter("claude-opus-4-20250514")
    print(adapter.generate_response("What is the capital of France?"))
