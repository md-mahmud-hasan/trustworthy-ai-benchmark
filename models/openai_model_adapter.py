import os

from openai import OpenAI
from dotenv import load_dotenv


class OpenAIModelAdapter:
    def __init__(self, model_name):
        self.model_name = model_name
        load_dotenv()

    def generate_response(self, prompt):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    adapter = OpenAIModelAdapter("gpt-4o-mini")
    print(adapter.generate_response("What is the capital of France?"))
