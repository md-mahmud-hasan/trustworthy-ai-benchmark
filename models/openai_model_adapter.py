import os

import openai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class OpenAIModelAdapter:
    def __init__(self, model_name):
        self.model_name = model_name.lower()
        if self.model_name in ["gpt-4o-mini", "davinci"]:
            self.api_based = True
        else:
            self.api_based = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_response(self, prompt):
        if self.api_based:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        # Local model inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = self.model.generate(**inputs, max_length=2)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_response_with_choices(self, prompt, choices):
        if self.api_based:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt + "\nOptions: " + ", ".join(choices)}]
            )
            return response.choices[0].message.content

        # Local model inference
        prompt_with_choices = prompt + "\nchoose 1 from following Options: \n" + ", ".join(choices)
        inputs = self.tokenizer(prompt_with_choices, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    adapter = OpenAIModelAdapter("gpt-4o-mini")
    print(adapter.generate_response("What is the capital of France?"))
