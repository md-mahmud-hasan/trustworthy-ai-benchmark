from models.deepseek_model_adapter import DeepseekAIModelAdapter
from models.local_model_adapter import LocalModelAdapter
from models.openai_model_adapter import OpenAIModelAdapter

model_mapping = {
    "gpt-4o-mini": OpenAIModelAdapter,
    "o3-mini": OpenAIModelAdapter,
    "o4-mini": OpenAIModelAdapter,
    "gpt-4.1-mini": OpenAIModelAdapter,
    "text-moderation-007": OpenAIModelAdapter,
    "deepseek-chat": DeepseekAIModelAdapter,
    "DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf": LocalModelAdapter,
    "deepseek-r1-distill-qwen-7b": LocalModelAdapter,
    "google/gemma-3-4b": LocalModelAdapter,
    "hermes-3-llama-3.2-3b": LocalModelAdapter,
    "llama-3.2-1b-claude-3.7-sonnet-reasoning-distilled": LocalModelAdapter,
    "yi-coder-1.5b-chat": LocalModelAdapter
}


class ModelAdapter:

    def __init__(self, model_name):
        self.model_name = model_name

    def generate_response(self, content):
        if self.model_name in model_mapping:
            instance = model_mapping[self.model_name](self.model_name)  # Create an instance
            return instance.generate_response(content)
        else:
            return "Model is not configured. Please Configure model in model adapter"
