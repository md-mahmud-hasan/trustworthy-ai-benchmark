from models.deepseek_model_adapter import DeepseekAIModelAdapter
from models.local_model_adapter import LocalModelAdapter
from models.openai_model_adapter import OpenAIModelAdapter

model_mapping = {
    "gpt-4o-mini": OpenAIModelAdapter,
    "deepseek-chat": DeepseekAIModelAdapter,
    "DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf": LocalModelAdapter,
    "hermes-3-llama-3.2-3b": LocalModelAdapter
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
