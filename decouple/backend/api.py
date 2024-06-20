from vlmeval.api import OpenAIWrapper, OpenAIWrapperInternal
from ..config import gpt_version_map, reasoning_mapping, stop_tokens
import os

class LLMWrapper:

    def __init__(self, model_name, max_tokens=512, verbose=True, retry=5):
        
        # api bases, openai default
        self.deepseek_api_base = 'https://api.deepseek.com/v1/chat/completions'

        # server settings of vllm
        self.PORT = 8080
        self.vllm_api_base = f'http://localhost:{self.PORT}/v1/chat/completions'

        if model_name.endswith('-2048'):
            model_name = model_name.replace('-2048', '')
            max_tokens = 2048
            
        print(model_name)
        if model_name in gpt_version_map:
            gpt_version = gpt_version_map[model_name]
            model = OpenAIWrapper(gpt_version, max_tokens=max_tokens, verbose=verbose, retry=retry)
        elif reasoning_mapping[model_name] == 'vllm':
            model = OpenAIWrapper(model_name, api_base=self.vllm_api_base, max_tokens=max_tokens, system_prompt='You are a helpful assistant.', verbose=verbose, retry=retry, stop=stop_tokens[model_name])
        elif reasoning_mapping[model_name] == 'deepseek':
            deepseek_key = os.environ['DEEPSEEK_API_KEY']
            model = OpenAIWrapper(model_name, api_base=self.deepseek_api_base, key=deepseek_key, max_tokens=max_tokens, system_prompt='You are a helpful assistant.', verbose=verbose, retry=retry)
        else:
            print('Unknown API model for inference')
        
        self.model = model

    def generate(self, prompt, **kwargs):
        response = self.model.generate(prompt, **kwargs)
        return response
    
    @staticmethod
    def api_models():
        gpt_models = list(gpt_version_map.keys())
        api_models = gpt_models.copy()
        api_models.extend(list(reasoning_mapping.keys()))
        return api_models