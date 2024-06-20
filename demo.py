from decouple.frontend import supported_VLM
from decouple.backend import  LLMWrapper, build_infer_prompt_external
from decouple.utils import build_self_prompt, generate
from decouple.prompt import prompt_human1, prompt_mapping
from functools import partial

def get_prompt_text(prompt_version, text):
    generic = prompt_human1
    if 'query-specific' in prompt_version:
        reasoning_module_name = prompt_version.split('_')[1]
        reasoning_module = LLMWrapper(reasoning_module_name)
        parts = build_self_prompt(reasoning_module, [dict(type='text', value=text)])
        prompt = generic + 'Especially, pay attention to ' + parts
        if not prompt.endswith('.'):
            prompt += '.'
    elif prompt_version in prompt_mapping:
        prompt = prompt_mapping[prompt_version]
    else:
        prompt = prompt_version
    return prompt

def prepare_perception_inputs(prompt_version, text, tgt_path):
    msgs = []
    if isinstance(tgt_path, list):
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
    else:
        msgs = [dict(type='image', value=tgt_path)]
    prompt = get_prompt_text(prompt_version, text)
    msgs.append(dict(type='text', value=prompt))
    return msgs

class Perception:
    def __init__(self, prompt_version, model):
        self.prompt_version = prompt_version
        self.get_prompt_text = partial(get_prompt_text, prompt_version)
        self.prepare_perception_inputs = partial(prepare_perception_inputs, prompt_version)
        self.model = supported_VLM[model]()
        self.prompt = None
    
    def fetch_prompt(self,text):
        prompt = get_prompt_text(self.prompt_version, text)
        self.prompt = prompt

    def generate(self, text, img_path):
        if self.prompt is None:
            text = self.fetch_prompt(text)
        else:
            text = self.prompt
        msgs = self.prepare_perception_inputs(text, img_path)
        return generate(self.model, msgs)

class Reasoning:
    def __init__(self, model):
        self.model = LLMWrapper(model)

    def generate(self, des, question):
        prompt = build_infer_prompt_external(question, des)
        return self.model.generate(prompt)