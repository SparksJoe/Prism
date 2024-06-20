from .config import mapping

prompt_human1 = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present.'
prompt_human2 = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, background and any text present. Please skip generating statements for non-existent contents and describe all you see. '
prompt_gpt1 = 'Given the image below, please provide a detailed description of what you see.'
prompt_gpt2 = 'Analyze the image below and describe the main elements and their relationship.'
prompt_cot = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present. Let\'s think step by step.'
prompt_decompose = 'Decompose the image into several parts and describe the fine-grained content of the image part by part, including scenes, objects, relationships, instance location, and any text present.'

genric_prompt_mapping = {
    'generic':prompt_human1,
    'human1':prompt_human1,
    'gpt1':prompt_gpt1,
    'gpt2':prompt_gpt2,
    'human2':prompt_human2,
    'cot': prompt_cot,
    'decompose': prompt_decompose,
}

class query_specific(dict):
    
    def __missing__(self, key):
        reasoning_module = key.split('_')[1]
        assert reasoning_module in mapping, f'Unspported prompt or unknown model for reasoning: {reasoning_module}'
        self[key] = key
        return key

prompt_mapping = query_specific(genric_prompt_mapping)