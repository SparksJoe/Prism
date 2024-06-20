# remap the gpt model name
gpt_version_map = {
    'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
    'gpt-4-0125': 'gpt-4-0125-preview',
    'gpt-4-turbo': 'gpt-4-1106-preview', 
    'gpt-4-0613': 'gpt-4-0613',
    'chatgpt-1106': 'gpt-3.5-turbo-1106',
    'chatgpt-0613': 'gpt-3.5-turbo-0613',
    'chatgpt-0125': 'gpt-3.5-turbo-0125',
    'gpt-4o': 'gpt-4o-2024-05-13'
}

# map the model name to the api type
reasoning_mapping = {
    'llama3-70b-chat':'vllm',
    'Mixtral-8x22B-chat':'vllm',
    'deepseek-chat':'deepseek',
}

# stop_tokens for deploying vllm
stop_tokens = {
    'llama3-70b-chat': ["<|eot_id|>"],
}

mapping = {}
mapping.update(gpt_version_map)
mapping.update(reasoning_mapping)