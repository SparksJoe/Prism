from functools import partial
from collections import defaultdict
from vlmeval.config import supported_VLM
from .model import LLaVA_XTuner_Wrapper
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModel, SiglipImageProcessor, SiglipVisionModel
import os.path as osp


def mapping(key):
    return osp.join('models', key)

def get_llm_kwargs(model_path):
    model_path = model_path.lower()
    llm_kwargs = {}
    if 'internlm2' in model_path or 'prismcaptioner' in model_path:
        llm_kwargs.update(dict(prompt_template='internlm2_chat'))
        if '7b' in model_path or 'prismcaptioner-7b' in model_path:
            llm_kwargs.update(dict(llm_path='internlm/internlm2-chat-7b'))
        if '1_8b' in model_path or 'prismcaptioner-2b' in model_path:
            llm_kwargs.update(dict(llm_path='internlm/internlm2-chat-1_8b'))
    elif 'qwen1.5' in model_path:
        llm_kwargs.update(dict(prompt_template='qwen_chat'))
        if '0.5b' in model_path:
            llm_kwargs.update(dict(llm_path='Qwen/Qwen1.5-0.5B-Chat'))
            
    return llm_kwargs
            
def get_vision_kwargs(model_path):
    model_path = model_path.lower()
    vision_kwargs = {}
    if 'clip' in model_path:
        vision_kwargs.update(dict(
            visual_encoder_path='openai/clip-vit-large-patch14-336',
            vision_encoder_type=CLIPVisionModel,
            visual_select_layer=-2,
            image_processor_type=CLIPImageProcessor,
        ))
    elif 'siglip' in model_path:
        vision_kwargs.update(dict(
            visual_encoder_path='google/siglip-so400m-patch14-384',
            vision_encoder_type=SiglipVisionModel,
            visual_select_layer=-2,
            image_processor_type=SiglipImageProcessor,
        ))
    elif 'internvit' in model_path:
        vision_kwargs.update(dict(
            visual_encoder_path='OpenGVLab/InternViT-6B-448px-V1-5',
            vision_encoder_type=AutoModel,
            visual_select_layer=-1,
            image_processor_type=CLIPImageProcessor,
        ))
    
    return vision_kwargs

def get_model(model_name):
    model_path = mapping(model_name)
    vision_kwargs = get_vision_kwargs(model_path)
    llm_kwargs = get_llm_kwargs(model_path)
    print(f'using followin kwargs:\n{vision_kwargs}\n{llm_kwargs}')
    if 'qlora' in model_name:
        model = partial(LLaVA_XTuner_Wrapper, llava_path=model_path, **vision_kwargs, **llm_kwargs)
    else:
        llm_kwargs.update(dict(llm_path=model_path))
        model = partial(LLaVA_XTuner_Wrapper, llava_path=model_path, **vision_kwargs, **llm_kwargs)
    return model

prismcaptioner_series = {
    'prismcaptioner-7b':partial(
        LLaVA_XTuner_Wrapper, 
        llm_path='internlm/internlm2-chat-7b', 
        llava_path='Yuxuan-Qiao/PrismCaptioner-7B', 
        visual_select_layer=-2, 
        prompt_template='internlm2_chat', 
        visual_encoder_path='google/siglip-so400m-patch14-384', 
        vision_encoder_type=SiglipVisionModel, 
        image_processor_type=SiglipImageProcessor),
    
    'prismcaptioner-2b':partial(
        LLaVA_XTuner_Wrapper, 
        llm_path='internlm/internlm2-chat-1_8b', 
        llava_path='Yuxuan-Qiao/PrismCaptioner-2B', 
        visual_select_layer=-2, 
        prompt_template='internlm2_chat', 
        visual_encoder_path='google/siglip-so400m-patch14-384', 
        vision_encoder_type=SiglipVisionModel, 
        image_processor_type=SiglipImageProcessor),
}

class sft_update(dict):
    
    def __missing__(self, key):
        assert osp.exists(mapping(key))
        value = get_model(key)
        self[key] = value
        return value

supported_VLM.update(prismcaptioner_series)
supported_VLM = sft_update(supported_VLM)