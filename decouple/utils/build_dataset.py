from copy import deepcopy
from vlmeval.utils import TSVDataset
from vlmeval.smp import *
from vlmeval.utils.dataset_config import *
from vlmeval.utils.dataset_config import img_root_map, dataset_URLs
from vlmeval.api import OpenAIWrapperInternal,OpenAIWrapper

class description_TSVDataset(TSVDataset):
    
    def __init__(self, dataset='MMBench', skip_noimg=True, prompt=None):
        
        self.prompt = prompt
        self.dataset = dataset
        super().__init__(dataset, skip_noimg)
        
#         use for single image
#         if listinstr(['MMMU'], dataset):
#             filtered_data = self.data.copy()

#             filtered_data['image_path'] = filtered_data['image_path'].apply(lambda x: x[0])
#             filtered_data['image'] = filtered_data['image'].apply(lambda x: x[0])
#             self.data = filtered_data
        
    def build_description_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
                                           
        if 'query-specific' in self.prompt:
            model = self.prompt.split('_')[1]
            prompts_map = load(f'prompts/{dataset}_{model}.json')
            suffix_prompt = prompts_map[str(line['index'])]
            if 'Failed to obtain' in suffix_prompt:
                prompt = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present.'
            else:
                prompt = 'Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present. ' + 'Especially, pay attention to ' + suffix_prompt
                if not prompt.endswith('.'):
                    prompt += '.'
            print(prompt)
        else:
            prompt = self.prompt
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs
                
    def build_text_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        prompt = line['question']
        if DATASET_TYPE(dataset) == 'multi-choice':
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if not question.endswith('\n'):
                question += '\n'
            prompt = question
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            if len(options):
                prompt += options_prompt
                if not prompt.endswith('\n'):
                    prompt += '\n'
                prompt += 'Please select the correct answer from the options above. \n'
        
        return prompt
    
    def display_prompt(self):
        print(self.prompt)
        
    def build_mmbench_coreset(self):
        coreset = deepcopy(self)
        assert 'mmbench' in coreset.dataset.lower(), 'coreset just suits for mmbench'
        data = coreset.data
        data = data[data['index'] < 1000000]
        coreset.data = data
        return coreset
