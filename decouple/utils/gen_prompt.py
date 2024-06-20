from vlmeval.api import OpenAIWrapper, OpenAIWrapperInternal
from vlmeval.smp import *
from vlmeval.utils import track_progress_rich
from decouple.utils import description_TSVDataset
from decouple.backend.api import LLMWrapper

example =  '''
Question: In which period the number of full time employees is the maximum?
Contents to observe: the number of full time employees
Question: What is the value of the smallest bar?
Contents to observe: the heights of all bars and their values
Question: What is the main subject of the image?
Contents to observe: the central theme or object
Question: What is the position of the catcher relative to the home plate?
Contents to observe: the spatial arrangement of the objects 
Question: What is the expected ratio of offspring with white spots to offspring with solid coloring? Choose the most likely ratio.
Contents to observe: the genetic information
'''
    
def build_self_prompt(model, message):
    text = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
    task_prompt = f'Your task is to give an concise instruction about what basic elements are needed to be described based on the given question. Ensure that your instructions do not cover the raw question, options or thought process of answering the question.\n'
    prompt = task_prompt + example + 'Question: ' + text + '\nContents to observe: ' 
    res = model.generate(prompt)
    return res
    
def build_self_prompts(dataset, model, nproc=4):
    
    os.makedirs('prompts', exist_ok=True)
    storage = f'prompts/{dataset}_{model}.json'
    if osp.exists(storage):
        print(f'prompts result saved in {storage}')
        return
    model = LLMWrapper(model)
    tmp_file = storage.replace('.json', '.pkl')
    
    dataset_name = dataset
    dataset = description_TSVDataset(dataset)
    if 'mmbench' in dataset_name.lower():
        dataset = dataset.build_mmbench_coreset()
    data = dataset.data
    lt = len(data)
    structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]
    lines = [data.iloc[i] for i in range(lt)]
    tups = [(model, struct) for struct in structs]
    indices = [line['index'] for line in lines]
    
    ans = {}
    if osp.exists(tmp_file):
        ans = load(tmp_file)
    tups = [x for x, i in zip(tups, indices) if i not in ans]
    indices = [i for i in indices if i not in ans]
    
    if len(indices):
        new_results = track_progress_rich(
            build_self_prompt, tups, nproc=nproc, chunksize=nproc,
            keys=indices, save=tmp_file)
        ans = load(tmp_file)
        
        for k, v in zip(indices, new_results):
            assert k in ans 
            assert ans[k] == v and ans[k] == v
    
    tags = defaultdict()
    all_inds = [int(line['index']) for line in lines]
    for k in all_inds:
        tags[k] = ans[k]
    
    dump(tags, storage)
    print(f'prompts result saved in {storage}')