import torch
import torch.distributed as dist
from vlmeval.smp import *
from vlmeval.utils import track_progress_rich
from vlmeval.utils import TSVDataset
from vlmeval.config import supported_VLM
from decouple.frontend import failure_check, description_TSVDataset
from decouple.config import gpt_version_map
from .api import LLMWrapper

INTERNAL = os.environ.get('INTERNAL', 0)

def get_infer_filename(description_file, infer_model):
    file_name = description_file.split('/')[-1]
    model_folder = description_file.replace(f'/{file_name}', '')
    model_name = file_name.replace('_description.xlsx', '')
    infer_folder = osp.join(model_folder, f'{infer_model}')
    os.makedirs(infer_folder, exist_ok=True)
    infer_filename = osp.join(infer_folder, f'{infer_model}.xlsx')
    return infer_filename

def build_infer_prompt_external(question, des):
    if not question.endswith('\n'):
        question += '\n'
    if not question.lower().startswith('question:') and not question.lower().startswith('hint:'):
        question = 'Question: ' + question 
    if not des.endswith('\n'):
        des += '\n'
    description = 'Description: ' + des
    role = 'You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image.\n\n'
    
    prompt =  role + description + question
    return prompt

def build_infer_prompt(line):
    question = DATASET.build_text_prompt(line, dataset=DATASET.dataset)
    if not question.endswith('\n'):
        question += '\n'
    if not question.lower().startswith('question:') and not question.lower().startswith('hint:'):
        question = 'Question: ' + question 
    des = str(line['description'])
    if not des.endswith('\n'):
        des += '\n'
    description = 'Description: ' + des
    role = 'You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image.\n\n'
    
    prompt =  role + description + question
    print(prompt)
    return prompt

def post_auxeval(model, line):
    prompt = build_infer_prompt(line)

    log = ''
    retry = 5
    try:
        res = line['prediction']
        if res != None and 'Failed to obtain answer via API.' not in res:
            return dict(log=log, res=res)
    except:
        pass
    for i in range(retry):
        description = line['description']
        res = model.generate(prompt, temperature=0)
        if res is None:
            log += f'Try {i}: output is {description}, failed to parse.\n'
        elif 'Failed to obtain answer via API' in res:
            time.sleep(10) 
        else:
            log += 'Succeed'
            return dict(log=log, res= res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')

def api_infer_post(data, storage, tmp_file, cfg):

    model_name = cfg.infer_model
    model = LLMWrapper(model_name, verbose=cfg.verbose, max_tokens=256, retry=10)

    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    tups = [(model, line) for line in lines]
    indices = [line['index'] for line in lines]

    ans = {}
    if osp.exists(tmp_file):
        ans = load(tmp_file)
    tups = [x for x, i in zip(tups, indices) if i not in ans]
    indices = [i for i in indices if i not in ans]
    
    if len(indices):
        new_results = track_progress_rich(
            post_auxeval, tups, nproc=cfg.nproc, chunksize=cfg.nproc,
            keys=indices, save=tmp_file)
        ans = load(tmp_file)
        for k, v in zip(indices, new_results):
            assert k in ans 
            assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']
    
    log_map, res_map = {}, {}
    all_inds = [line['index'] for line in lines]
    for k in all_inds:
        log_map[k] = ans[k]['log']
        res_map[k] = ans[k]['res']
    data['prediction'] = [res_map[idx] for idx in data['index']]
    data['post_log'] = [log_map[idx] for idx in data['index']]
    dump(data, storage)

    return storage

# model_infer_post is abandoned, and all reasoning model is called by api_infer_post

# def model_infer_post(data, out_file, cfg):
    
#     res = {}
#     if osp.exists(out_file):
#         res = load(out_file)
    
#     rank, world_size = get_rank_and_world_size()  
    
#     indices = list(range(rank, len(data), world_size))
#     lt = len(indices)
#     data = data.iloc[indices]

#     # If finished, will exit without building the model
#     all_finished = True
#     for i in range(lt):
#         idx = data.iloc[i]['index']
#         if idx not in res:
#             all_finished = False
#     if all_finished:
#         return 
#     data = data[~data['index'].isin(res)]
#     lt = len(data)

#     model_version = cfg.infer_model
#     model = supported_LLM[model_version]() if isinstance(model_version, str) else model_version
    
#     for i in tqdm(range(lt)):
#         idx = data.iloc[i]['index']
#         if idx in res:
#             continue
        
#         prompt = build_infer_prompt(data.iloc[i])
        
#         response = model.generate(prompt)
#         torch.cuda.empty_cache()
        
#         if cfg.verbose:
#             print(response, flush=True)

#         res[idx] = response
#         if (i + 1) % 20 == 0:
#             dump(res, out_file)

#     dump(res, out_file)
#     return model

def infer_post(description_file, cfg):
    logger =  get_logger('INFER')

    global DATASET
    DATASET = description_TSVDataset(cfg.data)
    suffix = description_file.split('.')[-1]
    storage = get_infer_filename(description_file, cfg.infer_model)
    tmp_file = storage.replace('.xlsx', '.pkl')
    logger.info(storage)
    if osp.exists(storage):
        data = load(storage)
    else:
        data = load(description_file)
    
    model = cfg.infer_model
    
    api_models = LLMWrapper.api_models()
    try:
        assert model in api_models
    except:
        max_tokens = model.split('-')[-1]
        assert model.replace(f'-{max_tokens}','') in api_models, 'Unknown reasoning module'
        
    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        logger.info(f'using {model} for post inference')

        api_infer_post(data, storage, tmp_file, cfg)

        if failure_check(storage, 'prediction'):
            infer_post(description_file, cfg)

    if world_size > 1:
        dist.barrier()
            

#     else:
        
#         model_version = model 
#         logger.info(f'using {model_version} for post inference')
        
#         rank, world_size = get_rank_and_world_size()
#         if world_size > 1:
#             torch.cuda.set_device(rank)
            
#         tmpl = tmp_file.replace('.pkl', '_{}' + f'{world_size}.pkl')
#         out_file = tmpl.format(rank)

#         if osp.exists(storage):
#             return storage
            
#         model_infer_post(data, out_file, cfg)
            
#         if world_size > 1:
#             dist.barrier()

#         if rank == 0:
#             data_all = {}
#             for i in range(world_size):
#                 data_all.update(load(tmpl.format(i)))
        
#             data['prediction'] = [str(data_all[x]) for x in data['index']]
#             dump(data, storage)             
#             for i in range(world_size):
#                 os.remove(tmpl.format(i))
        
#     logger.info(f'Description post inference successfully finished, results saved in {storage}')
    return storage
    
def mmbench_infer_post(description_file, cfg):
    rank, world_size = get_rank_and_world_size()
    suffix = description_file.split('.')[-1]
    storage = description_file
    if rank == 0:
        if osp.exists(storage):
            data = load(storage)
        else:
            data = load(description_file)
        mmbench = description_TSVDataset(cfg.data)
        mmbench_data = mmbench.data

        lt = len(data)
        for i in range(lt):
            item = data.iloc[i]
            index = item['index']
            description = item['description']
            mmbench_data.loc[mmbench_data['index'] % 1000000 == index, 'description'] = description

        mmbench_data = mmbench_data[mmbench_data['description'].notna()]
        mmbench_data.pop('image')
        os.rename(storage, storage.replace('.xlsx', '_core.xlsx'))
        dump(mmbench_data, storage)
    if world_size > 1:
        dist.barrier()
    return infer_post(storage, cfg) 
