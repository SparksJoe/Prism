import torch
import torch.distributed as dist
from vlmeval.utils import TSVDataset, track_progress_rich, split_MMMU
from vlmeval.smp import *
from decouple.utils.build_dataset import description_TSVDataset
from decouple.utils.custom_generate import *
from decouple.prompt import prompt_mapping

from .config import supported_VLM

FAIL_MSG = 'Failed to obtain answer via API.'

def failure_cases(result_file, key, verbose=True):
    data = load(result_file)
    failed_set = []
    data[key] = [str(x) for x in data[key]]
    for idx, pred in zip(data['index'], data[key]):
        if FAIL_MSG in str(pred):
            failed_set.append(idx)

    if verbose:
        print(f'{len(failed_set)} records failed in the original result file {result_file}')
    return failed_set, data

def failure_check(result_file, key, verbose=True):
    failed_set, _ = failure_cases(result_file, key, verbose)
    return len(failed_set)

# Only API model is accepted
def infer_data_api(cfg, index_set):
    rank, world_size = get_rank_and_world_size()   
    assert rank == 0 and world_size == 1
    prompt = prompt_mapping[cfg.prompt_version]
    dataset = description_TSVDataset(cfg.data)
    if 'mmbench' in dataset.dataset.lower():
        dataset = dataset.build_mmbench_coreset()
    dataset.prompt = prompt
    data = dataset.data
    data = data[data['index'].isin(index_set)]

    model_name = cfg.model
    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    is_api = getattr(model, 'is_api', False)
    assert is_api

    lt, indices = len(data), list(data['index'])
    structs = [dataset.build_description_prompt(data.iloc[i]) for i in range(lt)]

    out_file = osp.join(cfg.root_path, f'{model_name}_description_supp.pkl')
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    # using 'coco' dataset for its 'Catption' DATASET_TYPE
    structs = [dict(message=struct, dataset='coco') for struct in structs]
    
    inference_results = track_progress_rich(
        gen_func, structs, nproc=cfg.nproc, chunksize=cfg.nproc, save=out_file, keys=indices)

    res = load(out_file)
    for idx, text in zip(indices, inference_results):
        assert (res[idx] == text if idx in res else True)
        res[idx] = text
    return res

def infer_data(out_file, cfg):
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        dataset = description_TSVDataset(cfg.data)
    if world_size > 1:
        dist.barrier()
    dataset = description_TSVDataset(cfg.data)
    prompt = prompt_mapping[cfg.prompt_version]
    if 'mmbench' in dataset.dataset.lower():
        dataset = dataset.build_mmbench_coreset()
        print(f'building coreset for MMBench-series, {len(dataset)} lines for describing')
    dataset.prompt = prompt
    
    indices = list(range(rank, len(dataset), world_size))
    lt = len(indices)
    data = dataset.data.iloc[indices]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        return 
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[cfg.model]()

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(cfg, set(indices))
        for idx in indices:
            assert idx in supp
        res.update(supp)
        dump(res, out_file)
        return model

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        struct = dataset.build_description_prompt(data.iloc[i])
        # using 'coco' dataset for its 'Catption' DATASET_TYPE
        response = generate(model, message=struct, dataset='coco')

        torch.cuda.empty_cache()
        
        if cfg.verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    dump(res, out_file)
    return model

def infer_data_job(cfg):
    root_path, model_name = cfg.root_path, cfg.model
    result_file = osp.join(root_path, f'{model_name}_description.xlsx')
    rank, world_size = get_rank_and_world_size()   
    tmpl = osp.join(root_path, '{}' + f'{world_size}.pkl')
    out_file = tmpl.format(rank)

    prompt = prompt_mapping[cfg.prompt_version]
    if rank == 0:
        print(prompt)
        
    ignore_failed = cfg.ignore
    
    if '~' in model_name:
        models = model_name.split('~')
        dataset = description_TSVDataset(cfg.data)
        data = dataset.data
        data.pop('image')
        lt = len(data)
        merged_description = [''] * lt
        for model in models:
            model_root_path = root_path.replace(model_name, model)
            model_result = load(osp.join(model_root_path, f'{model}_description.xlsx'))
            merged_description = [model_result.loc[i, 'description'] + merged_description[i] for i in range(lt)]
        data['description'] = merged_description
        dump(data, result_file)
        return result_file
        
    if not osp.exists(result_file):
        
        model = infer_data(out_file, cfg)
        if world_size > 1:
            dist.barrier()
            
        if rank == 0:
            data_all = {}
            for i in range(world_size):
                data_all.update(load(tmpl.format(i)))
                
            dataset = description_TSVDataset(cfg.data)
            if 'mmbench' in dataset.dataset.lower():
                dataset = dataset.build_mmbench_coreset()
            data = dataset.data
            assert len(data_all) == len(data)
            data['description'] = [str(data_all[x]) for x in data['index']]
            data.pop('image')
            dump(data, result_file)
            for i in range(world_size):
                os.remove(tmpl.format(i))
    else:
        if failure_check(result_file, 'description', False) and (not ignore_failed):
            assert rank == 0 and world_size == 1
            failed_set, data = failure_cases(result_file, 'description')
            failed_set = set(failed_set)
            answer_map = {x: y for x, y in zip(data['index'], data['description'])}
            indices = list(data['index'])
            res = infer_data_api(cfg, set(indices))
            answer_map.update(res)
            data['description'] = [str(answer_map[x]) for x in data['index']]
            dump(data, result_file)
    
    if world_size > 1:
        dist.barrier()
    
    if failure_check(result_file, 'description', True) and (not ignore_failed):
        infer_data_job(cfg)
        
    return result_file