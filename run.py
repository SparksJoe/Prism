import torch
import torch.distributed as dist
import datetime
from vlmeval.smp import *
from vlmeval.utils import dataset_URLs, abbr2full, DATASET_TYPE
from vlmeval.smp import *
from vlmeval.evaluate import COCO_eval, YOrN_eval, MMVet_eval, VQAEval, MathVista_eval, LLaVABench_eval, multiple_choice_eval
from decouple.backend import *
from decouple.frontend import infer_data_job
from decouple.args2cfg import get_cfg
from decouple.utils import postproc_eval

def main():
    logger = get_logger('RUN')
    cfg = get_cfg()

    assert cfg.prompt_version is not None, "Set prompt_version for decouple pipeline"

    custom_flag = False

    dataset_name = cfg.data
    if dataset_name not in dataset_URLs:
        dataset_name = abbr2full(dataset_name)

    if dataset_name not in dataset_URLs:
        custom_flag = True

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    model = cfg.model
    prompt_version = cfg.prompt_version
            
    root_path = osp.join('results', f'{prompt_version}', dataset_name, model)
    os.makedirs(root_path, exist_ok=True)
    setattr(cfg, 'root_path', root_path)

    # frontend describing
    logger.info('frontend infering, describing')
    result_file = infer_data_job(cfg)
    logger.info('frontend infering done')
    
    torch.cuda.empty_cache()
    
    if cfg.mode == 'perception':
        return True
    
    # backend infering
    logger = get_logger('INFER')
    logger.info('infering')
    if listinstr(['mmbench'], result_file.lower()):
        logger.info('using mmbench_infer_post')
        result_file = mmbench_infer_post(result_file, cfg)
    else:
        logger.info('using infer_post')
        result_file = infer_post(result_file, cfg)

    logger.info('infering done, evaluting')
    
    if cfg.mode == 'reasoning':
        return True

    # evaluation
    if rank == 0:
        logger = get_logger('EVAL')
        dataset_name = cfg.data
        
        judge_kwargs = {
            'nproc': cfg.nproc,
            'verbose': cfg.verbose,
        }
        
        if DATASET_TYPE(dataset_name) in ['multi-choice', 'Y/N']:
            judge_kwargs['model'] = 'chatgpt-0613'
        elif listinstr(['MMVet', 'MathVista', 'LLaVABench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4-turbo'
                
        if 'OPENAI_API_KEY_JUDGE' in os.environ and len(os.environ['OPENAI_API_KEY_JUDGE']):
            judge_kwargs['key'] = os.environ['OPENAI_API_KEY_JUDGE']
        if 'OPENAI_API_BASE_JUDGE' in os.environ and len(os.environ['OPENAI_API_BASE_JUDGE']):
            judge_kwargs['api_base'] = os.environ['OPENAI_API_BASE_JUDGE']

            
        if cfg.postproc and DATASET_TYPE(dataset_name) == 'multi-choice':
            if dataset_name.endswith('_(F)'):
                dataset_name.replace('_(F)', '')
            else:
                dataset_name = 'default' if custom_flag else dataset_name
            postproc_eval(
                result_file,
                dataset=dataset_name,
                **judge_kwargs)
            
        elif DATASET_TYPE(dataset_name) == 'multi-choice':
            dataset_name = 'default' if custom_flag else dataset_name
            multiple_choice_eval(
                result_file,
                dataset=dataset_name,
                **judge_kwargs)

        elif DATASET_TYPE(dataset_name) == 'Y/N':
            YOrN_eval(
                result_file,
                dataset=dataset_name,
                **judge_kwargs)

        elif DATASET_TYPE(dataset_name) == 'Caption':
            COCO_eval(result_file)
        elif dataset_name == 'MMVet':
            MMVet_eval(result_file, **judge_kwargs)
        elif dataset_name == 'OCRBench':
            OCRBench_eval(result_file)
        elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA'], dataset_name):
            VQAEval(result_file, dataset_name)
        elif listinstr(['MathVista'], dataset_name):
            MathVista_eval(result_file, **judge_kwargs)
        elif listinstr(['LLaVABench'], dataset_name):
            LLaVABench_eval(result_file, **judge_kwargs)
        else:
            logger.error(f'Dataset {dataset_name} is not handled by evaluator, will be skipped. ')

if __name__ == '__main__':
    main()