import argparse

from vlmeval.smp import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data', type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--prompt_version", type=str)
    parser.add_argument("--infer_model", type=str)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument("--nproc", type=int, default=4, help="Parallel API calling")
    parser.add_argument("--ignore", action='store_true', help="Ignore failed indices. ")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--postproc", action='store_true')
    args = parser.parse_args()
    return args

def default_cfg():
    config = dict(
        data='MMStar',
        model='GPT4V',
        prompt_version='generic',
        infer_model='chatgpt-0125',
        nproc=4,
        ignore=True,
        verbose=True,
        postproc=False
    )
    return config

def update_cfg(cfg, args):
    provided_args = {
        arg: value 
        for arg, value in vars(args).items() 
        if (value is not None) and value
    }
    cfg.update(provided_args)
    return cfg

def args2cfg():
    config = default_cfg()
    
    try:
        args = parse_args()
        if args.config is not None:
            config = load(args.config)
        update_cfg(config, args)
    except:
        pass

    return config

class Config:
    pass

def get_cfg():
    cfg_obj = Config()
    config = args2cfg()
    
    for key, value in config.items():
        setattr(cfg_obj, key, value)
        
    return cfg_obj
