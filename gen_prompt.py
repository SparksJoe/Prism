import argparse
from decouple.utils import build_self_prompts
from vlmeval import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='gpt-4-0125-preview')
    parser.add_argument('--nproc', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data = args.data
    model = args.model
    nproc = args.nproc
    build_self_prompts(data, model, nproc)