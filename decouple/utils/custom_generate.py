import re
import copy as cp
import string
from vlmeval.smp import *
from vlmeval.utils.matching_util import *
from vlmeval.utils import split_MMMU

from decouple.prompt import *

def get_img_list(message):
    img_list = []
    for item in message:
        if item['type'] == 'image':
            img_list.append(item)
    return img_list

def get_text(message):
    for item in message:
        if item['type'] == 'text':
            return item

def single_turn(model, message, dataset):
    response = model.generate(message=message, dataset=dataset)
    return response

def multi_turn(model, message, dataset):
    img_list = get_img_list(message)
    text = get_text(message)
    response = ''
    lt = len(img_list)
    for i in range(lt):
        message_s = [img_list[i], text]
        response += f'\nimage {i+1}: '
        response += single_turn(model, message_s, dataset)
    return response

def generate(model, message, dataset=None):
    img_list = get_img_list(message)
    if len(img_list) == 1:
        response = single_turn(model, message, dataset)
    else:
        print(f'splitting {message}')
        response = multi_turn(model, message, dataset)
    return response

def api_single_turn(model, structs, dataset):
    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset) for struct in structs]
    return gen_func, structs

def api_multi_turn(model, structs, dataset):
    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset) for struct in structs]
    return gen_func, structs

def api_generate(model, message, dataset=None):
    gen_func, structs = api_single_turn(model, message, dataset)
    return gen_func, structs