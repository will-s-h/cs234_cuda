import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import traceback
from torch.utils.cpp_extension import load_inline
import io
import contextlib
# import tempfile
import random
import string
import os
import shutil

def parse_cuda_cpp(response):
    output = response
    parse_error = False
    end_think_idx = response.find('</think>')
    if end_think_idx >= 0:
        output = response[end_think_idx+len('</think>'):]
    else:
        parse_error = True
    
    CUDA_START_PHRASE, CUDA_END_PHRASE = 'cuda_source = """', '"""'
    cuda_start = output.find(CUDA_START_PHRASE) + len(CUDA_START_PHRASE)
    cuda_end = output.find(CUDA_END_PHRASE, cuda_start)
    if cuda_start < len(CUDA_START_PHRASE) or cuda_end < 0:
        parse_error = True
    cuda_source = output[cuda_start:cuda_end]

    CPP_START_PHRASE, CPP_END_PHRASE = 'cpp_source = (\n    "', '"\n)'
    cpp_start = output.find(CPP_START_PHRASE) + len(CPP_START_PHRASE)
    cpp_end = output.find(CPP_END_PHRASE, cpp_start)
    if cpp_start < len(CPP_START_PHRASE) or cpp_end < 0:
        parse_error = True
    cpp_source = output[cpp_start:cpp_end]
    return cuda_source, cpp_source, parse_error

### rewards

def reward_failed_parse(parse_exception_text):
    return -30, "The outputted CUDA kernel was not formatted correctly. Please follow the format of the examples given!"

def reward_cuda_exception(cuda_exception_text):
    num_errors = 1
    match = re.search(r'(\d+)\s+errors? detected', cuda_exception_text)
    if match:
        num_errors = int(match.group(1))
    return -num_errors, cuda_exception_text

def reward_harness_exception(harness_exception_text):
    # currently, return -1 regardless of text
    return -1, harness_exception_text

def reward_all_close(all_close):
    return (5 if all_close else 0), ("Correct!" if all_close else "The output of the CUDA code did not match the PyTorch functional. Try fixing the correctness.")


def reward(pytorch_functional, response, CACHE_DIR="./cache_dir"):
    # first, attempt parse
    try:
        cuda_source, cpp_source, parse_error = parse_cuda_cpp(response)
    except Exception as parse_exception:
        saved_parse_exception = traceback.format_exception(type(parse_exception), parse_exception, parse_exception.__traceback__)
        parse_exception_text = ''.join(saved_parse_exception)
        return reward_failed_parse(parse_exception_text)
    if parse_error:
        return reward_failed_parse("one of the .find() calls in our parser failed!")

    exec(pytorch_functional, globals())
    inputs = get_inputs()
    for i in range(len(inputs)):
        inputs[i] = inputs[i].cuda()

    try:
        throwaway_buffer = io.StringIO()
        random_sequence = ''.join(random.choice(string.ascii_letters) for _ in range(20))
        with contextlib.redirect_stderr(throwaway_buffer):
            cuda_mod = load_inline(
                name=random_sequence,
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=["forward"],
                extra_cflags=[""],
                extra_ldflags=[""],
            )
        cache_dir = f'{CACHE_DIR}/{random_sequence}'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception as cuda_exception:
        saved_cuda_traceback = traceback.format_exception(type(cuda_exception), cuda_exception, cuda_exception.__traceback__)
        cuda_exception_text = ''.join(saved_cuda_traceback)
        cache_dir = f'{CACHE_DIR}/{random_sequence}'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        return reward_cuda_exception(cuda_exception_text)

    try:
        class ModelNew(nn.Module):
            def __init__(self) -> nn.Module:
                super().__init__()
                self.cuda_mod = cuda_mod

            def forward(self, xs):
                return self.cuda_mod.forward(*xs)

        tmod = Model()
        cmod = ModelNew()
        all_close = torch.allclose(tmod.forward(*inputs), cmod.forward(inputs), rtol=1e-1, atol=1e-3)
        return reward_all_close(all_close)
    except Exception as harness_exception:
        saved_harness_traceback = traceback.format_exception(type(harness_exception), harness_exception, harness_exception.__traceback__)
        harness_exception_text = ''.join(saved_harness_traceback)
        return reward_harness_exception(harness_exception_text)