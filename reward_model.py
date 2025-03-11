import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import io
import contextlib
import tempfile

def reward_cuda_exception(cuda_exception_text):
    num_errors = 0
    match = re.search(r'(\d+)\s+errors? detected', exception_text)
    if match:
        num_errors = int(match.group(1))
    return -num_errors

def reward_harness_exception(harness_exception_text):
    # currently, return -1 regardless of text
    print('in harness exception')
    return -1

def reward_all_close(all_close):
    return 5 if all_close else 0



def reward(pytorch_functional, cpp_source, cuda_source):
    exec(pytorch_functional)
    inputs = get_inputs()
    for i in range(len(inputs)):
        inputs[i] = inputs[i].cuda()

    try:
        throwaway_buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as build_directory:
            with contextlib.redirect_stderr(throwaway_buffer):
                cuda_mod = load_inline(
                    name=name,
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=["forward"],
                    extra_cflags=[""],
                    extra_ldflags=[""],
                    build_directory=build_directory
                )
    except Exception as cuda_exception:
        saved_cuda_traceback = traceback.format_exception(type(cuda_exception), cuda_exception, cuda_exception.__traceback__)
        cuda_exception_text = ''.join(saved_cuda_traceback)
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