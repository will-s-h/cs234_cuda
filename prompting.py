import textwrap

PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to guarantee correctness and valid compiilation, and secondarily to get speedups. \n
You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the torch architecture named Model with CUDA operators! \n
Just create the cuda_source and cpp_source strings, as in the example CUDA architecture. Remember to name the main cuda function 'forward', as is done in the example cuda code. Don't add any other thoughts, comments, or pseudocode. Just put the CUDA code inside of the cuda_source string, and create the accompanying cpp_source string that only captures the forward method and the method signature, as is given in the example CUDA output. 
"""


def prompt_generate_custom_cuda(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += textwrap.dedent(f"""
Here's an example to show you the syntax of the architecture you will see implemented in torch: The example architecture is for element-wise addition: \n
``` \n
{example_arch_src}
``` \n
The example CUDA output is: 
```
{example_new_arch_src}
``` \n
""")

    prompt += textwrap.dedent(f"""
You are given the following torch architecture: \n
```
{arc_src}
```
""")
    prompt += PROBLEM_INSTRUCTION
    return prompt

def prompt_generate_custom_cuda_from_prompt_template(ref_arch_src: str, add_think: bool = True) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    example_arch = textwrap.dedent(
    '''
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, a, b):
            return a + b


    def get_inputs():
        # randomly generate input tensors based on the model architecture
        a = torch.randn(1, 128).cuda()
        b = torch.randn(1, 128).cuda()
        return [a, b]


    def get_init_inputs():
        # randomly generate tensors required for initialization based on the model architecture
        return []
    ''')

    example_new_arch = textwrap.dedent('''
    # Define the custom CUDA kernel for element-wise addition
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            out[idx] = a[idx] + b[idx];
        }
    }

    torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
        auto size = a.numel();
        auto out = torch::zeros_like(a);

        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;

        elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

        return out;
    }
    """

    cpp_source = (
        "torch::Tensor forward(torch::Tensor a, torch::Tensor b);"
    )
    ''')


    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch) + ("<think>" if add_think else "")

def parse_cuda_cpp(response):
    output = response[response.find('</think>')+len('</think>'):]
    output = output[output.find('```')+len('```'):output.rfind('```')]
    outputs = output.split('"""')
    cuda_source = outputs[1][1:-1]
    cpp_source = outputs[3][1:-1]
    return cuda_source, cpp_source