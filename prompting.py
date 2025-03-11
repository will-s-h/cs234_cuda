import textwrap

PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to guarantee correctness and valid compiilation, and secondarily to get speedups. \n
You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.
"""
PROBLEM_INSTRUCTION = """
Optimize the torch architecture named Model with CUDA operators! \n
Just create the cuda_source and cpp_source strings, as in the example CUDA architecture. Remember to name the main cuda function 'forward', as is done in the example cuda code. Don't add any other thoughts, comments, or pseudocode. Just put the CUDA code inside of the cuda_source string, and create the accompanying cpp_source string that only captures the forward method and the method signature, as is given in the example CUDA output. 
"""


def prompt_generate_custom_cuda(
    arc_src: str, example_torch: str, example_cuda: str, example_torch2: str, example_cuda2: str
) -> str:
    prompt = PROBLEM_STATEMENT

    if example_torch != "" and example_cuda != "":
        prompt += textwrap.dedent(f"""
Here's an example to show you the syntax of the architecture you will see implemented in torch. The example architecture is for element-wise addition: \n
```{example_torch}```\n
The example CUDA output is: \n
```{example_cuda}```\n
Here's another example to show you the syntax of the architecture you could see implemented in torch. The example below is for ReLU: \n
```{example_torch2}```\n
The example CUDA output is:
```{example_cuda2}```
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

    example_torch = textwrap.dedent(
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

    example_cuda = textwrap.dedent('''
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

    example_torch2 = textwrap.dedent('''
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        """
        Simple model that performs a ReLU activation.
        """
        def __init__(self):
            super(Model, self).__init__()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Applies ReLU activation to the input tensor.

            Args:
                x (torch.Tensor): Input tensor of any shape.

            Returns:
                torch.Tensor: Output tensor with ReLU applied, same shape as input.
            """
            return torch.relu(x)

    batch_size = 16
    dim = 16384

    def get_inputs():
        x = torch.randn(batch_size, dim)
        return [x]

    def get_init_inputs():
        return []  # No special initialization inputs needed
    ''')

    example_cuda2 = textwrap.dedent('''
    cuda_source = """
    #include <torch/extension.h>

    __global__ void relu_kernel(const float* a, float* out, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            out[i] = (a[i] > 0) ? a[i] : 0;
        }
    }

    torch::Tensor forward(torch::Tensor a) {
        auto size = a.numel();
        auto output = torch::zeros_like(a);

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(a.data_ptr<float>(), output.data_ptr<float>(), size);

        return output;
    }
    """
    
    cpp_source = (
        "torch::Tensor forward(torch::Tensor a);"
    )
    ''')


    return prompt_generate_custom_cuda(arch, example_torch, example_cuda, example_torch2, example_cuda2) + ("<think>" if add_think else "")