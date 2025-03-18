import os
import gc
import torch
import torch.nn.functional as F
import torch.optim
from prompting import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_reprompt
from reward_model import reward
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

# clear cache before starting
gc.collect()
torch.cuda.empty_cache()

# dataset
dataset_name = "SakanaAI/AI-CUDA-Engineer-Archive"
cache_dir = "./cache_dir"
os.environ['TORCH_HOME'] = cache_dir
os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir
dataset = load_dataset(dataset_name, cache_dir=cache_dir)
df_l1 = dataset["level_1"].to_pandas()
l1_samples = df_l1[df_l1.Kernel_Name == df_l1.Op_Name]

# hyperparameters
model_str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
short_str = "Qwen-1.5B"
dtype = "auto"
batch_size = 10           # size of our batch (number of prompts)
reprompts  = 2           # number of times we try while including the error message. includes the first prompt
num_problems = 10
temperature = 1
max_new_tokens = 1_000
add_think = True
save_files = True

tokenizer = AutoTokenizer.from_pretrained(model_str, torch_dtype=dtype, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=dtype, attn_implementation='flash_attention_2', cache_dir=cache_dir)
device = torch.device('cuda:0')
model = model.to(device)
if save_files: 
    os.makedirs('./evaluate_outputs', exist_ok=True)

# Initialize wandb
run = wandb.init(
    project="cs234-cuda",  # change to your desired wandb project name
    entity="abcisosm",
    name=f"evaluate-{short_str}-{num_problems}problems",
    config={
        "batch_size": batch_size,
        "reprompts": reprompts,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "add_think": add_think,
        "model": model_str,
        "num_problems": num_problems
    }
)

for p in range(num_problems):
    print(f'problem {p}')
    pytorch_str = l1_samples.iloc[p]['PyTorch_Code_Module']
    batch_outputs = []  # list of lists. batch_outputs[i][j] = ith reprompt, jth sample in batch.
    batch_rewards = []  # similar format list of lists

    batch_outputs = ["" for _ in range(batch_size)]
    prompt = prompt_generate_custom_cuda_from_prompt_template(pytorch_str, add_think=add_think, add_end_think=(not add_think))
    prompts = tokenizer([prompt for _ in range(batch_size)]) # basically dictionary of keys 'input_ids', 'attention_mask'. in list form
    pads = [0 for _ in range(batch_size)]
    tokens_to_ignore_later = [[(0, len(ids))] for ids in prompts['input_ids']]

    for j in range(reprompts):
        print(f'\treprompt attempt {j}')
        inputs = tokenizer.pad(prompts, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )

        # take only the tokens after the inputs
        batch_outputs.append([tokenizer.decode(ids[inputs.input_ids.shape[1]:], skip_special_tokens=True) for ids in gen_ids])
        batch_rewards.append([])
        for i, output_txt in enumerate(batch_outputs[-1]):
            # calculate reward
            print(f'\t\tcalculating reward for item {i} in batch')
            r, msg = reward(pytorch_str, output_txt)
            batch_rewards[-1].append(r)
            print(f'\t\treward: {r}')

            # save files
            if save_files:
                with open(f"./evaluate_outputs/problem_{p}_item_{i}_reprompt_{j}.txt", "w") as file:
                    file.write(output_txt)
                with open(f"./evaluate_outputs/problem_{p}_item_{i}_reprompt_{j}_errmsg.txt", "w") as file:
                    file.write(msg)

            # start reprompt
            reprompt = prompt_generate_reprompt(msg)
            output_ids = tokenizer(output_txt).input_ids
            reprompt_ids = tokenizer(reprompt).input_ids

            orig_tokens = tokens_to_ignore_later[i][-1][1]
            output_tokens = len(output_ids)
            reprompt_tokens = len(reprompt_ids)

            prompts.input_ids[i].extend(output_ids)
            prompts.input_ids[i].extend(reprompt_ids)
            prompts.attention_mask[i].extend([1] * (output_tokens + reprompt_tokens))
            tokens_to_ignore_later[i].append((orig_tokens + output_tokens, orig_tokens + output_tokens + reprompt_tokens))
        
        run.log({'num_reprompts': j, 'mean_reward': sum(batch_rewards[-1]) / batch_size})
        print(f'\tnum_reprompts: {j}, mean_reward: {sum(batch_rewards[-1]) / batch_size}')