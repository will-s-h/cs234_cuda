import os
import gc
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD, Adafactor
import torch.optim
from prompting import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_reprompt
from reward_model import reward
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
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

# model
model_q1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_q1,torch_dtype="auto",cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_q1, torch_dtype="auto", attn_implementation='flash_attention_2', cache_dir=cache_dir)
device = torch.device('cuda:1')
model = model.to(device)

# Configure LoRA
lora_config = LoraConfig(
    r=8,                      # rank of the low-rank matrices
    lora_alpha=16,            # scaling factor
    target_modules=["q_proj", "v_proj"],  # adjust based on your model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap the model with LoRA. This freezes the base model parameters and injects trainable adapters.
model = get_peft_model(model, lora_config)

# the batch size hyperparameters
batch_size = 1           # size of our batch (number of prompts)
reprompts  = 2           # number of times we try while including the error message. includes the first prompt

# PPO/training hyperparameters
clip_range = 0.2          # clipping range for PPO
num_iterations = 100      # total training iterations
ppo_epochs = 10           # number of PPO updates per batch
log_prob_min_ratio = -10
log_prob_max_ratio = 5
lr = 1e-5

# text generation hyperparameters
temperature = 1
max_new_tokens = 1_000

# the PyTorch Code Module we're tackling.
problem_id = 0
pytorch_str = l1_samples.iloc[problem_id]['PyTorch_Code_Module']

## create output folder
os.makedirs('./train_outputs', exist_ok=True)

# Initialize wandb
run = wandb.init(
    project="cs234-cuda", 
    entity="abcisosm",
    config = {
        "batch_size": batch_size,
        "reprompts": reprompts,
        "clip_range": clip_range,
        "num_iterations": num_iterations,
        "ppo_epochs": ppo_epochs,
        "log_prob_min_ratio": log_prob_min_ratio,
        "log_prob_max_ratio": log_prob_max_ratio,
        "lr": lr,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens
    }
)

# Helper function to create a mask from a list of ignore ranges.
def create_ignore_mask(seq_len, attention_mask, ignore_ranges, dtype=torch.bfloat16):
    """
    Create a 1D mask (length seq_len) where tokens in any ignore range are set to 0,
    and others are 1.
    ignore_ranges: list of tuples (start, end) with end not included.
    """
    if attention_mask is not None:
        assert len(attention_mask.shape) == 1
        mask = F.pad(attention_mask.to(dtype), (0, seq_len - attention_mask.shape[0]), mode='constant', value=1)
    else:
        mask = torch.ones(seq_len, dtype=dtype)
    for (start, end) in ignore_ranges:
        # Ensure the ignore indices are within bounds
        start = max(0, start)
        end = min(seq_len, end)
        mask[start:end] = 0.0
    return mask

# optimizer = SGD(model.parameters(), lr=lr)
optimizer = Adafactor(model.parameters(), lr=lr)
dtype = torch.bfloat16

exp_mean_reward = -1

for iteration in range(num_iterations):
    print(f'iteration num: {iteration}')
    batch_outputs = []
    batch_rewards = [0 for _ in range(batch_size)]

    ################################
    # (old policy): generate data. #
    ################################

    batch_outputs = ["" for _ in range(batch_size)]
    prompt = prompt_generate_custom_cuda_from_prompt_template(pytorch_str, add_think=False, add_end_think=True)
    prompts = tokenizer([prompt for _ in range(batch_size)]) # basically dictionary of keys 'input_ids', 'attention_mask'. in list form
    pads = [0 for _ in range(batch_size)]
    tokens_to_ignore_later = [[(0, len(ids))] for ids in prompts['input_ids']]
    # format: list of list of tuples. first index is the number of the item within the batch. 
    # for each batch, list of tuples of indices to ignore (start, end) where start is included and end is NOT.

    # variables required for the old_log_prob calculation
    gen_ids = None
    last_attention_mask = None

    for idx in range(reprompts):
        print(f'\treprompt attempt {idx}')
        inputs = tokenizer.pad(prompts, padding=True, return_tensors='pt').to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        if idx == reprompts - 1: # used to calculate the number of pad tokens for the last iteration
            last_attention_mask = inputs.attention_mask

        # take only the tokens after the inputs
        batch_outputs = [tokenizer.decode(ids[inputs.input_ids.shape[1]:], skip_special_tokens=True) for ids in gen_ids]
        for i, output_txt in enumerate(batch_outputs):
            print(f'\t\tcalculating reward for item {i} in batch')

            with open(f"./train_outputs/iteration_{iteration}_item_{i}_reprompt_{idx}.txt", "w") as file:
                file.write(output_txt)
            r, msg = reward(pytorch_str, output_txt)
            run.log({"iteration": iteration, "reward": r, "reprompt": idx, "batch_item": i})
            print(f'\t\treward: {r}')

            with open(f"./train_outputs/iteration_{iteration}_item_{i}_reprompt_{idx}_errmsg.txt", "w") as file:
                file.write(msg)
            batch_rewards[i] += r
            reprompt = prompt_generate_reprompt(msg)

            output_ids = tokenizer(batch_outputs[i]).input_ids
            reprompt_ids = tokenizer(reprompt).input_ids

            orig_tokens = tokens_to_ignore_later[i][-1][1]
            output_tokens = len(output_ids)
            reprompt_tokens = len(reprompt_ids)

            prompts.input_ids[i].extend(output_ids)
            prompts.input_ids[i].extend(reprompt_ids)
            prompts.attention_mask[i].extend([1] * (output_tokens + reprompt_tokens))
            tokens_to_ignore_later[i].append((orig_tokens + output_tokens, orig_tokens + output_tokens + reprompt_tokens))
        
    # compute log probs. ignore any instances of pad_token using the attention mask
    with torch.no_grad():
        outputs = model(gen_ids) # this uses the last instance of gen_ids
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        gen_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)

        # Create a masked log probability for each batch element.
        old_log_probs_list = []
        batch_size, seq_len = gen_ids.shape
        for i in range(batch_size):
            # Create a mask for this batch element using tokens_to_ignore_later.
            # tokens_to_ignore_later[i] is a list of tuples (start, end) to ignore.
            mask = create_ignore_mask(seq_len, last_attention_mask[i], tokens_to_ignore_later[i], dtype=dtype).to(gen_log_probs.device)
            masked_log_probs = gen_log_probs[i] * mask
            # Sum over the sequence to get a scalar log-prob for this example.
            old_log_prob = masked_log_probs.sum()
            old_log_probs_list.append(old_log_prob)
        
        old_log_probs_tensor = torch.stack(old_log_probs_list)
    
    print()
    print(f'rewards: {batch_rewards}')
    if all(r == exp_mean_reward for r in batch_rewards):
        print(f'\tthis batch had the same rewards. not performing gradient steps on this batch.')
        continue
    # update exp_mean_reward
    mean_reward = sum(batch_rewards) / len(batch_rewards)
    exp_mean_reward = 0.8 * exp_mean_reward + 0.2 * mean_reward

    print(f'old log probs: {old_log_probs_tensor}')
    print('done with generations in old policy')
    print()

    rewards_tensor = torch.tensor(batch_rewards, dtype=dtype).to(device)

    ####################################
    # PPO Update Loop with Masking   #
    ####################################
    # In the PPO update loop, it is important that we compute the new log probs
    # over the full concatenated sequence (i.e. the updated prompts) and then apply the same mask.

    for _ in range(ppo_epochs):
        print(f'\tppo_epoch: {_}')

        # Process each batch element individually.
        for i, output_text in enumerate(batch_outputs):
            # Instead of re-tokenizing only the new output, use the entire sequence stored in prompts[i]
            full_ids = torch.tensor(prompts.input_ids[i], device=device).unsqueeze(0)
            outputs = model(full_ids)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            gen_log_probs = log_probs.gather(2, full_ids.unsqueeze(-1)).squeeze(-1)
            # Create the ignore mask using tokens_to_ignore_later for this batch item. Note None since prompts[i] shouldn't have any padding tokens, thus no attention mask necessary
            mask = create_ignore_mask(full_ids.shape[1], None, tokens_to_ignore_later[i], dtype=dtype).to(device)
            # Compute the masked sequence log probability.
            sequence_log_prob = (gen_log_probs[0] * mask).sum()

            ratio = torch.exp(torch.clamp(sequence_log_prob - old_log_probs_tensor[i], log_prob_min_ratio, log_prob_max_ratio))
            advantage = rewards_tensor[i] - exp_mean_reward
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage
            loss = -torch.min(surr1, surr2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_val = loss.item()
            del full_ids, outputs, log_probs, gen_log_probs, sequence_log_prob, ratio, advantage, surr1, surr2, loss
            gc.collect()
            torch.cuda.empty_cache()
            print(f'\t\tloss = {loss_val:.4f}')
            run.log({"iteration": iteration, "loss": loss_val, "ppo_epoch": _, "batch_item": i})

    gc.collect()
    torch.cuda.empty_cache()

    if (iteration + 1) % 10 == 0:
        save_path = f'./train_outputs/model_checkpoint_iter_{iteration + 1}'
        os.makedirs(save_path, exist_ok=True)
        print(f'\tSaving model at iteration {iteration + 1} to {save_path}')
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)