from datasets import load_dataset
import os

# dataset
dataset_name = "SakanaAI/AI-CUDA-Engineer-Archive"
cache_dir = "./cache_dir"
os.environ['TORCH_HOME'] = cache_dir
os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir
dataset = load_dataset(dataset_name, cache_dir=cache_dir)
df_l1 = dataset["level_1"].to_pandas()
l1_samples = df_l1[df_l1.Kernel_Name == df_l1.Op_Name]

print(l1_samples.iloc[0]['PyTorch_Code_Module'])

breakpoint()