import torch

"""
A function for giving a rough estimate on the amount of memory required to train a model without activation checkpointing. 

Notes: 
1. This function only encodes the amount of memory required for the training process-it does not include the memory required for loading the model itself.
2. This funciton assumes that the intermediate size of the feed-forward network (or MLP) is 4*h where h is the hidden size.

l: number of layers
p: precision
s: sequence length
b: batch size
h: hidden size
a: number of attention heads 
 """
def mem_required(l, p, s, b, h, a):
  total_bytes = l*p*s*b*h*(16+(2/p)+(2*a*s/h)+a*s/(p*h))
  return f"{total_bytes/(1024**3):.2f} GB"

"""
A function for viewing the current amount of memory being used across the current machine's available GPUs.
"""

def mem_status(): 
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        print("Memory status: ")
        for i in range(gpus):
            properties = torch.cuda.get_device_properties(i)
            total_memory = properties.total_memory / (1024 ** 3)  # Convert to GB
            allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)  # Convert to GB
            available_memory = total_memory - reserved_memory
            print(f"GPU {i}:")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Allocated memory: {allocated_memory:.2f} GB")
            print(f"  Reserved memory: {reserved_memory:.2f} GB")
            print(f"  Available memory: {available_memory:.2f} GB")
    else:
        print("No GPU available.")