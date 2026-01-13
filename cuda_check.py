import torch
print(torch.cuda.memory_allocated()/1024**2, "MB allocated")
print(torch.cuda.memory_reserved()/1024**2, "MB reserved")
print(torch.cuda.memory_summary())
