import torch
import torch.distributed as dist
import os

os.environ["NCCL_DEBUG"] = "WARN"
dist.init_process_group("nccl")


rank = dist.get_rank()
torch.cuda.set_device(rank)
world_size = dist.get_world_size()
print(f"world_size: {world_size}")


# Create tensor with size = world_size (so each rank gets 1 element)
x = torch.full((world_size,), rank, dtype=torch.int32, device='cuda')
y = torch.empty_like(x)

print(f"BEFORE rank {rank}: {x}")

# Each rank sends one element to every other rank
dist.all_to_all_single(y, x)

print(f"AFTER rank {rank}: {y}")


# torchrun --nproc_per_node=4 --master_addr=localhost --master_port=54321 dist_all2all_v2.py