import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    # init
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

    # Input tensor, just for demonstration
    inp = torch.arange(4) + rank * 10       # rank0=[0,1,2,3], rank1=[10,11,12,13]
    out = torch.empty_like(inp)

    # all_to_all_single always split on dim 0 (by default split equally by world_size)
    dist.all_to_all_single(out, inp)

    print(f"Rank {rank} input:  {inp.tolist()}")
    print(f"Rank {rank} output: {out.tolist()}\n")

    dist.destroy_process_group()


def main():
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
