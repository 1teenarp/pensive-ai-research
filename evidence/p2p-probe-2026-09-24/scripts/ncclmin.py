import os, torch, time, torch.distributed as dist, torch.multiprocessing as mp
def run(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29573")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    print(f"[{rank}] init_process_group done", flush=True)
    t = torch.ones(1024*1024, dtype=torch.float16, device=f"cuda:{rank}")
    print(f"[{rank}] starting all_reduce (2MB)", flush=True)
    st=time.perf_counter()
    dist.all_reduce(t)
    torch.cuda.synchronize()
    print(f"[{rank}] all_reduce OK in {(time.perf_counter()-st)*1e3:.1f} ms  sum={t[0].item()}", flush=True)
    dist.destroy_process_group()
if __name__=="__main__":
    mp.set_start_method("spawn"); mp.spawn(run, args=(2,), nprocs=2, join=True)
