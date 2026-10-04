import os, torch, time, torch.distributed as dist, torch.multiprocessing as mp

def run(rank, world, sizes, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29571")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    out=[]
    for mb in sizes:
        n = mb*1024*1024//2
        t_ = torch.ones(n, dtype=torch.float16, device=f"cuda:{rank}")
        for _ in range(5): dist.all_reduce(t_)
        torch.cuda.synchronize(); dist.barrier()
        st=time.perf_counter()
        for _ in range(20): dist.all_reduce(t_)
        torch.cuda.synchronize()
        dt=(time.perf_counter()-st)/20
        # busbw for ring allreduce = 2*(n-1)/n * size / time
        sz = mb*1024*1024
        algbw = sz/dt/1e9
        busbw = algbw*2*(world-1)/world
        out.append((mb, dt*1e3, algbw, busbw))
    if rank==0: q.put(out)
    dist.destroy_process_group()

if __name__=="__main__":
    mp.set_start_method("spawn")
    sizes=[1,8,64,256]
    q=mp.Queue()
    mp.spawn(run, args=(2,sizes,q), nprocs=2, join=True)
    print(f"{'size':>8} {'lat(ms)':>10} {'algbw':>10} {'busbw':>10}")
    for mb,ms,alg,bus in q.get():
        print(f"{mb:>6}MB {ms:>10.3f} {alg:>8.2f}GB/s {bus:>8.2f}GB/s")
