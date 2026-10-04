import torch, time
N = 512*1024*1024
def sync(): torch.cuda.synchronize(0); torch.cuda.synchronize(1)

a = torch.empty(N//2, dtype=torch.float16, device="cuda:0")
b = torch.empty(N//2, dtype=torch.float16, device="cuda:1")
h = torch.empty(N//2, dtype=torch.float16, pin_memory=True)

def timeit(fn, iters=20):
    for _ in range(5): fn()
    sync(); t=time.perf_counter()
    for _ in range(iters): fn()
    sync(); return N*iters/(time.perf_counter()-t)/1e9

print("=== direct peer copy (P2P path) ===")
print(f"  0->1 : {timeit(lambda: b.copy_(a)):.2f} GB/s")

print("=== manual host-staged copy (D2H then H2D) ===")
def staged():
    h.copy_(a); b.copy_(h)
print(f"  0->h->1: {timeit(staged):.2f} GB/s")

print("=== bidirectional simultaneous (full-duplex check) ===")
a2 = torch.empty(N//2, dtype=torch.float16, device="cuda:0")
b2 = torch.empty(N//2, dtype=torch.float16, device="cuda:1")
s0 = torch.cuda.Stream(device=0); s1 = torch.cuda.Stream(device=1)
def bidir():
    with torch.cuda.stream(s0): b.copy_(a, non_blocking=True)
    with torch.cuda.stream(s1): a2.copy_(b2, non_blocking=True)
for _ in range(5): bidir()
sync(); t=time.perf_counter()
for _ in range(20): bidir()
sync(); dt=time.perf_counter()-t
print(f"  aggregate: {2*N*20/dt/1e9:.2f} GB/s  (per-direction {N*20/dt/1e9:.2f})")
