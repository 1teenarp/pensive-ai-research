import torch
N=1024*1024
ref=(torch.arange(N,dtype=torch.int32)+0x7000000)
print("=== harness sanity (must all be CORRECT) ===")
for dev in (0,1):
    a=torch.arange(N,dtype=torch.int32,device=f"cuda:{dev}")+0x7000000
    b=torch.zeros(N,dtype=torch.int32,device=f"cuda:{dev}")
    b.copy_(a); torch.cuda.synchronize(dev)
    print(f"  same-GPU copy on cuda:{dev}: {'CORRECT' if (b.cpu()==ref).all() else 'CORRUPT'}")
    h=ref.pin_memory(); d=torch.zeros(N,dtype=torch.int32,device=f"cuda:{dev}")
    d.copy_(h); torch.cuda.synchronize(dev)
    print(f"  H2D->D2H on cuda:{dev}   : {'CORRECT' if (d.cpu()==ref).all() else 'CORRUPT'}")
print("\n=== cross-GPU via explicit host staging (the NCCL_P2P_DISABLE=1 path) ===")
a=torch.arange(N,dtype=torch.int32,device="cuda:0")+0x7000000
b=torch.zeros(N,dtype=torch.int32,device="cuda:1")
b.copy_(a.cpu())   # force through host
torch.cuda.synchronize(1)
print(f"  0 -> host -> 1: {'CORRECT' if (b.cpu()==ref).all() else 'CORRUPT'}")
