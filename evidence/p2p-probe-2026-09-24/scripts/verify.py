import torch
print("=== correctness of cross-GPU copies (torch) ===")
N = 1024*1024
a = torch.arange(N, dtype=torch.int32, device="cuda:0") + 0x7000000
b = torch.zeros(N, dtype=torch.int32, device="cuda:1")
b.copy_(a); torch.cuda.synchronize(0); torch.cuda.synchronize(1)
ref = (torch.arange(N, dtype=torch.int32) + 0x7000000)
got = b.cpu()
bad = (got != ref).sum().item()
print(f"  torch b.copy_(a) 0->1 : mismatches {bad}/{N}  -> {'CORRECT' if bad==0 else 'CORRUPT'}")
print(f"    expected[0]=0x{ref[0].item():X} got[0]=0x{got[0].item():X}")
print(f"    expected[-1]=0x{ref[-1].item():X} got[-1]=0x{got[-1].item():X}")
print(f"    got unique sample: {got[:8].tolist()}")

print("\n=== is peer access actually enabled in this ctx? ===")
print("  can_device_access_peer(0,1):", torch.cuda.can_device_access_peer(0,1))

# round trip back
c = torch.zeros(N, dtype=torch.int32, device="cuda:0")
c.copy_(b); torch.cuda.synchronize(0); torch.cuda.synchronize(1)
bad2 = (c.cpu() != ref).sum().item()
print(f"  round-trip 1->0       : mismatches {bad2}/{N} -> {'CORRECT' if bad2==0 else 'CORRUPT'}")
