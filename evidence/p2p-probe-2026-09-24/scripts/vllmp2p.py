import os
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm.distributed.device_communicators.custom_all_reduce import gpu_p2p_access_check
print("=== vLLM gpu_p2p_access_check (real data transfer test) ===", flush=True)
for i,j in [(0,1),(1,0)]:
    try:
        r = gpu_p2p_access_check(i,j)
        print(f"  {i}->{j}: {r}", flush=True)
    except Exception as e:
        print(f"  {i}->{j}: EXC {type(e).__name__}: {e}", flush=True)
