# P2P data-integrity probe — 2026-09-24

Direct measurement of whether GPU↔GPU peer access on `pensive` moves **correct data**, prompted by
the observation that every prior P2P conclusion here rested on topology inference or on bandwidth
timings that were never checked against a known payload.

Host state at probe time: driver **580.178.04** (open kernel module), CUDA **13.0**,
`vllm/vllm-openai:latest` (vLLM 0.27.1, torch 2.13.0+cu130, nvcc 13.0), both GPUs idle.

## Headline

**Peer access is advertised, accepted, and silently discards the data.**
`cudaDeviceCanAccessPeer(0,1)` = `True` both directions; `nvidia-smi topo -p2p r/w/p` = `OK`.
Every peer write then lands nowhere, with **no error returned at any layer**.

| probe | script | result |
|---|---|---|
| `cudaMemcpyPeer` (copy engine) | `p2pprobe.cu` T4 | 1048576/1048576 mismatches — **all zeros** |
| SM-initiated peer store | `p2pprobe.cu` T1 | **all zeros** |
| SM-initiated peer `atomicAdd` | `p2pprobe.cu` T2 | counter 0, expected 16384 |
| NCCL flag pattern (peer write → `__threadfence_system()` → flag) | `p2pprobe.cu` T3 | flag never arrives, 120 s timeout |
| `torch` cross-device `.copy_()` | `verify.py` | **all zeros**, both directions |

Control — same harness, all **CORRECT**:

| control | script |
|---|---|
| same-GPU copy, cuda:0 and cuda:1 | `sanity.py` |
| H2D → D2H, cuda:0 and cuda:1 | `sanity.py` |
| cross-GPU via explicit host staging | `sanity.py` |

So the harness is sound and the fault is specific to the peer path.

## NCCL

All four peer transports hang in `all_reduce` after a clean `init_process_group`
(`ncclmin.py`, 2 MB, 120 s timeout each):

| config | transport selected | result |
|---|---|---|
| default | `P2P/CUMEM` | hang |
| `NCCL_CUMEM_ENABLE=0` | `P2P/IPC` | hang |
| `NCCL_PROTO=Simple` | — | hang |
| `NCCL_PROTO=LL` | — | hang |
| `NCCL_P2P_DISABLE=1` | host-staged | **PASS** |

Hang signature: 100 % GPU utilisation, **0 %** memory utilisation, ~95 W of 300 W — spin-wait on a
flag that never arrives, not work.

Fallback-path all-reduce bus bandwidth (`nccltest.py`, `NCCL_P2P_DISABLE=1`):

| size | latency | busbw |
|---|---|---|
| 1 MB | 0.156 ms | 6.74 GB/s |
| 8 MB | 1.051 ms | 7.98 GB/s |
| 64 MB | 8.329 ms | 8.06 GB/s |
| 256 MB | 33.540 ms | 8.00 GB/s |

vLLM's own `gpu_p2p_access_check` (`vllmp2p.py`) performs a real data transfer and correctly
returns **False** both directions — which is why `--disable-custom-all-reduce` has held.

## The measurement trap

`p2p2.py` timed `torch` cross-device `.copy_()` at **13.89 GB/s** — a clean, plausible Gen3 x16
figure, ~2× the 7.06 GB/s host-staged path — while delivering all zeros. Dropped writes still
consume wire time, so **timing alone endorsed a broken path**. Any future interconnect probe must
assert on a known payload.

## Topology / platform facts captured

- GPU0 `0000:01:00.0`, root complex `0000:00`, NUMA 3 · GPU1 `0000:c5:00.0`, root complex `0000:c0`,
  NUMA 0 → different root complexes (`SYS`). Single-socket EPYC 7663 in **NPS4**, so this is a
  cross-NUMA-*quadrant* crossing over Infinity Fabric, not cross-socket.
- No NVLink (`topo -p2p nvl` = NS); P2P atomics `topo -p2p a` = NS.
- AMD-Vi `Default domain type: Translated`; no `iommu=` flag on `/proc/cmdline`; 71 IOMMU groups,
  each GPU isolated in its own.
- Resizable BAR fully enabled: **BAR1 = 64 GB** on both cards.
- Both GPU root ports advertise `max_link_speed` **8.0 GT/s (Gen3)** while the GPUs advertise
  32.0 GT/s; `nvidia-smi` reports `pcie.link.gen.max=3`.

## Not established

The **root cause was not reached.** Reading ACS / AtomicOp control bits needs root — a non-root
process can read only the first 64 bytes of PCI config space. Untested levers, cost order:

1. `iommu=pt` (one reboot). If ACS `RequestRedirect`/`UpstreamForwarding` is set, peer TLPs are
   forced up to a root complex with no valid peer mapping and dropped — exactly this signature.
2. Re-slotting both GPUs onto root complex `0000:c0`, which exposes five x16 root ports (`c1`–`c5`).
   Removes the fabric crossing entirely.

## Re-running

```bash
docker run --rm --gpus all --ipc=host -v "$PWD/scripts":/w -w /w \
  --entrypoint bash vllm/vllm-openai:latest -lc \
  'nvcc -O2 -arch=sm_120 -o /w/p2pprobe /w/p2pprobe.cu && for t in 4 1 2 3; do timeout 120 /w/p2pprobe $t; done'
```
`verify.py`, `sanity.py`, `ncclmin.py`, `nccltest.py`, `vllmp2p.py` run the same way with
`--entrypoint python3`. Verify **payload**, not bandwidth.
