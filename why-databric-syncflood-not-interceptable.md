# Why a "data-fabric sync flood" can't be gracefully intercepted — and why the system resets instead

Date: 2026-09-06
Purpose: an educational deep-dive the user requested on WHY an uncorrectable memory/fabric error
escalates to a hard system reset, and WHY there is no software "gate" that can intercept it and fail
gracefully — even though multiple layers of abstraction exist that *would* ideally do so. This doc also
answers the "is this a cascade/snowball that can't be stopped without power-off?" question.

> The concrete recurring fault on this box: **data-fabric sync-flood reset** (`reset reason 0x08000a00`,
> "uncorrected error caused a data fabric sync flood"), traced to a marginal DIMM on **channel G / slot MM4**
> (see `power-trip-instances.md`). This document explains the mechanism generically + specifically.

---

## 1. The failure is in the *memory subsystem*, not the app

The model server (vLLM) is only the **stimulus**. The actual fault is **hardware**: a marginal DIMM
(RAM stick) on channel G throws a memory error. The software (weights, KV cache, PLE offload) just
happens to hammer that failing region hard — which is why the trips correlate with heavy load.

The **layered system** that sits between the application and the hardware:

```
  APPLICATION  (vLLM serving)
        │
  LANGUAGE RUNTIME  (Python / PyTorch / CUDA)
        │
  DRIVER / CUDA runtime  (NVIDIA driver)
        │
  OS KERNEL  (Linux: VMM/MMU, page tables)
        │
  CPU ARCH  (memory management, caches, MMU)
        │
  MEMORY CONTROLLER  (integrated in CPU, "IMC")
        │
  DATA FABRIC / INFINITY FABRIC  (CPU <-> DDR <-> PCIe <-> GPU)
        │
  DIMM (RAM stick) / GPU HBM
```

Note: **memory is not "owned" by software** — it's directly attached to the CPU's memory controller.
The application addresses virtual memory; the kernel maps it; the CPU's MMU translates it; the memory
controller reads/writes the DRAM cells. When DRAM returns bad data, **no software above it ever sees a
"this was an error" signal until after the fact.**

---

## 2. The two kinds of memory errors (and why one is survivable, the other isn't)

- **Correctable Error (CE / CECC):** DRAM returns a bad bit, but ECC (Error-Correcting Code) is
  sufficient to **fix it on the fly** and hand the CPU correct data. The hardware recovers. The EDAC
  subsystem logs it, but no crash occurs. This is what your box logs constantly (e.g.
  `ras-mc-ctl` shows **hundreds/thousands of CE** on channel 6/7 and channel G).
- **Uncorrectable Error (UE / UECC):** ECC cannot fix it. The memory controller gets data it cannot
  trust to be correct. This is fatal *for the data* because it may have already been used to compute.

**The key asymmetry:** a CE is recoverable because the bad data never escapes — it's corrected before
it's consumed. A UE is **not** recoverable because the corrupted value may already be in a register,
a cache line, or mid-computation. It cannot be "rolled back" — there's no memory-versioning of a DRAM
read at the hardware level.

---

## 3. Why the UE triggers a *system-wide* reset rather than a graceful error

This is the crux of your question. When a UE occurs:

1. The memory controller signals a **Machine Check** (AMD calls this an **SMI/MCA event**, part of the
   **Scalable MCA (SMCA)** framework — a hardware exception mechanism in the CPU).
2. The CPU raises a **Machine Check Exception (MCE)**. For a *correctable* error, the system quiesces
   and continues. For an **uncorrectable** one, the CPU often cannot produce a trustworthy result.
3. **The catch:** many UEs are *detected after* the corrupted data has already been consumed or when the
   integrity of the machine state is in question. The CPU/hardware cannot guarantee the registers,
   cache, or pipeline contain valid data. Continuing to execute on a poisoned machine is dangerous —
   it can silently corrupt filesystems, databases, or firmware.

So the hardware's design philosophy is: **"If I can't be sure the machine is in a correct state, stop
the whole thing rather than risk running on garbage."** That's the rational choice — better a reboot
than silent corruption.

### Why "data fabric sync flood" specifically

AMD EPYC's Infinity Fabric ties together the CPU cores, memory controllers, and PCIe. A UE on the
**data fabric** (the interconnect) is especially bad because the fabric is the *bus* everything rides on.
A "sync flood" means the fabric's synchronization/validity checks failed — the fabric cannot trust its
own internal coherence. This is a **fabric-level fatal error**, not just a single cache line.

---

## 4. Why there is NO software gate that can catch it (your honest question)

You're right to expect a "gate." There ARE layers that *would* be the ideal place to intercept — but
each is structurally unable to stop a data-fabric UE:

| Layer | What it could catch | Why it can't stop a fabric UE |
|---|---|---|
| **Application (vLLM)** | Nothing — it never sees a memory error signal | Software reads/writes are just bus transactions; it has no idea a DRAM read was wrong until the result is wrong (and by then it's often too late, and even a "wrong result" isn't distinguishable from a legitimate value). |
| **Runtime/CUDA** | Nothing | Same — CUDA/PyTorch don't get notified of host-memory ECC faults directly; for attached GPU memory, die-level errors are different. |
| **OS kernel (MCE handler)** | An MCE *event* | The kernel's `mce`/`ras` handler DOES receive the MCE, logs it, and for a **correctable** error can allow continued operation. But for an **uncorrectable** error it **cannot** (a) undo the bad memory, or (b) safely keep running. Historically the kernel either **(a) panics** (which is a "graceful-ish" stop) or **(b) lets the hardware reset** because the machine state is poisoned. On AMD, the **sync-flood** path is a **hardware** decision the kernel cannot override. |
| **BIOS/firmware (SMCA)** | Machine-check configuration, thresholds | Firmware sets up the threshold scoring (e.g. CE threshold → offline a DIMM), but a **single UE** is an immediate fatal condition that firmware is not designed to "recover" from. |
| **Memory controller / SMCA hardware** | Detects the error | This is the *source* of the detection. It can correct CEs, but for a fatal fabric UE it **asserts a reset** because the machine can't be trusted. This is the lowest, most authoritative layer — by definition it can't "catch" an error it IS the one declaring fatal. |

**Bottom line:** the "gate" that would ideally intercept the failure **is the memory controller + MCA
hardware**, and it **does** do its job — it *detects* the error and *chooses the safest action*. But the
**safest action for an uncorrectable data-integrity error is to reset**, not to continue. There is no
layer that can "fail gracefully" because no layer above the hardware can reconstruct trustworthy state.
The gate exists, but its output IS "power down," not "continue safely."

---

## 5. Is it a "snowball that can't be stopped without power-off"? — Yes, and it's deliberate.

Your instinct is correct. Once an uncorrectable fault is detected:

1. The hardware's **integrity contract is broken** — it cannot prove machine state validity.
2. **Continuing risks silent, permanent data corruption** (filesystems, VMs, the very model weights,
   or the BIOS/SPI flash).
3. A **clean reset** (reboot) is the only action that guarantees returning to a known-good state.

This is **not** an uncontrolled failure — it's the **designed fail-safe**. Most electronics reset on a
fundamental fault precisely to avoid heating up or corrupting further. An AMD CPU resetting on a
data-fabric UE is analogous to a consumer PSU's over-current shutdown or a CPU's thermal `THERMTRIP` —
it's a *protective* intervention, not a malfunction.

**So the cascade IS "unstoppable without power-off" — and that is the CORRECT design choice.** Stopping
mid-fault is strictly better than continuing on a machine that may be lying about its own state.

---

## 6. What the *correct* mitigations are (and what they do)

Since the fault cannot be intercepted mid-flight, the only strategies are:

1. **Prevent the fault from occurring** (fix the cause):
   - **Replace/reseat the marginal DIMM (channel G / MM4)** — the actual root cause.
   - **Downclock RAM** (3200 → 2933/2666) to give the DRAM timing margin.
   - **Verify with memtest86** on channel G.
   - **Reduce memory/fabric stress** (serialize weight-loading, PLE offload, GPU power cap) — reduces how
     hard the failing DIMM is hit, which **lowers the frequency** of CE/UEs. This is a *probability
     reduction*, not a prevention of the underlying hardware flaw.
2. **Detect earlier (so fewer UEs reach the reset):** the CE→UE progression means the hardware often
   "warns" via Correctable Errors first. **Monitoring CE counts** (rasdaemon / `ras-mc-ctl`) and alerting
   on a threshold lets you **proactively swap the DIMM before it escalates to a fatal UE.** This is the
   closest thing to a "gate" you can actually exploit — it's a **predictive, preventive** gate, not a
   mid-fault one.
3. **Reduce blast radius** (if you must run on failing memory): minimize writes to critical storage, take
   snapshots, keep model weights cached/re-downloadable. But this is damage-control, not a fix.

---

## 7. The essential mental model

- **CE = warning.** Hardware corrected it; the machine continues. Watch the count — it's your early-warning
  gate. A CE **burst** (e.g. hundreds on one channel, as on channel 6/7 here) is the smoke before the fire.
- **UE = the fire.** Hardware cannot correct it; the machine cannot trust itself; it **resets**. No software
  layer above can prevent this because the failure is at the memory-interconnect level and there is no
  way to reconstruct a trustworthy state.
- The **power-off/reset is the manufacturer's fail-safe**, and it is working as designed. The real task is
  to prevent the fault (fix the DIMM) and detect the warning sign (CE escalation) *before* the fail-safe
  fires.

---

## 8. Summary answers

- **Why no gate at any layer?** Because the failure is at the memory-controller/interconnect ("data fabric")
  level; the only layer that can act is the hardware itself, and the safe action for an uncorrectable
  data-integrity error is to reset. There is no software construct (filesystem, process, VM, language
  runtime) that can reconstruct a corrupt machine state.
- **Is it a snowball that can't be stopped without power-off?** Yes — and that's the correct, protective
  behavior. Continual execution on a machine that may be silently corrupting data is worse than a reboot.
- **What's the actionable "gate"?** Not mid-fault, but *pre-fault*: monitor Correctable-ECC counts and act
  (swap the DIMM) *before* it becomes an uncorrectable error. That's the only interception point you
  realistically have.

---

## 9. CE vs UE — what ECC actually does, and why one is survivable and the other isn't

**ECC (Error-Correcting Code)** stores **redundant parity bits** alongside the data in each DRAM word.
Depending on the scheme, ECC can detect some number of bit errors, and correct a subset of those.

- **CE (Correctable Error):** the number of flipped bits is **within what ECC can correct**. The memory
  controller **fixes the value on the fly** and hands the CPU *correct* data. Nothing crashes; the fault is
  logged by EDAC. This is exactly the **2745 "Corrected error, no action required"** you see — ECC doing
  its job, silently.
- **UE (Uncorrectable Error):** the bit-flip pattern **exceeds what ECC can correct** (e.g. too many bits
  in a word, or a multi-bit pattern beyond the code's reach). ECC can **detect** the data is wrong but
  **cannot reconstruct the original**. The corrupt value is essentially "delivered as-is" conceptually.

**Why can't the DIMM just flush the corrupt part and crash only the using process?** Sometimes it CAN,
via Linux page-poisoning + `SIGBUS`: if the kernel learns a *specific physical page* holds an
uncorrectable error, it marks that page poisoned and signals the process that mapped it → that process
crashes, the OS survives. But this only works when **both** hold:
1. **Attribution:** the kernel knows *which page* is bad (hardware reports a page/address).
2. **Timing:** the bad value hasn't already been consumed into a register/cache and used in a computation
   (you can't "un-compute" with a poisoned operand).

When either is absent (and for a fabric fault both usually are), the kernel has **nothing to point at**,
so there's no process to kill and no page to poison. Only a system reset is safe.

---

## 10. Why a "flood in the lanes" is worse than a bad DIMM cell — and what the mechanism is

The **data fabric (Infinity Fabric)** is the shared interconnect: CPU cores ↔ memory controllers ↔ PCIe ↔
GPUs. A **sync-flood** is NOT "one corrupted value"; it is the **interconnect losing its own
synchronization/validity handshake.** The JPEG-in-header analogy is apt: a bad *pixel* (data cell UE) is
identifiable and droppable; a corrupted *container/header* (fabric coherence) makes the whole stream
unparseable.

Why a fabric flood is worse:

1. **No locality / no culprit.** The fault doesn't say "this DIMM bit is bad"; it says
   "communication integrity between components broke." There is **no address** of what was affected.
   This is why it can't be mapped back to a process — there's nothing to attribute.
2. **Systemic / compounding, not isolated.** Because the fabric is shared, a coherence failure makes
   **every** component's view of memory suspect: the CPU can't trust reads from RAM *or* writes to a GPU
   over PCIe. The machine may compute/store using data it cannot verify.
3. **No safe partial recovery.** Since you can't identify *which* data is suspect, you can't flush just
   the corrupt part. The only safe action is to discard the **entire machine state** → reset.

**Crucial distinction:** an *isolated* CE/UE (bad DIMM cell) is sometimes handlable (ECC corrects a CE; a
page-addressable UE gets poisoned + a process gets SIGBUS). A *fabric coherence flood* is not a "thing you
can point at," so there is nothing to flush but the whole system. That is the fundamental difference.

---

## 11. Is it really a faulty DIMM (hardware)? — Yes, and this is a legitimate warranty case

Yes — by "faulty DIMM" we mean the **physical RAM stick**. Strong evidence:

- **CE counts are massively concentrated on specific channels**, not spread randomly:
  from `ras-mc-ctl --summary` (this box): channel#6 = **1367** and channel#6 (row1) = **930**, channel#7 = 310/5,
  vs. single-digit to tens on others. A single channel/slot accumulating **hundreds→thousands** of
  corrected errors at a much higher rate than all peers is the textbook signature of a **marginal/failing DIMM**.
- **Slot↔channel map (DMI/dmidecode):** `A=MM7 B=MM5 C=MM3 D=MM1 E=MM8 F=MM6 G=MM4 H=MM2`.
  So **EDAC channel#6 = Channel G = slot MM4** (the prime suspect), channel#7 = channel H = slot MM2.
- **Escalation to uncorrectable:** corrected errors → an uncorrectable → `reset reason 0x08000a00`
  ("uncorrected error caused a data fabric sync flood"), recorded across multiple trips.

### Vendor / RMA claim checklist
1. `recipe/diag-dimm-fault.sh` output (CE-heavy channel + mapped slot).
2. `ras-mc-ctl --summary` (per-channel CE counts).
3. `grep "Previous system reset" /var/log/syslog` (`0x08000a00` uncorrectable records).
4. Optionally **memtest86** on the identified slot — note the fault is **intermittent / load/heat
   dependent**, so a short memtest pass may not reproduce it; mention that the box reproduces under
   sustained load (heavy memory/fabric usage).
5. State that it is a **single-slot DIMM** fault (concentrated CE on channel G), not a board-wide issue,
   which makes it an RMA candidate for the RAM stick itself.
