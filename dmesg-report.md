# Kernel / dmesg Error Report — host `pensive`

**Generated:** 2026-08-17
**Updated:** 2026-09-14 (added network link-flap issue)
**Sources:** `/var/log/dmesg` (current boot, Aug 15 05:56), `/var/log/kern.log` (Aug 15–17), `/var/log/kern.log.1` (Aug 14–15)
**Method note:** `dmesg` itself is denied to non-root (`Operation not permitted`, kernel ring buffer root-only). Report compiled from the persistent logs above.

---

## CRITICAL — Machine Check (MCE) hardware errors

Recurring machine-check exceptions, all "Corrected ECC" (CECC) memory errors. Dominant signal across the entire log history.

| Source | Time range | Count |
|---|---|---|
| `kern.log.1` | Aug 14 09:08 → Aug 15 10:18 | 4,497 MCE events |
| `kern.log` (current boot) | Aug 15–17 | 10 MCE events |
| **Total** | | **≈ 4,507** |

**Per-CPU distribution (combined):**
- CPU 3: **3,888** events
- CPU 2: **1,044** events
- CPU 1: 3 · CPU 0: 6

**Sample error details:**
```
[Hardware Error]: Corrected error, no action required.
[Hardware Error]: CPU:3 (19:1:1) MC18_STATUS[Over|CE|MiscV|AddrV|-|-|SyndV|CECC|-|-|-]: 0xdc2040000000011b
[Hardware Error]: Error Addr: 0x...
[Hardware Error]: IPID: 0x0000009600750f00   // Unified Memory Controller
[Hardware Error]: cache level: L3/GEN, tx: GEN, mem-tx: RD
```
- CPU family/model: **AMD Family 19h Model 1** (EPYC / Ryzen gen).
- ~**4,930 distinct error addresses** logged.
- One harder event in history: `Memory failure: 0x14c5104: ... Recovered` (a memory page failed and was recovered — more severe than corrected ECC).

**Assessment:** Corrected ECC errors are non-fatal and auto-corrected (no corruption yet), but the volume (~4,500 events, two CCDs/controllers, thousands of unique addresses) strongly indicates **degrading/failing RAM**, most likely a DIMM on the memory controller feeding CPU 2/3. At this rate it is a reliability risk.

---

## Other / benign findings

| Item | Severity | Note |
|---|---|---|
| `KHO: Failed to reserve lowmem scratch buffer / scratch area` (5×) | Low | Kexec handover disabled at boot; benign |
| `nvidia: module verification failed ... tainting kernel` | Low | Expected with proprietary NVIDIA driver; `nouveau` blacklisted in cmdline |
| `BERT: [Hardware Error]: Skipped 1 error records` | Low | Firmware error records carried from prior boot (related to MCEs) |
| `ZFS: ... with kernel 7.0.0-29-generic is EXPERIMENTAL and SERIOUS DATA LOSS may occur!` | Warning | Standard OpenZFS message; informational |
| `usb 3-1: Warning! Unlikely big volume range` | Info | USB audio device descriptor quirk; harmless |
| `kvm_amd: Nested Virtualization enabled` | Info | Normal |
| `Disabling lock debugging due to kernel taint` | Info | Consequence of ZFS/nvidia taint; not an error |

---

## Live in-OS diagnostic results (Aug 17, non-root)

Run without sudo (password required, not used). Results from EDAC sysfs + logs:

- **EDAC MC0 (`F19h_M01h`):** `ce_count = 11`, `ue_count = 0`, `seconds_since_reset = 170632` (~47h). No uncorrectable errors since reset.
- **Total system memory:** 1,048,576 MB = **1 TB** RAM (large EPYC server).
- **History (log-derived):** ~4,507 total corrected MCE events, ~4,930 unique addresses, skewed to CPU 2/3 controllers.
- **Tools not runnable without root:** `ras-mc-ctl`, `mcelog`, `edac-util`, `dmidecode` (all require sudo; non-interactive sudo unavailable). Per-csrow EDAC counters not exposed on this AMD platform (rank dirs lack counters), so DIMM slot mapping wasn't obtainable without root/`dmidecode`.

**Net:** All errors remain corrected (UE=0); machine stable but under heavy corrected-ECC load. Pinpointing the exact DIMM requires root (`dmidecode -t memory`) and/or a boot-time `memtest86+`.

---

## Recommendations

1. **Triage memory:** run `memtest86+` (requires reboot), plus `edac-util` / `ras-mc-ctl` and `mcelog` to isolate the failing DIMM (likely on the memory controller feeding CPU 2/3, L3-cache-adjacent). Given thousands of unique addresses, replace the suspect DIMM(s).
2. Errors are corrected so far (no data loss observed), but **back up** given the ZFS experimental note combined with failing memory.
3. Kernel ring buffer is root-only by default; use `sudo dmesg` or `journalctl -k` for live output as non-root.

---

## Network: enp194s0 intermittent link flapping (Sep 14, 2026)

Symptom: intermittent network drops; simple downloads (e.g. huggingface.co) intermittently failing. Host uses `systemd-networkd` (NetworkManager inactive); active interface `enp194s0` (192.168.1.171/24, DHCP). Second NIC `enp193s0` is `NO-CARRIER` (unused).

**Findings:**
- NIC: Intel `igc` `0000:c2:00.0` (I225/I226 2.5GbE), negotiated **2500 Mbps Full Duplex**.
- **143 `NIC Link is Down` events on `enp194s0` in ~24 h** (kernel journal); each flap is down ~4 s then re-negotiates at 2500 Mbps.
- `ethtool -S`: `rx_crc_errors: 21`, `rx_errors: 21`, `collisions: 25` — CRC errors on a *wired* link point to a physical-layer signal-integrity problem.
- `ping 8.8.8.8`: ~20–33% packet loss, loss windows correlating with flaps; pings to the gateway during up-windows show 0% loss.
- No `igc` driver errors/resets in `journalctl -k`; Tailscale logs show repeated major link changes / rebinding following each flap.

Sample:
```
Sep 14 14:04:50 kernel: igc 0000:c2:00.0 enp194s0: NIC Link is Down
Sep 14 14:04:54 kernel: igc 0000:c2:00.0 enp194s0: NIC Link is Up 2500 Mbps Full Duplex, Flow Control: RX
Sep 14 14:04:57 kernel: igc 0000:c2:00.0 enp194s0: NIC Link is Down
...
```

**Assessment:** Physical-layer (cable / connector / switch port), not a software stack problem. 2.5GBASE-T uses 256-QAM across all 4 pairs and has far less signal margin than 1GBASE-T, so a marginal cable/connector flaps at 2.5G but works at 1G. A network-stack restart does not fix this.

**Workaround applied (temporary, Sep 14 ~14:23):**
```
sudo ethtool -s enp194s0 speed 1000 duplex full autoneg on
```
Verified: link renegotiated to **1000 Mb/s Full Duplex**; no further flaps since the 14:23:06 change; 10/10 pings to gateway and 8.8.8.8 (0% loss, was ~20% at 2.5G); huggingface.co HTTP 200 in 0.34 s. CRC/collision counters stopped increasing.

**Caveat:** `ethtool -s` is not persistent across reboot. To make 1G permanent, add a oneshot service:
```
# /etc/systemd/system/enp194s0-1g.service
[Unit]
Description=Force enp194s0 to 1GbE
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/ethtool -s enp194s0 speed 1000 duplex full autoneg on
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```
then `sudo systemctl daemon-reload && sudo systemctl enable --now enp194s0-1g.service`.

**Recommendations:**
1. Monitor ~30–60 min at 1G; if still 0 flaps / 0 new CRC errors, the 2.5G signal margin was the trigger.
2. Fix the physical layer: swap the Ethernet cable, reseat both RJ45 ends, or try a different switch port — then re-negotiate back to 2.5G.
3. If flaps/CRCs recur at 1G, suspect the NIC port or switch port (test against the unused `enp193s0` port or a different switch).
