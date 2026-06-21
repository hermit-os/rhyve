# rhyve-x86

x86 hardware-virtualization backend for the [rhyve](../../README.md) hypervisor.

`rhyve-x86` is a `#![no_std]` library that implements
[`rhyve-core`](../rhyve-core)'s `VcpuBackend` trait on x86-64, with two
interchangeable backends:

- **`vmx`** — Intel VT-x: VMXON, VMCS, Extended Page Tables (EPT) and the
  VMLAUNCH/VMRESUME trampoline.
- **`svm`** — AMD-V: host-save area, VMCB, Nested Page Tables (NPT) and the
  VMRUN trampoline.

Both boot a nested guest, intercept the instructions the host must emulate
(CPUID, port I/O, MSR accesses, …) and intercept host *physical* interrupts so
the host stays responsive while the guest runs.

## Public API

- **`check_supported_cpu() -> Result<HypervisorExtension, HypervisorError>`** —
  detects whether the host CPU offers `Vmx` (GenuineIntel + VT-x) or `Svm`
  (AuthenticAMD + SVM).
- **`HypervisorExtension`** — `{ Vmx, Svm }`.
- **`init_guest_memory(guest_slice)`** — writes the guest's boot GDT and
  identity-mapping page tables so it can enter long mode.
- **`vmx` / `svm`** — the backend modules: the per-vCPU types (`VmxCpu` /
  `SvmCpu`), the nested-paging tables (`Ept` / `Npt`) and the shared
  `GuestRegisters` layout.
- **Boot constants** (`GDT_OFFSET`, `BOOT_INFO_OFFSET`, …) describing the guest's
  initial memory layout.

A typical embedder picks a backend from `check_supported_cpu()`, prepares guest
memory with `init_guest_memory()`, and drives the resulting `VcpuBackend`
through `rhyve-core`. Host-virtual → host-physical translation is injected via
`rhyve_core::set_host_memory` (see [`rhyve-core`](../rhyve-core)), so this crate
stays independent of the host kernel.

## Usage

`rhyve-x86` is part of the rhyve workspace and is consumed as a path dependency;
it is not published to crates.io.

## Licensing

Apache-2.0 OR MIT — see the [workspace README](../../README.md#licensing).
