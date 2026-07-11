# rhyve-core

[![crates.io](https://img.shields.io/crates/v/rhyve-core.svg)](https://crates.io/crates/rhyve-core)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)

Backend-agnostic core of the [rhyve](../../README.md) hypervisor.

`rhyve-core` is a small `#![no_std]` library that defines the contract between
rhyve and its hardware-virtualization backends (such as
[`rhyve-x86`](../rhyve-x86)). It contains no architecture-specific code.

## What it provides

- **`VcpuBackend<T>`** — the trait a virtualization backend implements to run a
  virtual CPU. `run()` enters the guest and returns an `ExitReason`; the generic
  `T` is the backend's guest-register type.
- **`ExitReason`** — the backend-agnostic result of a VM exit: `Success`,
  `IoInstruction(qualification)` or `Shutdown`.
- **`error::HypervisorError`** — the shared error type.
- **Host-memory injection.** Backends must translate host-virtual to
  host-physical addresses, but *how* is the embedder's job. The embedder
  registers a `HostMemory` implementation once via `set_host_memory(...)`;
  backends then call `virtual_to_physical(...)`:

  ```rust
  struct MyHostMemory;
  impl rhyve_core::HostMemory for MyHostMemory {
      fn virtual_to_physical(&self, vaddr: u64) -> Option<u64> { /* … */ }
  }
  static HOST_MEMORY: MyHostMemory = MyHostMemory;

  rhyve_core::set_host_memory(&HOST_MEMORY);
  ```

  This keeps the backend crates independent of any particular host kernel.

## Usage

`rhyve-core` is part of the rhyve workspace and is consumed as a path
dependency; it is not published to crates.io.

## Licensing

Apache-2.0 OR MIT — see the [workspace README](../../README.md#licensing).
