//! AMD-V (SVM) intercept exit codes.
//!
//! On a `#VMEXIT` the processor writes the reason for leaving the guest into the
//! `EXITCODE` field of the VMCB control area. This module names the subset the
//! [`SvmCpu`](super::SvmCpu) backend configures intercepts for.
//!
//! Reference: AMD64 Architecture Programmer's Manual, Volume 2, Appendix C
//! "SVM Intercept Exit Codes".

#![allow(dead_code)]

/// Physical (external) interrupt arrived while the guest ran (`VMEXIT_INTR`).
/// The host services it on the way out (after `stgi`); the guest is then resumed.
pub const INTR: u64 = 0x60;
/// CPUID instruction (`VMEXIT_CPUID`).
pub const CPUID: u64 = 0x72;
/// PAUSE instruction (`VMEXIT_PAUSE`).
pub const PAUSE: u64 = 0x77;
/// HLT instruction (`VMEXIT_HLT`).
pub const HLT: u64 = 0x78;
/// IN/OUT accessing protected ports (`VMEXIT_IOIO`).
pub const IOIO: u64 = 0x7B;
/// RDMSR or WRMSR accessing protected MSRs (`VMEXIT_MSR`).
pub const MSR: u64 = 0x7C;
/// Shutdown (a triple fault, used by the guest to power off) (`VMEXIT_SHUTDOWN`).
pub const SHUTDOWN: u64 = 0x7F;
/// VMRUN instruction (`VMEXIT_VMRUN`).
pub const VMRUN: u64 = 0x80;
/// XSETBV instruction (`VMEXIT_XSETBV`).
pub const XSETBV: u64 = 0x8D;
/// Nested-paging fault (`VMEXIT_NPF`).
pub const NPF: u64 = 0x400;
/// A consistency check on the VMCB failed, so the guest never started
/// (`VMEXIT_INVALID`).
pub const INVALID: u64 = u64::MAX;
