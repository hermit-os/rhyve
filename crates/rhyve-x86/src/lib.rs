//! Hardware-virtualization backend for the rhyve hypervisor.
//!
//! This library contains the architecture-specific part of the hypervisor: the
//! Intel VT-x (VMX) backend together with the backend-agnostic abstractions it
//! implements ([`Vm`], [`Cpu`] and their traits). The `rhyve` binary drives it,
//! providing the guest image, boot information and the hermit runtime glue.

#![no_std]

#[macro_use]
extern crate log;
extern crate alloc;

pub mod vmx;

use core::mem::MaybeUninit;

use raw_cpuid::CpuId;
use rhyve_core::error::HypervisorError;

#[allow(dead_code)]
const EFER_SCE: u64 = 1; /* System Call Extensions */
const EFER_LME: u64 = 1 << 8; /* Long mode enable */
const EFER_LMA: u64 = 1 << 10; /* Long mode active (read-only) */
#[allow(dead_code)]
const EFER_NXE: u64 = 1 << 11; /* PTE No-Execute bit enable */

const GDT_KERNEL_CODE: u16 = 1;
const GDT_KERNEL_DATA: u16 = 2;
pub const GDT_OFFSET: u64 = 0x1000;
/// Guest-physical address of the boot PML4.
const BOOT_PML4: u64 = 0x10000;
/// Guest-physical address of the boot information.
pub const BOOT_INFO_OFFSET: u64 = 0x9000;
#[allow(dead_code)]
const BOOT_GDT_NULL: usize = 0;
#[allow(dead_code)]
const BOOT_GDT_CODE: usize = 1;
#[allow(dead_code)]
const BOOT_GDT_DATA: usize = 2;
const BOOT_GDT_MAX: usize = 3;

/// Guest-physical address of the boot page-directory-pointer table.
const BOOT_PDPTE: u64 = BOOT_PML4 + 0x1000;
/// Guest-physical address of the boot page directory.
const BOOT_PDE: u64 = BOOT_PML4 + 0x2000;
/// Initial guest stack pointer (grows down, below the loaded kernel).
pub const BOOT_STACK_TOP: u64 = 0x70000;

/// Page-table/GDT entry flags.
pub const PG_PRESENT: u64 = 1 << 0;
pub const PG_RW: u64 = 1 << 1;
pub const PG_HUGE: u64 = 1 << 7;

/// HypervisorExtension indicates the support of hardware
/// extension to accelerate a virtual machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HypervisorExtension {
	/// Support for Intel VT-x (VMX) is available.
	Vmx,
	/// Support for AMD-V (SVM) is available.
	Svm,
}

/// Initializes the guest's boot memory: a flat GDT and 4-level page tables that
/// identity-map the first 1 GiB of guest-physical memory with 2 MiB pages.
///
/// The guest enters in long mode with `CR3 = BOOT_PML4`, so these structures
/// must already be present before the first instruction executes.
pub fn init_guest_memory(guest_slice: &mut [MaybeUninit<u8>]) {
	let base = guest_slice.as_mut_ptr() as *mut u8;
	let write_entry =
		|off: u64, val: u64| unsafe { (base.add(off as usize) as *mut u64).write(val) };

	// Boot GDT: null, 64-bit ring-0 code, ring-0 data.
	write_entry(GDT_OFFSET, 0);
	write_entry(GDT_OFFSET + 8, 0x00AF_9B00_0000_FFFF);
	write_entry(GDT_OFFSET + 16, 0x00CF_9300_0000_FFFF);

	// PML4 and PDPT: a single entry each, pointing at the next level.
	for i in 0..512 {
		write_entry(BOOT_PML4 + i * 8, 0);
		write_entry(BOOT_PDPTE + i * 8, 0);
	}
	write_entry(BOOT_PML4, BOOT_PDPTE | PG_PRESENT | PG_RW);
	write_entry(BOOT_PDPTE, BOOT_PDE | PG_PRESENT | PG_RW);

	// Page directory: 512 × 2 MiB pages identity-mapping the first 1 GiB.
	for i in 0..512 {
		write_entry(BOOT_PDE + i * 8, (i << 21) | PG_PRESENT | PG_RW | PG_HUGE);
	}
}

/// Checks whether the CPU supports a hardware virtualization extension.
///
/// Detects Intel VT-x (`GenuineIntel` with the VMX feature bit) and AMD-V
/// (`AuthenticAMD` with the SVM feature bit, CPUID `8000_0001h:ECX[2]`).
///
/// # Returns
///
/// Returns `Ok(HypervisorExtension)` indicating the supported extension, or
/// `Err(HypervisorError::VmUnsupported)` if neither is available.
pub fn check_supported_cpu() -> Result<HypervisorExtension, HypervisorError> {
	let cpuid = CpuId::new();

	if let Some(vf) = cpuid.get_vendor_info() {
		match vf.as_str() {
			"GenuineIntel"
				if cpuid
					.get_feature_info()
					.is_some_and(|finfo| finfo.has_vmx()) =>
			{
				return Ok(HypervisorExtension::Vmx);
			}
			"AuthenticAMD"
				if cpuid
					.get_extended_processor_and_feature_identifiers()
					.is_some_and(|finfo| finfo.has_svm()) =>
			{
				return Ok(HypervisorExtension::Svm);
			}
			_ => {}
		}
	}

	Err(HypervisorError::VmUnsupported)
}
