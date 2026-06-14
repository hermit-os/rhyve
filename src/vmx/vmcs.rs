#![allow(dead_code)]

use core::arch::asm;
use core::fmt;

use bitfield::BitMut;
use hermit::arch::{BasePageSize, PageSize};
use x86_64::PrivilegeLevel;
use x86_64::instructions::segmentation::{CS, DS, ES, FS, GS, SS, Segment};
use x86_64::instructions::tables::{sgdt, sidt};
use x86_64::registers::control::{Cr0, Cr0Flags, Cr3, Cr4, Cr4Flags};
use x86_64::registers::model_specific::Msr;
use x86_64::registers::rflags::{self, RFlags};
use x86_64::structures::gdt::SegmentSelector;

use crate::error::HypervisorError;
use crate::vmx::IA32_VMX_BASIC;
use crate::vmx::run::GuestRegisters;
use crate::vmx::vmcs::control::{EntryControls, ExitControls, PrimaryControls, SecondaryControls};

/// IA32_FS_BASE model-specific register.
pub(crate) const IA32_FS_BASE: u32 = 0xC000_0100;
/// IA32_GS_BASE model-specific register.
pub(crate) const IA32_GS_BASE: u32 = 0xC000_0101;
/// IA32_SYSENTER_CS model-specific register.
const IA32_SYSENTER_CS: u32 = 0x174;
/// IA32_SYSENTER_ESP model-specific register.
const IA32_SYSENTER_ESP: u32 = 0x175;
/// IA32_SYSENTER_EIP model-specific register.
const IA32_SYSENTER_EIP: u32 = 0x176;
/// IA32_EFER model-specific register.
pub(crate) const IA32_EFER: u32 = 0xC000_0080;

// VMX capability MSRs reporting allowed settings of the control fields.
const IA32_VMX_PINBASED_CTLS: u32 = 0x481;
const IA32_VMX_PROCBASED_CTLS: u32 = 0x482;
const IA32_VMX_EXIT_CTLS: u32 = 0x483;
const IA32_VMX_ENTRY_CTLS: u32 = 0x484;
const IA32_VMX_PROCBASED_CTLS2: u32 = 0x48B;
const IA32_VMX_TRUE_PINBASED_CTLS: u32 = 0x48D;
const IA32_VMX_TRUE_PROCBASED_CTLS: u32 = 0x48E;
const IA32_VMX_TRUE_EXIT_CTLS: u32 = 0x48F;
const IA32_VMX_TRUE_ENTRY_CTLS: u32 = 0x490;
/// `IA32_VMX_BASIC[55]`: the `IA32_VMX_TRUE_*` capability MSRs are supported.
const IA32_VMX_BASIC_TRUE_CTLS: u64 = 1 << 55;

/// Identifies which VMX control field a capability adjustment applies to.
#[derive(Clone, Copy)]
enum VmxControl {
	PinBased,
	ProcessorBased,
	ProcessorBased2,
	VmExit,
	VmEntry,
}

/// Constrains a requested control value to what the hardware allows.
///
/// Each VMX capability MSR packs the bits that *may be 0* in its low half and
/// the bits that *may be 1* in its high half. Bits that must be 1 are forced on,
/// bits that must be 0 are forced off. See Intel SDM, Appendix A.3.
fn adjust_control(control: VmxControl, requested: u64) -> u64 {
	let true_ctls = unsafe { Msr::new(IA32_VMX_BASIC).read() } & IA32_VMX_BASIC_TRUE_CTLS != 0;

	let cap_msr = match (control, true_ctls) {
		(VmxControl::PinBased, true) => IA32_VMX_TRUE_PINBASED_CTLS,
		(VmxControl::PinBased, false) => IA32_VMX_PINBASED_CTLS,
		(VmxControl::ProcessorBased, true) => IA32_VMX_TRUE_PROCBASED_CTLS,
		(VmxControl::ProcessorBased, false) => IA32_VMX_PROCBASED_CTLS,
		(VmxControl::VmExit, true) => IA32_VMX_TRUE_EXIT_CTLS,
		(VmxControl::VmExit, false) => IA32_VMX_EXIT_CTLS,
		(VmxControl::VmEntry, true) => IA32_VMX_TRUE_ENTRY_CTLS,
		(VmxControl::VmEntry, false) => IA32_VMX_ENTRY_CTLS,
		// There is no "TRUE" variant for the secondary controls.
		(VmxControl::ProcessorBased2, _) => IA32_VMX_PROCBASED_CTLS2,
	};

	let capabilities = unsafe { Msr::new(cap_msr).read() };
	let allowed0 = capabilities as u32; // bits that may be 0
	let allowed1 = (capabilities >> 32) as u32; // bits that may be 1
	let mut effective = requested as u32;
	effective |= allowed0; // force the must-be-1 bits on
	effective &= allowed1; // force the must-be-0 bits off
	u64::from(effective)
}

/// Reads the host task register (TR) selector.
fn read_tr_selector() -> u16 {
	let tr: u16;
	unsafe { asm!("str {0:x}", out(reg) tr, options(nomem, nostack, preserves_flags)) };
	tr
}

/// Resolves the base address of a TSS descriptor in the host GDT.
fn tss_base(gdt_base: u64, tr_selector: u16) -> u64 {
	let index = (tr_selector >> 3) as u64;
	let descriptor = (gdt_base + index * 8) as *const u64;
	// A 64-bit TSS descriptor spans two consecutive 8-byte slots.
	let low = unsafe { descriptor.read() };
	let high = unsafe { descriptor.add(1).read() };
	((low >> 16) & 0xFFFF)
		| (((low >> 32) & 0xFF) << 16)
		| (((low >> 56) & 0xFF) << 24)
		| ((high & 0xFFFF_FFFF) << 32)
}

/// Read a specified field from a VMCS.
#[inline(always)]
fn vmread(field: u32) -> Result<u64, HypervisorError> {
	let field: u64 = field.into();
	let value: u64;
	unsafe { asm!("vmread {0}, {1}", out(reg) value, in(reg) field) };
	let flags = rflags::read();
	if flags.contains(RFlags::ZERO_FLAG) || flags.contains(RFlags::CARRY_FLAG) {
		Err(HypervisorError::VMREADFailed)
	} else {
		Ok(value)
	}
}

/// Write to a specified field in a VMCS.
#[inline(always)]
unsafe fn vmwrite(field: u32, value: u64) -> Result<(), HypervisorError> {
	let field: u64 = field.into();
	unsafe {
		asm!("vmwrite {1}, {0}", in(reg) value, in(reg) field);
	}
	let flags = rflags::read();
	if flags.contains(RFlags::ZERO_FLAG) || flags.contains(RFlags::CARRY_FLAG) {
		Err(HypervisorError::VMWRITEFailed)
	} else {
		Ok(())
	}
}

/// VM-execution, VM-exit, and VM-entry control fields.
pub mod control {
	use bitflags::bitflags;

	// B.1.1.: 16-bit control fields
	/// Virtual-processor identifier (VPID).
	pub const VPID: u32 = 0x0;
	/// Posted-interrupt notification vector.
	pub const POSTED_INTERRUPT_NOTIFICATION_VECTOR: u32 = 0x2;
	/// EPTP index.
	pub const EPTP_INDEX: u32 = 0x4;

	// B.2.1.: 64-bit control fields
	/// Address of I/O bitmap A (full).
	pub const IO_BITMAP_A_ADDR_FULL: u32 = 0x2000;
	/// Address of I/O bitmap A (high).
	pub const IO_BITMAP_A_ADDR_HIGH: u32 = 0x2001;
	/// Address of I/O bitmap B (full).
	pub const IO_BITMAP_B_ADDR_FULL: u32 = 0x2002;
	/// Address of I/O bitmap B (high).
	pub const IO_BITMAP_B_ADDR_HIGH: u32 = 0x2003;
	/// Address of MSR bitmaps (full).
	pub const MSR_BITMAPS_ADDR_FULL: u32 = 0x2004;
	/// Address of MSR bitmaps (high).
	pub const MSR_BITMAPS_ADDR_HIGH: u32 = 0x2005;
	/// VM-exit MSR-store address (full).
	pub const VMEXIT_MSR_STORE_ADDR_FULL: u32 = 0x2006;
	/// VM-exit MSR-store address (high).
	pub const VMEXIT_MSR_STORE_ADDR_HIGH: u32 = 0x2007;
	/// VM-exit MSR-load address (full).
	pub const VMEXIT_MSR_LOAD_ADDR_FULL: u32 = 0x2008;
	/// VM-exit MSR-load address (high).
	pub const VMEXIT_MSR_LOAD_ADDR_HIGH: u32 = 0x2009;
	/// VM-entry MSR-load address (full).
	pub const VMENTRY_MSR_LOAD_ADDR_FULL: u32 = 0x200A;
	/// VM-entry MSR-load address (high).
	pub const VMENTRY_MSR_LOAD_ADDR_HIGH: u32 = 0x200B;
	/// Executive-VMCS pointer (full).
	pub const EXECUTIVE_VMCS_PTR_FULL: u32 = 0x200C;
	/// Executive-VMCS pointer (high).
	pub const EXECUTIVE_VMCS_PTR_HIGH: u32 = 0x200D;
	/// PML address (full).
	pub const PML_ADDR_FULL: u32 = 0x200E;
	/// PML address (high).
	pub const PML_ADDR_HIGH: u32 = 0x200F;
	/// TSC offset (full).
	pub const TSC_OFFSET_FULL: u32 = 0x2010;
	/// TSC offset (high).
	pub const TSC_OFFSET_HIGH: u32 = 0x2011;
	/// Virtual-APIC address (full).
	pub const VIRT_APIC_ADDR_FULL: u32 = 0x2012;
	/// Virtual-APIC address (high).
	pub const VIRT_APIC_ADDR_HIGH: u32 = 0x2013;
	/// APIC-access address (full).
	pub const APIC_ACCESS_ADDR_FULL: u32 = 0x2014;
	/// APIC-access address (high).
	pub const APIC_ACCESS_ADDR_HIGH: u32 = 0x2015;
	/// Posted-interrupt descriptor address (full).
	pub const POSTED_INTERRUPT_DESC_ADDR_FULL: u32 = 0x2016;
	/// Posted-interrupt descriptor address (high).
	pub const POSTED_INTERRUPT_DESC_ADDR_HIGH: u32 = 0x2017;
	/// VM-function controls (full).
	pub const VM_FUNCTION_CONTROLS_FULL: u32 = 0x2018;
	/// VM-function controls (high).
	pub const VM_FUNCTION_CONTROLS_HIGH: u32 = 0x2019;
	/// EPT pointer (full).
	pub const EPTP_FULL: u32 = 0x201A;
	/// EPT pointer (high).
	pub const EPTP_HIGH: u32 = 0x201B;
	/// EOI-exit bitmap 0 (full).
	pub const EOI_EXIT0_FULL: u32 = 0x201C;
	/// EOI-exit bitmap 0 (high).
	pub const EOI_EXIT0_HIGH: u32 = 0x201D;
	/// EOI-exit bitmap 1 (full).
	pub const EOI_EXIT1_FULL: u32 = 0x201E;
	/// EOI-exit bitmap 1 (high).
	pub const EOI_EXIT1_HIGH: u32 = 0x201F;
	/// EOI-exit bitmap 2 (full).
	pub const EOI_EXIT2_FULL: u32 = 0x2020;
	/// EOI-exit bitmap 2 (high).
	pub const EOI_EXIT2_HIGH: u32 = 0x2021;
	/// EOI-exit bitmap 3 (full).
	pub const EOI_EXIT3_FULL: u32 = 0x2022;
	/// EOI-exit bitmap 3 (high).
	pub const EOI_EXIT3_HIGH: u32 = 0x2023;
	/// EPTP-list address (full).
	pub const EPTP_LIST_ADDR_FULL: u32 = 0x2024;
	/// EPTP-list address (high).
	pub const EPTP_LIST_ADDR_HIGH: u32 = 0x2025;
	/// VMREAD-bitmap address (full).
	pub const VMREAD_BITMAP_ADDR_FULL: u32 = 0x2026;
	/// VMREAD-bitmap address (high).
	pub const VMREAD_BITMAP_ADDR_HIGH: u32 = 0x2027;
	/// VMWRITE-bitmap address (full).
	pub const VMWRITE_BITMAP_ADDR_FULL: u32 = 0x2028;
	/// VMWRITE-bitmap address (high).
	pub const VMWRITE_BITMAP_ADDR_HIGH: u32 = 0x2029;
	/// Virtualization-exception information address (full).
	pub const VIRT_EXCEPTION_INFO_ADDR_FULL: u32 = 0x202A;
	/// Virtualization-exception information address (high).
	pub const VIRT_EXCEPTION_INFO_ADDR_HIGH: u32 = 0x202B;
	/// XSS-exiting bitmap (full).
	pub const XSS_EXITING_BITMAP_FULL: u32 = 0x202C;
	/// XSS-exiting bitmap (high).
	pub const XSS_EXITING_BITMAP_HIGH: u32 = 0x202D;
	/// ENCLS-exiting bitmap (full).
	pub const ENCLS_EXITING_BITMAP_FULL: u32 = 0x202E;
	/// ENCLS-exiting bitmap (high).
	pub const ENCLS_EXITING_BITMAP_HIGH: u32 = 0x202F;
	/// Sub-page-permission-table pointer (full).
	pub const SUBPAGE_PERM_TABLE_PTR_FULL: u32 = 0x2030;
	/// Sub-page-permission-table pointer (high).
	pub const SUBPAGE_PERM_TABLE_PTR_HIGH: u32 = 0x2031;
	/// TSC multiplier (full).
	pub const TSC_MULTIPLIER_FULL: u32 = 0x2032;
	/// TSC multiplier (high).
	pub const TSC_MULTIPLIER_HIGH: u32 = 0x2033;

	// B.3.1.: 32-bit control fields
	/// Pin-based VM-execution controls.
	pub const PINBASED_EXEC_CONTROLS: u32 = 0x4000;
	/// Primary processor-based VM-execution controls.
	pub const PRIMARY_PROCBASED_EXEC_CONTROLS: u32 = 0x4002;
	/// Exception bitmap.
	pub const EXCEPTION_BITMAP: u32 = 0x4004;
	/// Page-fault error-code mask.
	pub const PAGE_FAULT_ERR_CODE_MASK: u32 = 0x4006;
	/// Page-fault error-code match.
	pub const PAGE_FAULT_ERR_CODE_MATCH: u32 = 0x4008;
	/// CR3-target count.
	pub const CR3_TARGET_COUNT: u32 = 0x400A;
	/// VM-exit controls.
	pub const VMEXIT_CONTROLS: u32 = 0x400C;
	/// VM-exit MSR-store count.
	pub const VMEXIT_MSR_STORE_COUNT: u32 = 0x400E;
	/// VM-exit MSR-load count.
	pub const VMEXIT_MSR_LOAD_COUNT: u32 = 0x4010;
	/// VM-entry controls.
	pub const VMENTRY_CONTROLS: u32 = 0x4012;
	/// VM-entry MSR-load count.
	pub const VMENTRY_MSR_LOAD_COUNT: u32 = 0x4014;
	/// VM-entry interruption-information field.
	pub const VMENTRY_INTERRUPTION_INFO_FIELD: u32 = 0x4016;
	/// VM-entry exception error code.
	pub const VMENTRY_EXCEPTION_ERR_CODE: u32 = 0x4018;
	/// VM-entry instruction length.
	pub const VMENTRY_INSTRUCTION_LEN: u32 = 0x401A;
	/// TPR threshold.
	pub const TPR_THRESHOLD: u32 = 0x401C;
	/// Secondary processor-based VM-execution controls.
	pub const SECONDARY_PROCBASED_EXEC_CONTROLS: u32 = 0x401E;
	/// PLE_Gap.
	pub const PLE_GAP: u32 = 0x4020;
	/// PLE_Window.
	pub const PLE_WINDOW: u32 = 0x4022;

	// B.4.1.: natural-width control fields
	/// CR0 guest/host mask.
	pub const CR0_GUEST_HOST_MASK: u32 = 0x6000;
	/// CR4 guest/host mask.
	pub const CR4_GUEST_HOST_MASK: u32 = 0x6002;
	/// CR0 read shadow.
	pub const CR0_READ_SHADOW: u32 = 0x6004;
	/// CR4 read shadow.
	pub const CR4_READ_SHADOW: u32 = 0x6006;
	/// CR3-target value 0.
	pub const CR3_TARGET_VALUE0: u32 = 0x6008;
	/// CR3-target value 1.
	pub const CR3_TARGET_VALUE1: u32 = 0x600A;
	/// CR3-target value 2.
	pub const CR3_TARGET_VALUE2: u32 = 0x600C;
	/// CR3-target value 3.
	pub const CR3_TARGET_VALUE3: u32 = 0x600E;

	bitflags! {
		/// Pin-based VM-execution controls.
		///
		/// A set of bitmask flags useful when setting up [`PINBASED_EXEC_CONTROLS`] VMCS field.
		///
		/// See Intel SDM, Volume 3C, Section 24.6.1.
		pub struct PinbasedControls: u32 {
			/// External-interrupt exiting.
			const EXTERNAL_INTERRUPT_EXITING = 1 << 0;
			/// NMI exiting.
			const NMI_EXITING = 1 << 3;
			/// Virtual NMIs.
			const VIRTUAL_NMIS = 1 << 5;
			/// Activate VMX-preemption timer.
			const VMX_PREEMPTION_TIMER = 1 << 6;
			/// Process posted interrupts.
			const POSTED_INTERRUPTS = 1 << 7;
		}
	}

	bitflags! {
		/// Primary processor-based VM-execution controls.
		///
		/// A set of bitmask flags useful when setting up [`PRIMARY_PROCBASED_EXEC_CONTROLS`] VMCS field.
		///
		/// See Intel SDM, Volume 3C, Section 24.6.2, Table 24-6.
		pub struct PrimaryControls: u32 {
			/// Interrupt-window exiting.
			const INTERRUPT_WINDOW_EXITING = 1 << 2;
			/// Use TSC offsetting.
			const USE_TSC_OFFSETTING = 1 << 3;
			/// HLT exiting.
			const HLT_EXITING = 1 << 7;
			/// INVLPG exiting.
			const INVLPG_EXITING = 1 << 9;
			/// MWAIT exiting.
			const MWAIT_EXITING = 1 << 10;
			/// RDPMC exiting.
			const RDPMC_EXITING = 1 << 11;
			/// RDTSC exiting.
			const RDTSC_EXITING = 1 << 12;
			/// CR3-load exiting.
			const CR3_LOAD_EXITING = 1 << 15;
			/// CR3-store exiting.
			const CR3_STORE_EXITING = 1 << 16;
			/// CR8-load exiting.
			const CR8_LOAD_EXITING = 1 << 19;
			/// CR8-store exiting.
			const CR8_STORE_EXITING = 1 << 20;
			/// Use TPR shadow.
			const USE_TPR_SHADOW = 1 << 21;
			/// NMI-window exiting.
			const NMI_WINDOW_EXITING = 1 << 22;
			/// MOV-DR exiting
			const MOV_DR_EXITING = 1 << 23;
			/// Unconditional I/O exiting.
			const UNCOND_IO_EXITING = 1 << 24;
			/// Use I/O bitmaps.
			const USE_IO_BITMAPS = 1 << 25;
			/// Monitor trap flag.
			const MONITOR_TRAP_FLAG = 1 << 27;
			/// Use MSR bitmaps.
			const USE_MSR_BITMAPS = 1 << 28;
			/// MONITOR exiting.
			const MONITOR_EXITING = 1 << 29;
			/// PAUSE exiting.
			const PAUSE_EXITING = 1 << 30;
			/// Activate secondary controls.
			const SECONDARY_CONTROLS = 1 << 31;
		}
	}

	bitflags! {
		/// Secondary processor-based VM-execution controls.
		///
		/// A set of bitmask flags useful when setting up [`SECONDARY_PROCBASED_EXEC_CONTROLS`] VMCS field.
		///
		/// See Intel SDM, Volume 3C, Section 24.6.2, Table 24-7.
		pub struct SecondaryControls: u32 {
			/// Virtualize APIC accesses.
			const VIRTUALIZE_APIC = 1 << 0;
			/// Enable EPT.
			const ENABLE_EPT = 1 << 1;
			/// Descriptor-table exiting.
			const DTABLE_EXITING = 1 << 2;
			/// Enable RDTSCP.
			const ENABLE_RDTSCP = 1 << 3;
			/// Virtualize x2APIC mode.
			const VIRTUALIZE_X2APIC = 1 << 4;
			/// Enable VPID.
			const ENABLE_VPID = 1 << 5;
			/// WBINVD exiting.
			const WBINVD_EXITING = 1 << 6;
			/// Unrestricted guest.
			const UNRESTRICTED_GUEST = 1 << 7;
			/// APIC-register virtualization.
			const VIRTUALIZE_APIC_REGISTER = 1 << 8;
			/// Virtual-interrupt delivery.
			const VIRTUAL_INTERRUPT_DELIVERY = 1 << 9;
			/// PAUSE-loop exiting.
			const PAUSE_LOOP_EXITING = 1 << 10;
			/// RDRAND exiting.
			const RDRAND_EXITING = 1 << 11;
			/// Enable INVPCID.
			const ENABLE_INVPCID = 1 << 12;
			/// Enable VM functions.
			const ENABLE_VM_FUNCTIONS = 1 << 13;
			/// VMCS shadowing.
			const VMCS_SHADOWING = 1 << 14;
			/// Enable ENCLS exiting.
			const ENCLS_EXITING = 1 << 15;
			/// RDSEED exiting.
			const RDSEED_EXITING = 1 << 16;
			/// Enable PML.
			const ENABLE_PML = 1 << 17;
			/// EPT-violation #VE.
			const EPT_VIOLATION_VE = 1 << 18;
			/// Conceal VMX from PT.
			const CONCEAL_VMX_FROM_PT = 1 << 19;
			/// Enable XSAVES/XRSTORS.
			const ENABLE_XSAVES_XRSTORS = 1 << 20;
			/// Mode-based execute control for EPT.
			const MODE_BASED_EPT = 1 << 22;
			/// Sub-page write permissions for EPT.
			const SUB_PAGE_EPT = 1 << 23;
			/// Intel PT uses guest physical addresses.
			const INTEL_PT_GUEST_PHYSICAL = 1 << 24;
			/// Use TSC scaling.
			const USE_TSC_SCALING = 1 << 25;
			/// Enable user wait and pause.
			const ENABLE_USER_WAIT_PAUSE = 1 << 26;
			/// Enable ENCLV exiting.
			const ENCLV_EXITING = 1 << 28;
		}
	}

	bitflags! {
		/// VM-entry controls.
		///
		/// A set of bitmask flags useful when setting up [`VMENTRY_CONTROLS`] VMCS field.
		///
		/// See Intel SDM, Volume 3C, Section 24.8.
		pub struct EntryControls: u32 {
			/// Load debug controls.
			const LOAD_DEBUG_CONTROLS = 1 << 2;
			/// IA-32e mode guest.
			const IA32E_MODE_GUEST = 1 << 9;
			/// Entry to SMM.
			const ENTRY_TO_SMM = 1 << 10;
			/// Deactivate dual-monitor treatment.
			const DEACTIVATE_DUAL_MONITOR = 1 << 11;
			/// Load IA32_PERF_GLOBAL_CTRL.
			const LOAD_IA32_PERF_GLOBAL_CTRL = 1 << 13;
			/// Load IA32_PAT.
			const LOAD_IA32_PAT = 1 << 14;
			/// Load IA32_EFER.
			const LOAD_IA32_EFER = 1 << 15;
			/// Load IA32_BNDCFGS.
			const LOAD_IA32_BNDCFGS = 1 << 16;
			/// Conceal VMX from PT.
			const CONCEAL_VMX_FROM_PT = 1 << 17;
			/// Load IA32_RTIT_CTL.
			const LOAD_IA32_RTIT_CTL = 1 << 18;
		}
	}

	bitflags! {
		/// VM-exit controls.
		///
		/// A set of bitmask flags useful when setting up [`VMEXIT_CONTROLS`] VMCS field.
		///
		/// See Intel SDM, Volume 3C, Section 24.7.
		pub struct ExitControls: u32 {
			/// Save debug controls.
			const SAVE_DEBUG_CONTROLS = 1 << 2;
			/// Host address-space size.
			const HOST_ADDRESS_SPACE_SIZE = 1 << 9;
			/// Load IA32_PERF_GLOBAL_CTRL.
			const LOAD_IA32_PERF_GLOBAL_CTRL = 1 << 12;
			/// Acknowledge interrupt on exit.
			const ACK_INTERRUPT_ON_EXIT = 1 << 15;
			/// Save IA32_PAT.
			const SAVE_IA32_PAT = 1 << 18;
			/// Load IA32_PAT.
			const LOAD_IA32_PAT = 1 << 19;
			/// Save IA32_EFER.
			const SAVE_IA32_EFER = 1 << 20;
			/// Load IA32_EFER.
			const LOAD_IA32_EFER = 1 << 21;
			/// Save VMX-preemption timer.
			const SAVE_VMX_PREEMPTION_TIMER = 1 << 22;
			/// Clear IA32_BNDCFGS.
			const CLEAR_IA32_BNDCFGS = 1 << 23;
			/// Conceal VMX from PT.
			const CONCEAL_VMX_FROM_PT = 1 << 24;
			/// Clear IA32_RTIT_CTL.
			const CLEAR_IA32_RTIT_CTL = 1 << 25;
		}
	}
}

/// Fields used to access guest-state area.
pub mod guest {
	// B.1.2.: 16-bit guest-state fields
	/// Guest ES selector.
	pub const ES_SELECTOR: u32 = 0x800;
	/// Guest CS selector.
	pub const CS_SELECTOR: u32 = 0x802;
	/// Guest SS selector.
	pub const SS_SELECTOR: u32 = 0x804;
	/// Guest DS selector.
	pub const DS_SELECTOR: u32 = 0x806;
	/// Guest FS selector.
	pub const FS_SELECTOR: u32 = 0x808;
	/// Guest GS selector.
	pub const GS_SELECTOR: u32 = 0x80A;
	/// Guest LDTR selector.
	pub const LDTR_SELECTOR: u32 = 0x80C;
	/// Guest TR selector.
	pub const TR_SELECTOR: u32 = 0x80E;
	/// Guest interrupt status.
	pub const INTERRUPT_STATUS: u32 = 0x810;
	/// PML index.
	pub const PML_INDEX: u32 = 0x812;

	// B.2.3.: 64-bit guest-state fields
	/// VMCS link pointer (full).
	pub const LINK_PTR_FULL: u32 = 0x2800;
	/// VMCS link pointer (high).
	pub const LINK_PTR_HIGH: u32 = 0x2801;
	/// Guest IA32_DEBUGCTL (full).
	pub const IA32_DEBUGCTL_FULL: u32 = 0x2802;
	/// Guest IA32_DEBUGCTL (high).
	pub const IA32_DEBUGCTL_HIGH: u32 = 0x2803;
	/// Guest IA32_PAT (full).
	pub const IA32_PAT_FULL: u32 = 0x2804;
	/// Guest IA32_PAT (high).
	pub const IA32_PAT_HIGH: u32 = 0x2805;
	/// Guest IA32_EFER (full).
	pub const IA32_EFER_FULL: u32 = 0x2806;
	/// Guest IA32_EFER (high).
	pub const IA32_EFER_HIGH: u32 = 0x2807;
	/// Guest IA32_PERF_GLOBAL_CTRL (full).
	pub const IA32_PERF_GLOBAL_CTRL_FULL: u32 = 0x2808;
	/// Guest IA32_PERF_GLOBAL_CTRL (high).
	pub const IA32_PERF_GLOBAL_CTRL_HIGH: u32 = 0x2809;
	/// Guest PDPTE0 (full).
	pub const PDPTE0_FULL: u32 = 0x280A;
	/// Guest PDPTE0 (high).
	pub const PDPTE0_HIGH: u32 = 0x280B;
	/// Guest PDPTE1 (full).
	pub const PDPTE1_FULL: u32 = 0x280C;
	/// Guest PDPTE1 (high).
	pub const PDPTE1_HIGH: u32 = 0x280D;
	/// Guest PDPTE2 (full).
	pub const PDPTE2_FULL: u32 = 0x280E;
	/// Guest PDPTE2 (high).
	pub const PDPTE2_HIGH: u32 = 0x280F;
	/// Guest PDPTE3 (full).
	pub const PDPTE3_FULL: u32 = 0x2810;
	/// Guest PDPTE3 (high).
	pub const PDPTE3_HIGH: u32 = 0x2811;
	/// Guest IA32_BNDCFGS (full).
	pub const IA32_BNDCFGS_FULL: u32 = 0x2812;
	/// Guest IA32_BNDCFGS (high).
	pub const IA32_BNDCFGS_HIGH: u32 = 0x2813;
	/// Guest IA32_RTIT_CTL (full).
	pub const IA32_RTIT_CTL_FULL: u32 = 0x2814;
	/// Guest IA32_RTIT_CTL (high).
	pub const IA32_RTIT_CTL_HIGH: u32 = 0x2815;

	// B.3.3.: 32-bit guest-state fields
	/// Guest ES limit.
	pub const ES_LIMIT: u32 = 0x4800;
	/// Guest CS limit.
	pub const CS_LIMIT: u32 = 0x4802;
	/// Guest SS limit.
	pub const SS_LIMIT: u32 = 0x4804;
	/// Guest DS limit.
	pub const DS_LIMIT: u32 = 0x4806;
	/// Guest FS limit.
	pub const FS_LIMIT: u32 = 0x4808;
	/// Guest GS limit.
	pub const GS_LIMIT: u32 = 0x480A;
	/// Guest LDTR limit.
	pub const LDTR_LIMIT: u32 = 0x480C;
	/// Guest TR limit.
	pub const TR_LIMIT: u32 = 0x480E;
	/// Guest GDTR limit.
	pub const GDTR_LIMIT: u32 = 0x4810;
	/// Guest IDTR limit.
	pub const IDTR_LIMIT: u32 = 0x4812;
	/// Guest ES access rights.
	pub const ES_ACCESS_RIGHTS: u32 = 0x4814;
	/// Guest CS access rights.
	pub const CS_ACCESS_RIGHTS: u32 = 0x4816;
	/// Guest SS access rights.
	pub const SS_ACCESS_RIGHTS: u32 = 0x4818;
	/// Guest DS access rights.
	pub const DS_ACCESS_RIGHTS: u32 = 0x481A;
	/// Guest FS access rights.
	pub const FS_ACCESS_RIGHTS: u32 = 0x481C;
	/// Guest GS access rights.
	pub const GS_ACCESS_RIGHTS: u32 = 0x481E;
	/// Guest LDTR access rights.
	pub const LDTR_ACCESS_RIGHTS: u32 = 0x4820;
	/// Guest TR access rights.
	pub const TR_ACCESS_RIGHTS: u32 = 0x4822;
	/// Guest interruptibility state.
	pub const INTERRUPTIBILITY_STATE: u32 = 0x4824;
	/// Guest activity state.
	pub const ACTIVITY_STATE: u32 = 0x4826;
	/// Guest SMBASE.
	pub const SMBASE: u32 = 0x4828;
	/// Guest IA32_SYSENTER_CS.
	pub const IA32_SYSENTER_CS: u32 = 0x482A;
	/// VMX-preemption timer value.
	pub const VMX_PREEMPTION_TIMER_VALUE: u32 = 0x482E;

	// B.4.3.: natural-width guest-state fields
	/// Guest CR0.
	pub const CR0: u32 = 0x6800;
	/// Guest CR3.
	pub const CR3: u32 = 0x6802;
	/// Guest CR4.
	pub const CR4: u32 = 0x6804;
	/// Guest ES base.
	pub const ES_BASE: u32 = 0x6806;
	/// Guest CS base.
	pub const CS_BASE: u32 = 0x6808;
	/// Guest SS base.
	pub const SS_BASE: u32 = 0x680A;
	/// Guest DS base.
	pub const DS_BASE: u32 = 0x680C;
	/// Guest FS base.
	pub const FS_BASE: u32 = 0x680E;
	/// Guest GS base.
	pub const GS_BASE: u32 = 0x6810;
	/// Guest LDTR base.
	pub const LDTR_BASE: u32 = 0x6812;
	/// Guest TR base.
	pub const TR_BASE: u32 = 0x6814;
	/// Guest GDTR base.
	pub const GDTR_BASE: u32 = 0x6816;
	/// Guest IDTR base.
	pub const IDTR_BASE: u32 = 0x6818;
	/// Guest DR7.
	pub const DR7: u32 = 0x681A;
	/// Guest RSP.
	pub const RSP: u32 = 0x681C;
	/// Guest RIP.
	pub const RIP: u32 = 0x681E;
	/// Guest RFLAGS.
	pub const RFLAGS: u32 = 0x6820;
	/// Guest pending debug exceptions.
	pub const PENDING_DBG_EXCEPTIONS: u32 = 0x6822;
	/// Guest IA32_SYSENTER_ESP.
	pub const IA32_SYSENTER_ESP: u32 = 0x6824;
	/// Guest IA32_SYSENTER_EIP.
	pub const IA32_SYSENTER_EIP: u32 = 0x6826;
}

/// Fields used to access host-state area.
pub mod host {
	// B.1.3.: 16-bit host-state fields
	/// Host ES selector.
	pub const ES_SELECTOR: u32 = 0xC00;
	/// Host CS selector.
	pub const CS_SELECTOR: u32 = 0xC02;
	/// Host SS selector.
	pub const SS_SELECTOR: u32 = 0xC04;
	/// Host DS selector.
	pub const DS_SELECTOR: u32 = 0xC06;
	/// Host FS selector.
	pub const FS_SELECTOR: u32 = 0xC08;
	/// Host GS selector.
	pub const GS_SELECTOR: u32 = 0xC0A;
	/// Host TR selector.
	pub const TR_SELECTOR: u32 = 0xC0C;

	// B.2.4.: 64-bit host-state fields
	/// Host IA32_PAT (full).
	pub const IA32_PAT_FULL: u32 = 0x2C00;
	/// Host IA32_PAT (high).
	pub const IA32_PAT_HIGH: u32 = 0x2C01;
	/// Host IA32_EFER (full).
	pub const IA32_EFER_FULL: u32 = 0x2C02;
	/// Host IA32_EFER (high).
	pub const IA32_EFER_HIGH: u32 = 0x2C03;
	/// Host IA32_PERF_GLOBAL_CTRL (full).
	pub const IA32_PERF_GLOBAL_CTRL_FULL: u32 = 0x2C04;
	/// Host IA32_PERF_GLOBAL_CTRL (high).
	pub const IA32_PERF_GLOBAL_CTRL_HIGH: u32 = 0x2C05;

	// B.3.4.: 32-bit host-state field
	/// Host IA32_SYSENTER_CS.
	pub const IA32_SYSENTER_CS: u32 = 0x4C00;

	// B.4.4.: natural-width host-state fields
	/// Host CR0.
	pub const CR0: u32 = 0x6C00;
	/// Host CR3.
	pub const CR3: u32 = 0x6C02;
	/// Host CR4.
	pub const CR4: u32 = 0x6C04;
	/// Host FS base.
	pub const FS_BASE: u32 = 0x6C06;
	/// Host GS base.
	pub const GS_BASE: u32 = 0x6C08;
	/// Host TR base.
	pub const TR_BASE: u32 = 0x6C0A;
	/// Host GDTR base.
	pub const GDTR_BASE: u32 = 0x6C0C;
	/// Host IDTR base.
	pub const IDTR_BASE: u32 = 0x6C0E;
	/// Host IA32_SYSENTER_ESP.
	pub const IA32_SYSENTER_ESP: u32 = 0x6C10;
	/// Host IA32_SYSENTER_EIP.
	pub const IA32_SYSENTER_EIP: u32 = 0x6C12;
	/// Host RSP.
	pub const RSP: u32 = 0x6C14;
	/// Host RIP.
	pub const RIP: u32 = 0x6C16;
}

/// VM-exit information fields.
pub mod ro {
	// B.2.2.: 64-bit read-only data fields
	/// Guest-physical address (full).
	pub const GUEST_PHYSICAL_ADDR_FULL: u32 = 0x2400;
	/// Guest-physical address (high).
	pub const GUEST_PHYSICAL_ADDR_HIGH: u32 = 0x2401;

	// B.3.2.: 32-bit read-only data fields
	/// VM-instruction error.
	pub const VM_INSTRUCTION_ERROR: u32 = 0x4400;
	/// Exit reason.
	pub const EXIT_REASON: u32 = 0x4402;
	/// VM-exit interruption information.
	pub const VMEXIT_INTERRUPTION_INFO: u32 = 0x4404;
	/// VM-exit interruption error code.
	pub const VMEXIT_INTERRUPTION_ERR_CODE: u32 = 0x4406;
	/// IDT-vectoring information field.
	pub const IDT_VECTORING_INFO: u32 = 0x4408;
	/// IDT-vectoring error code.
	pub const IDT_VECTORING_ERR_CODE: u32 = 0x440A;
	/// VM-exit instruction length.
	pub const VMEXIT_INSTRUCTION_LEN: u32 = 0x440C;
	/// VM-exit instruction information.
	pub const VMEXIT_INSTRUCTION_INFO: u32 = 0x440E;

	// B.4.2.: natural-width read-only data fields
	/// Exit qualification.
	pub const EXIT_QUALIFICATION: u32 = 0x6400;
	/// I/O RCX.
	pub const IO_RCX: u32 = 0x6402;
	/// I/O RSI.
	pub const IO_RSI: u32 = 0x6404;
	/// I/O RDI.
	pub const IO_RDI: u32 = 0x6406;
	/// I/O RIP.
	pub const IO_RIP: u32 = 0x6408;
	/// Guest-linear address.
	pub const GUEST_LINEAR_ADDR: u32 = 0x640A;
}

/// Represents the VMCS region in memory.
///
/// The VMCS region is essential for VMX operations on the CPU.
/// This structure offers methods for setting up the VMCS region, adjusting VMCS entries,
/// and performing related tasks.
///
/// Reference: Intel® 64 and IA-32 Architectures Software Developer's Manual: 25.2 FORMAT OF THE VMCS REGION
#[repr(C, align(4096))]
pub struct Vmcs {
	pub revision_id: u32,
	pub abort_indicator: u32,
	pub reserved: [u8; BasePageSize::SIZE as usize - 8],
}

impl Vmcs {
	/// Initializes the VMCS region.
	pub fn init(&mut self) {
		let msr = Msr::new(IA32_VMX_BASIC);
		self.revision_id = unsafe { msr.read() as u32 };
		self.revision_id.set_bit(31, false);
	}

	/// Configures the VM-execution, VM-entry and VM-exit control fields.
	///
	/// Enables EPT (using `eptp`) and configures the guest to run as a 64-bit
	/// (IA-32e mode) guest while the host returns to 64-bit mode on VM-exit.
	/// Every value is constrained by the VMX capability MSRs via
	/// [`adjust_control`].
	pub fn setup_controls(&mut self, eptp: u64) -> Result<(), HypervisorError> {
		// No pin-based controls are required for a minimal guest.
		let pinbased = adjust_control(VmxControl::PinBased, 0);
		// Activate the secondary controls so EPT can be enabled, and make all I/O
		// instructions cause a VM-exit so the guest's port I/O can be emulated.
		let procbased = adjust_control(
			VmxControl::ProcessorBased,
			(PrimaryControls::SECONDARY_CONTROLS | PrimaryControls::UNCOND_IO_EXITING).bits()
				as u64,
		);
		// Besides EPT, enable the instructions the guest expects to be usable
		// because CPUID advertises them: without their secondary-control enable
		// bit, RDTSCP/INVPCID/XSAVES raise #UD in the guest.
		let secondary = adjust_control(
			VmxControl::ProcessorBased2,
			(SecondaryControls::ENABLE_EPT
				| SecondaryControls::ENABLE_RDTSCP
				| SecondaryControls::ENABLE_INVPCID
				| SecondaryControls::ENABLE_XSAVES_XRSTORS)
				.bits() as u64,
		);
		// The guest runs in long mode; the host resumes in long mode. The
		// "load/save debug controls" bits are part of the default1 set; even
		// though the TRUE capability MSRs allow clearing them, KVM rejects
		// VM-entry/exit controls with them cleared, so request them explicitly.
		let entry = adjust_control(
			VmxControl::VmEntry,
			(EntryControls::IA32E_MODE_GUEST | EntryControls::LOAD_DEBUG_CONTROLS).bits() as u64,
		);
		let exit = adjust_control(
			VmxControl::VmExit,
			(ExitControls::HOST_ADDRESS_SPACE_SIZE | ExitControls::SAVE_DEBUG_CONTROLS).bits()
				as u64,
		);

		debug!(
			"VMX controls: pin={pinbased:#x} proc={procbased:#x} secondary={secondary:#x} entry={entry:#x} exit={exit:#x} eptp={eptp:#x}"
		);

		unsafe {
			vmwrite(control::PINBASED_EXEC_CONTROLS, pinbased)?;
			vmwrite(control::PRIMARY_PROCBASED_EXEC_CONTROLS, procbased)?;
			vmwrite(control::SECONDARY_PROCBASED_EXEC_CONTROLS, secondary)?;
			vmwrite(control::VMENTRY_CONTROLS, entry)?;
			vmwrite(control::VMEXIT_CONTROLS, exit)?;
			vmwrite(control::EPTP_FULL, eptp)?;

			// Explicitly clear the count, threshold and bitmap-related control
			// fields. After VMCLEAR these are not guaranteed to be zero, and a
			// stray non-zero MSR-load/store or CR3-target count makes VM-entry
			// fail with "invalid control field(s)".
			vmwrite(control::VMENTRY_MSR_LOAD_COUNT, 0)?;
			vmwrite(control::VMEXIT_MSR_LOAD_COUNT, 0)?;
			vmwrite(control::VMEXIT_MSR_STORE_COUNT, 0)?;
			vmwrite(control::VMENTRY_INTERRUPTION_INFO_FIELD, 0)?;
			vmwrite(control::CR3_TARGET_COUNT, 0)?;
			vmwrite(control::EXCEPTION_BITMAP, 0)?;
			vmwrite(control::PAGE_FAULT_ERR_CODE_MASK, 0)?;
			vmwrite(control::PAGE_FAULT_ERR_CODE_MATCH, 0)?;
			vmwrite(control::TPR_THRESHOLD, 0)?;
		}

		Ok(())
	}

	/// Configures the host-state area from the current (hermit) host state.
	///
	/// On VM-exit the processor reloads these registers, so they must describe
	/// the environment `rhyve` continues running in. Host RSP and RIP are not
	/// set here; the [`run_vmx_vm`](crate::vmx::run::run_vmx_vm) trampoline
	/// writes them right before `VMLAUNCH`.
	pub fn setup_host(&mut self) -> Result<(), HypervisorError> {
		let gdt = sgdt();
		let idt = sidt();
		let tr_selector = read_tr_selector();
		let tr_base = tss_base(gdt.base.as_u64(), tr_selector);

		// RPL/TI bits must be zero for host segment selectors.
		const SELECTOR_MASK: u16 = 0xFFF8;

		unsafe {
			vmwrite(host::CR0, Cr0::read_raw())?;
			vmwrite(host::CR3, Cr3::read().0.start_address().as_u64())?;
			vmwrite(host::CR4, Cr4::read_raw())?;

			vmwrite(host::CS_SELECTOR, (CS::get_reg().0 & SELECTOR_MASK) as u64)?;
			vmwrite(host::SS_SELECTOR, (SS::get_reg().0 & SELECTOR_MASK) as u64)?;
			vmwrite(host::DS_SELECTOR, (DS::get_reg().0 & SELECTOR_MASK) as u64)?;
			vmwrite(host::ES_SELECTOR, (ES::get_reg().0 & SELECTOR_MASK) as u64)?;
			vmwrite(host::FS_SELECTOR, (FS::get_reg().0 & SELECTOR_MASK) as u64)?;
			vmwrite(host::GS_SELECTOR, (GS::get_reg().0 & SELECTOR_MASK) as u64)?;
			vmwrite(host::TR_SELECTOR, (tr_selector & SELECTOR_MASK) as u64)?;

			vmwrite(host::FS_BASE, Msr::new(IA32_FS_BASE).read())?;
			vmwrite(host::GS_BASE, Msr::new(IA32_GS_BASE).read())?;
			vmwrite(host::TR_BASE, tr_base)?;
			vmwrite(host::GDTR_BASE, gdt.base.as_u64())?;
			vmwrite(host::IDTR_BASE, idt.base.as_u64())?;

			vmwrite(host::IA32_SYSENTER_CS, Msr::new(IA32_SYSENTER_CS).read())?;
			vmwrite(host::IA32_SYSENTER_ESP, Msr::new(IA32_SYSENTER_ESP).read())?;
			vmwrite(host::IA32_SYSENTER_EIP, Msr::new(IA32_SYSENTER_EIP).read())?;
		}

		Ok(())
	}

	/// Configures the guest-state area for a freshly booted 64-bit kernel.
	///
	/// Segment selectors refer to the boot GDT that `rhyve` writes into guest
	/// memory; all segment bases are flat (0) with a 1 MiB byte-granular limit
	/// (segment limits are ignored in 64-bit mode anyway). `entry_point` and
	/// `rsp` become the guest's initial RIP and RSP.
	pub fn setup_guest(&mut self, entry_point: u64, rsp: u64) -> Result<(), HypervisorError> {
		let code = SegmentSelector::new(crate::GDT_KERNEL_CODE, PrivilegeLevel::Ring0).0 as u64;
		let data = SegmentSelector::new(crate::GDT_KERNEL_DATA, PrivilegeLevel::Ring0).0 as u64;

		// Access rights: present 64-bit code (0xA09B), present data (0xC093),
		// busy 64-bit TSS (0x008B) and an unusable LDTR (bit 16 set).
		const CS_AR: u64 = 0xA09B;
		const DATA_AR: u64 = 0xC093;
		const TR_AR: u64 = 0x008B;
		const LDTR_AR: u64 = 1 << 16;
		// Flat segment: byte-granular 4 GiB limit expressed with G = 1.
		const SEG_LIMIT: u64 = 0xFFFF_FFFF;

		let cr0 = Cr0Flags::PROTECTED_MODE_ENABLE
			| Cr0Flags::EXTENSION_TYPE
			| Cr0Flags::NUMERIC_ERROR
			| Cr0Flags::PAGING;
		let cr4 = Cr4Flags::PHYSICAL_ADDRESS_EXTENSION | Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS;

		unsafe {
			// Selectors.
			vmwrite(guest::CS_SELECTOR, code)?;
			vmwrite(guest::SS_SELECTOR, data)?;
			vmwrite(guest::DS_SELECTOR, data)?;
			vmwrite(guest::ES_SELECTOR, data)?;
			vmwrite(guest::FS_SELECTOR, data)?;
			vmwrite(guest::GS_SELECTOR, data)?;
			vmwrite(guest::TR_SELECTOR, 0)?;
			vmwrite(guest::LDTR_SELECTOR, 0)?;

			// Access rights.
			vmwrite(guest::CS_ACCESS_RIGHTS, CS_AR)?;
			vmwrite(guest::SS_ACCESS_RIGHTS, DATA_AR)?;
			vmwrite(guest::DS_ACCESS_RIGHTS, DATA_AR)?;
			vmwrite(guest::ES_ACCESS_RIGHTS, DATA_AR)?;
			vmwrite(guest::FS_ACCESS_RIGHTS, DATA_AR)?;
			vmwrite(guest::GS_ACCESS_RIGHTS, DATA_AR)?;
			vmwrite(guest::TR_ACCESS_RIGHTS, TR_AR)?;
			vmwrite(guest::LDTR_ACCESS_RIGHTS, LDTR_AR)?;

			// Bases (flat) and limits.
			for base in [
				guest::CS_BASE,
				guest::SS_BASE,
				guest::DS_BASE,
				guest::ES_BASE,
				guest::FS_BASE,
				guest::GS_BASE,
				guest::LDTR_BASE,
				guest::TR_BASE,
			] {
				vmwrite(base, 0)?;
			}
			// Code/data segments use page granularity (G = 1), so a 4 GiB limit
			// is valid. TR/LDTR are byte-granular (G = 0); their limit must keep
			// bits 31:20 clear, hence a 16-bit limit.
			for limit in [
				guest::CS_LIMIT,
				guest::SS_LIMIT,
				guest::DS_LIMIT,
				guest::ES_LIMIT,
				guest::FS_LIMIT,
				guest::GS_LIMIT,
			] {
				vmwrite(limit, SEG_LIMIT)?;
			}
			vmwrite(guest::TR_LIMIT, 0xFFFF)?;
			vmwrite(guest::LDTR_LIMIT, 0xFFFF)?;

			// Descriptor tables: the boot GDT lives in guest memory, the guest
			// starts without an IDT.
			vmwrite(guest::GDTR_BASE, crate::GDT_OFFSET)?;
			vmwrite(
				guest::GDTR_LIMIT,
				((core::mem::size_of::<u64>() * crate::BOOT_GDT_MAX) - 1) as u64,
			)?;
			vmwrite(guest::IDTR_BASE, 0)?;
			vmwrite(guest::IDTR_LIMIT, 0xFFFF)?;

			// Control registers and EFER (long mode enabled and active).
			vmwrite(guest::CR0, cr0.bits())?;
			vmwrite(guest::CR3, crate::BOOT_PML4)?;
			vmwrite(guest::CR4, cr4.bits())?;
			vmwrite(guest::IA32_EFER_FULL, crate::EFER_LME | crate::EFER_LMA)?;
			vmwrite(
				control::CR0_GUEST_HOST_MASK,
				(Cr0Flags::CACHE_DISABLE | Cr0Flags::NOT_WRITE_THROUGH | cr0).bits(),
			)?;
			vmwrite(control::CR0_READ_SHADOW, cr0.bits())?;
			vmwrite(control::CR4_GUEST_HOST_MASK, cr4.bits())?;
			vmwrite(control::CR4_READ_SHADOW, cr4.bits())?;

			// Initial RIP/RSP/RFLAGS and remaining guest state.
			vmwrite(guest::RIP, entry_point)?;
			vmwrite(guest::RSP, rsp)?;
			vmwrite(guest::RFLAGS, 0x2)?; // only the reserved bit 1 is set
			vmwrite(guest::DR7, 0x400)?;
			vmwrite(guest::IA32_DEBUGCTL_FULL, 0)?;
			vmwrite(guest::IA32_SYSENTER_CS, 0)?;
			vmwrite(guest::IA32_SYSENTER_ESP, 0)?;
			vmwrite(guest::IA32_SYSENTER_EIP, 0)?;
			vmwrite(guest::ACTIVITY_STATE, 0)?;
			vmwrite(guest::INTERRUPTIBILITY_STATE, 0)?;
			vmwrite(guest::PENDING_DBG_EXCEPTIONS, 0)?;
			vmwrite(guest::LINK_PTR_FULL, u64::MAX)?;
		}

		Ok(())
	}

	/// Writes the cached guest RIP/RSP/RFLAGS into the VMCS before a VM-entry.
	pub fn load_guest_registers(&self, regs: &GuestRegisters) -> Result<(), HypervisorError> {
		unsafe {
			vmwrite(guest::RIP, regs.rip)?;
			vmwrite(guest::RSP, regs.rsp)?;
			vmwrite(guest::RFLAGS, regs.rflags)?;
		}
		Ok(())
	}

	/// Reads RIP/RSP/RFLAGS from the VMCS into the cached guest registers after a VM-exit.
	pub fn save_guest_registers(&self, regs: &mut GuestRegisters) -> Result<(), HypervisorError> {
		regs.rip = vmread(guest::RIP)?;
		regs.rsp = vmread(guest::RSP)?;
		regs.rflags = vmread(guest::RFLAGS)?;
		Ok(())
	}

	/// Returns the 32-bit VM-exit reason field.
	pub fn exit_reason(&self) -> Result<u32, HypervisorError> {
		Ok(vmread(ro::EXIT_REASON)? as u32)
	}

	/// Returns the VM-instruction error field (valid after a failed VMX instruction).
	pub fn instruction_error(&self) -> Result<u64, HypervisorError> {
		vmread(ro::VM_INSTRUCTION_ERROR)
	}

	/// Reads a VMCS `field` (e.g. one of the [`guest`], [`host`] or [`ro`]
	/// encodings).
	///
	/// Note that `VMREAD` always operates on the VMCS that is *currently loaded*
	/// on this core (the last one activated with `VMPTRLD`), so this only returns
	/// this region's data while it is the active VMCS.
	pub fn read(&self, field: u32) -> Result<u64, HypervisorError> {
		vmread(field)
	}

	pub fn write(&self, field: u32, value: u64) -> Result<(), HypervisorError> {
		unsafe { vmwrite(field, value) }
	}
}

/// Debug implementation to dump the VMCS fields.
impl fmt::Debug for Vmcs {
	/// Formats the VMCS for display.
	///
	/// # Arguments
	/// * `format` - Formatter instance.
	///
	/// # Returns
	/// Formatting result.
	fn fmt(&self, format: &mut fmt::Formatter<'_>) -> fmt::Result {
		format
			.debug_struct("Vmcs")
			.field("Current VMCS: ", &(self as *const _))
			.field("Revision ID: ", &self.revision_id)
			/* VMCS Guest state fields */
			.field("Guest CR0: ", &vmread(guest::CR0).unwrap_or(0))
			.field("Guest CR3: ", &vmread(guest::CR3).unwrap_or(0))
			.field("Guest CR4: ", &vmread(guest::CR4).unwrap_or(0))
			.field("Guest DR7: ", &vmread(guest::DR7).unwrap_or(0))
			.field("Guest RSP: ", &vmread(guest::RSP).unwrap_or(0))
			.field("Guest RIP: ", &vmread(guest::RIP).unwrap_or(0))
			.field("Guest RFLAGS: ", &vmread(guest::RFLAGS).unwrap_or(0))
			.field(
				"Guest CS Selector: ",
				&vmread(guest::CS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Guest SS Selector: ",
				&vmread(guest::SS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Guest DS Selector: ",
				&vmread(guest::DS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Guest ES Selector: ",
				&vmread(guest::ES_SELECTOR).unwrap_or(0),
			)
			.field(
				"Guest FS Selector: ",
				&vmread(guest::FS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Guest GS Selector: ",
				&vmread(guest::GS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Guest LDTR Selector: ",
				&vmread(guest::LDTR_SELECTOR).unwrap_or(0),
			)
			.field(
				"Guest TR Selector: ",
				&vmread(guest::TR_SELECTOR).unwrap_or(0),
			)
			.field("Guest CS Base: ", &vmread(guest::CS_BASE).unwrap_or(0))
			.field("Guest SS Base: ", &vmread(guest::SS_BASE).unwrap_or(0))
			.field("Guest DS Base: ", &vmread(guest::DS_BASE).unwrap_or(0))
			.field("Guest ES Base: ", &vmread(guest::ES_BASE).unwrap_or(0))
			.field("Guest FS Base: ", &vmread(guest::FS_BASE).unwrap_or(0))
			.field("Guest GS Base: ", &vmread(guest::GS_BASE).unwrap_or(0))
			.field("Guest LDTR Base: ", &vmread(guest::LDTR_BASE).unwrap_or(0))
			.field("Guest TR Base: ", &vmread(guest::TR_BASE).unwrap_or(0))
			.field("Guest CS Limit: ", &vmread(guest::CS_LIMIT).unwrap_or(0))
			.field("Guest SS Limit: ", &vmread(guest::SS_LIMIT).unwrap_or(0))
			.field("Guest DS Limit: ", &vmread(guest::DS_LIMIT).unwrap_or(0))
			.field("Guest ES Limit: ", &vmread(guest::ES_LIMIT).unwrap_or(0))
			.field("Guest FS Limit: ", &vmread(guest::FS_LIMIT).unwrap_or(0))
			.field("Guest GS Limit: ", &vmread(guest::GS_LIMIT).unwrap_or(0))
			.field(
				"Guest LDTR Limit: ",
				&vmread(guest::LDTR_LIMIT).unwrap_or(0),
			)
			.field("Guest TR Limit: ", &vmread(guest::TR_LIMIT).unwrap_or(0))
			.field(
				"Guest CS Access Rights: ",
				&vmread(guest::CS_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field(
				"Guest SS Access Rights: ",
				&vmread(guest::SS_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field(
				"Guest DS Access Rights: ",
				&vmread(guest::DS_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field(
				"Guest ES Access Rights: ",
				&vmread(guest::ES_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field(
				"Guest FS Access Rights: ",
				&vmread(guest::FS_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field(
				"Guest GS Access Rights: ",
				&vmread(guest::GS_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field(
				"Guest LDTR Access Rights: ",
				&vmread(guest::LDTR_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field(
				"Guest TR Access Rights: ",
				&vmread(guest::TR_ACCESS_RIGHTS).unwrap_or(0),
			)
			.field("Guest GDTR Base: ", &vmread(guest::GDTR_BASE).unwrap_or(0))
			.field("Guest IDTR Base: ", &vmread(guest::IDTR_BASE).unwrap_or(0))
			.field(
				"Guest GDTR Limit: ",
				&vmread(guest::GDTR_LIMIT).unwrap_or(0),
			)
			.field(
				"Guest IDTR Limit: ",
				&vmread(guest::IDTR_LIMIT).unwrap_or(0),
			)
			.field(
				"Guest IA32_DEBUGCTL_FULL: ",
				&vmread(guest::IA32_DEBUGCTL_FULL).unwrap_or(0),
			)
			.field(
				"Guest IA32_SYSENTER_CS: ",
				&vmread(guest::IA32_SYSENTER_CS).unwrap_or(0),
			)
			.field(
				"Guest IA32_SYSENTER_ESP: ",
				&vmread(guest::IA32_SYSENTER_ESP).unwrap_or(0),
			)
			.field(
				"Guest IA32_SYSENTER_EIP: ",
				&vmread(guest::IA32_SYSENTER_EIP).unwrap_or(0),
			)
			.field(
				"Guest IA32_EFER_FULL: ",
				&vmread(guest::IA32_EFER_FULL).unwrap_or(0),
			)
			.field(
				"Guest VMCS Link Pointer: ",
				&vmread(guest::LINK_PTR_FULL).unwrap_or(0),
			)
			.field(
				"Guest Activity State: ",
				&vmread(guest::ACTIVITY_STATE).unwrap_or(0),
			)
			/* VMCS Host state fields */
			.field("Host CR0: ", &vmread(host::CR0).unwrap_or(0))
			.field("Host CR3: ", &vmread(host::CR3).unwrap_or(0))
			.field("Host CR4: ", &vmread(host::CR4).unwrap_or(0))
			.field("Host RSP: ", &vmread(host::RSP).unwrap_or(0))
			.field("Host RIP: ", &vmread(host::RIP).unwrap_or(0))
			.field(
				"Host CS Selector: ",
				&vmread(host::CS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Host SS Selector: ",
				&vmread(host::SS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Host DS Selector: ",
				&vmread(host::DS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Host ES Selector: ",
				&vmread(host::ES_SELECTOR).unwrap_or(0),
			)
			.field(
				"Host FS Selector: ",
				&vmread(host::FS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Host GS Selector: ",
				&vmread(host::GS_SELECTOR).unwrap_or(0),
			)
			.field(
				"Host TR Selector: ",
				&vmread(host::TR_SELECTOR).unwrap_or(0),
			)
			.field("Host FS Base: ", &vmread(host::FS_BASE).unwrap_or(0))
			.field("Host GS Base: ", &vmread(host::GS_BASE).unwrap_or(0))
			.field("Host TR Base: ", &vmread(host::TR_BASE).unwrap_or(0))
			.field("Host GDTR Base: ", &vmread(host::GDTR_BASE).unwrap_or(0))
			.field("Host IDTR Base: ", &vmread(host::IDTR_BASE).unwrap_or(0))
			.field(
				"Host IA32_SYSENTER_CS: ",
				&vmread(host::IA32_SYSENTER_CS).unwrap_or(0),
			)
			.field(
				"Host IA32_SYSENTER_ESP: ",
				&vmread(host::IA32_SYSENTER_ESP).unwrap_or(0),
			)
			.field(
				"Host IA32_SYSENTER_EIP: ",
				&vmread(host::IA32_SYSENTER_EIP).unwrap_or(0),
			)
			/* VMCS Control fields */
			.field(
				"Primary Proc Based Execution Controls: ",
				&vmread(control::PRIMARY_PROCBASED_EXEC_CONTROLS).unwrap_or(0),
			)
			.field(
				"Secondary Proc Based Execution Controls: ",
				&vmread(control::SECONDARY_PROCBASED_EXEC_CONTROLS).unwrap_or(0),
			)
			.field(
				"VM Entry Controls: ",
				&vmread(control::VMENTRY_CONTROLS).unwrap_or(0),
			)
			.field(
				"VM Exit Controls: ",
				&vmread(control::VMEXIT_CONTROLS).unwrap_or(0),
			)
			.field(
				"Pin Based Execution Controls: ",
				&vmread(control::PINBASED_EXEC_CONTROLS).unwrap_or(0),
			)
			.field(
				"CR0 Read Shadow: ",
				&vmread(control::CR0_READ_SHADOW).unwrap_or(0),
			)
			.field(
				"CR4 Read Shadow: ",
				&vmread(control::CR4_READ_SHADOW).unwrap_or(0),
			)
			.field(
				"MSR Bitmaps Address: ",
				&vmread(control::MSR_BITMAPS_ADDR_FULL).unwrap_or(0),
			)
			.field("EPT Pointer: ", &vmread(control::EPTP_FULL).unwrap_or(0))
			.field("VPID: ", &vmread(control::VPID).unwrap_or(0))
			.finish_non_exhaustive()
	}
}
