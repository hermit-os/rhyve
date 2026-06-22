//! AMD-V Virtual Machine Control Block (VMCB).
//!
//! The VMCB is the AMD-V counterpart of the Intel VMCS: a single 4 KiB,
//! page-aligned structure holding both the *control area* (intercepts, nested
//! paging, exit information) at offsets `0x000..0x400` and the *state-save area*
//! (the guest's architectural register state) from `0x400` onwards. Unlike the
//! VMCS, it is plain memory accessed by ordinary loads and stores rather than
//! through `VMREAD`/`VMWRITE`, so this module reads and writes its fields by
//! offset.
//!
//! Reference: AMD64 Architecture Programmer's Manual, Volume 2, Appendix B
//! "Layout of VMCB".

#![allow(dead_code)]

use core::ptr;

use x86_64::structures::paging::page::{PageSize, Size4KiB as BasePageSize};

// --- Control-area field offsets ----------------------------------------------

/// Intercept word for assorted instructions (CPUID, IOIO, MSR, SHUTDOWN, ...).
const INTERCEPT_INSTR1: usize = 0x00C;
/// Intercept word for the SVM instructions plus a few others (VMRUN, XSETBV, ...).
const INTERCEPT_INSTR2: usize = 0x010;
/// Physical base address of the I/O-permission map.
const IOPM_BASE_PA: usize = 0x040;
/// Physical base address of the MSR-permission map.
const MSRPM_BASE_PA: usize = 0x048;
/// Guest ASID (bits 31:0) and TLB-control byte (bits 39:32).
const GUEST_ASID: usize = 0x058;
/// Guest interrupt-state field; bit 0 is the interrupt shadow (set after `STI` /
/// `MOV SS`, when an interrupt may not yet be delivered).
pub const INT_STATE: usize = 0x068;
/// `#VMEXIT` reason.
pub const EXITCODE: usize = 0x070;
/// `#VMEXIT` information field 1.
pub const EXITINFO1: usize = 0x078;
/// `#VMEXIT` information field 2.
pub const EXITINFO2: usize = 0x080;
/// Nested-paging control; bit 0 enables nested paging.
const NP_ENABLE: usize = 0x090;
/// Nested CR3 (`nCR3`).
const N_CR3: usize = 0x0B0;
/// Event-injection field: an event the processor delivers to the guest on the
/// next `VMRUN` (valid bit 31, type bits 10:8, vector bits 7:0). Unlike the
/// guest's own pending interrupts, an injected event ignores `RFLAGS.IF`, so the
/// backend gates injection on guest interruptibility itself.
pub const EVENTINJ: usize = 0x0A8;
/// Next sequential instruction pointer (valid for many intercepts when the CPU
/// supports the NRIPS feature).
pub const NRIP: usize = 0x0C8;

// Intercept bits within `INTERCEPT_INSTR1`.
/// Physical (external) interrupt: any host interrupt arriving while the guest
/// runs causes a clean `#VMEXIT (VMEXIT_INTR)` instead of being injected into
/// the guest, so the host keeps ownership of its own device interrupts.
const INTERCEPT_INTR: u32 = 1 << 0;
const INTERCEPT_CPUID: u32 = 1 << 18;
const INTERCEPT_HLT: u32 = 1 << 24;
const INTERCEPT_PAUSE: u32 = 1 << 23;
const INTERCEPT_IOIO: u32 = 1 << 27;
const INTERCEPT_MSR: u32 = 1 << 28;
const INTERCEPT_SHUTDOWN: u32 = 1 << 31;
// Intercept bits within `INTERCEPT_INSTR2`.
const INTERCEPT_VMRUN: u32 = 1 << 0;
const INTERCEPT_XSETBV: u32 = 1 << 13;

// --- State-save-area field offsets (absolute, from the VMCB base) ------------

/// State-save area: ES segment (the first segment slot).
const SAVE_ES: usize = 0x400;
const SAVE_CS: usize = 0x410;
const SAVE_SS: usize = 0x420;
const SAVE_DS: usize = 0x430;
const SAVE_FS: usize = 0x440;
const SAVE_GS: usize = 0x450;
const SAVE_GDTR: usize = 0x460;
const SAVE_LDTR: usize = 0x470;
const SAVE_IDTR: usize = 0x480;
const SAVE_TR: usize = 0x490;
/// Current privilege level (one byte).
const SAVE_CPL: usize = 0x4CB;
const SAVE_EFER: usize = 0x4D0;
const SAVE_CR4: usize = 0x548;
const SAVE_CR3: usize = 0x550;
const SAVE_CR0: usize = 0x558;
const SAVE_DR7: usize = 0x560;
const SAVE_DR6: usize = 0x568;
const SAVE_RFLAGS: usize = 0x570;
const SAVE_RIP: usize = 0x578;
const SAVE_RSP: usize = 0x5D8;
const SAVE_RAX: usize = 0x5F8;
const SAVE_CR2: usize = 0x640;
/// Guest PAT, consulted while nested paging is active.
const SAVE_G_PAT: usize = 0x648;

/// The base address of a segment lives 8 bytes into its 16-byte slot
/// (`selector:u16, attrib:u16, limit:u32, base:u64`).
const SEG_BASE: usize = 8;

/// Guest RIP within the state-save area (mirrored by the run loop).
pub const RIP: usize = SAVE_RIP;
/// Guest RSP within the state-save area.
pub const RSP: usize = SAVE_RSP;
/// Guest RAX within the state-save area (the other GPRs live in the trampoline's
/// register block, but RAX is part of the VMCB-managed state).
pub const RAX: usize = SAVE_RAX;
/// Guest RFLAGS within the state-save area.
pub const RFLAGS: usize = SAVE_RFLAGS;
/// Guest IA32_EFER within the state-save area.
pub const EFER: usize = SAVE_EFER;
/// Guest FS base within the state-save area.
pub const FS_BASE: usize = SAVE_FS + SEG_BASE;
/// Guest GS base within the state-save area.
pub const GS_BASE: usize = SAVE_GS + SEG_BASE;

/// `EFER.SVME`: the guest must keep secure-virtual-machine mode enabled, or
/// `VMRUN` aborts the consistency check with `VMEXIT_INVALID`.
pub const EFER_SVME: u64 = 1 << 12;

/// The Virtual Machine Control Block.
///
/// A 4 KiB, page-aligned region; the processor keeps it at a stable host-physical
/// address (the owner boxes it for that reason). Fields are accessed by offset
/// rather than as named struct members because the control and state-save areas
/// are sparse and partly reserved.
#[repr(C, align(4096))]
pub struct Vmcb {
	data: [u8; BasePageSize::SIZE as usize],
}

impl Vmcb {
	/// Reads a 64-bit field at `offset`.
	pub fn read_u64(&self, offset: usize) -> u64 {
		// SAFETY: `offset` is one of the in-range field constants above; the read
		// stays within `data` and unaligned access is permitted on x86-64.
		unsafe { ptr::read_unaligned(self.data.as_ptr().add(offset).cast::<u64>()) }
	}

	/// Writes a 64-bit field at `offset`.
	pub fn write_u64(&mut self, offset: usize, value: u64) {
		// SAFETY: see [`Vmcb::read_u64`].
		unsafe { ptr::write_unaligned(self.data.as_mut_ptr().add(offset).cast::<u64>(), value) }
	}

	fn write_u32(&mut self, offset: usize, value: u32) {
		// SAFETY: see [`Vmcb::read_u64`].
		unsafe { ptr::write_unaligned(self.data.as_mut_ptr().add(offset).cast::<u32>(), value) }
	}

	fn write_u16(&mut self, offset: usize, value: u16) {
		// SAFETY: see [`Vmcb::read_u64`].
		unsafe { ptr::write_unaligned(self.data.as_mut_ptr().add(offset).cast::<u16>(), value) }
	}

	fn write_u8(&mut self, offset: usize, value: u8) {
		self.data[offset] = value;
	}

	/// Writes a segment slot: `selector`, packed `attrib`, `limit` and `base`.
	fn write_segment(&mut self, slot: usize, selector: u16, attrib: u16, limit: u32, base: u64) {
		self.write_u16(slot, selector);
		self.write_u16(slot + 2, attrib);
		self.write_u32(slot + 4, limit);
		self.write_u64(slot + 8, base);
	}

	/// Configures the control area: the intercepts the backend handles, the
	/// permission maps, the guest ASID and nested paging.
	///
	/// `ncr3` is the nested CR3 from [`Npt::ncr3`](super::npt::Npt::ncr3),
	/// `iopm_pa`/`msrpm_pa` the host-physical addresses of the I/O- and
	/// MSR-permission maps.
	pub fn setup_control(&mut self, ncr3: u64, iopm_pa: u64, msrpm_pa: u64) {
		// Intercept the instructions the backend emulates on the host's behalf,
		// plus physical interrupts so the host stays responsive while the guest
		// runs (see `INTERCEPT_INTR`). HLT is intercepted so an idle guest hands
		// control back to the backend, which polls the emulated APIC-timer deadline
		// there — AMD-V has no VMX-preemption-timer equivalent to raise a timed
		// exit on its own.
		self.write_u32(
			INTERCEPT_INSTR1,
			INTERCEPT_INTR
				| INTERCEPT_CPUID
				| INTERCEPT_HLT
				| INTERCEPT_PAUSE
				| INTERCEPT_IOIO
				| INTERCEPT_MSR
				| INTERCEPT_SHUTDOWN,
		);
		// VMRUN *must* be intercepted, otherwise VMRUN itself faults.
		self.write_u32(INTERCEPT_INSTR2, INTERCEPT_VMRUN | INTERCEPT_XSETBV);

		self.write_u64(IOPM_BASE_PA, iopm_pa);
		self.write_u64(MSRPM_BASE_PA, msrpm_pa);

		// ASID 0 is reserved for the host; tag the guest TLB entries with ASID 1
		// and request a full TLB flush on entry (TLB-control byte = 1).
		self.write_u64(GUEST_ASID, 1 | (1 << 32));

		// Enable nested paging and point the processor at the guest's NPT.
		self.write_u64(NP_ENABLE, 1);
		self.write_u64(N_CR3, ncr3);
	}

	/// Configures the state-save area for a freshly booted 64-bit kernel, mirroring
	/// the VT-x backend's guest setup. Segment bases are flat, the boot GDT lives in
	/// guest memory and `entry_point`/`rsp` become the initial RIP/RSP.
	pub fn setup_guest(&mut self, entry_point: u64, rsp: u64) {
		let code = crate::GDT_KERNEL_CODE << 3;
		let data = crate::GDT_KERNEL_DATA << 3;

		// Packed segment attributes (descriptor bits [55:52][47:40]):
		// present 64-bit code (0xA9B), present 4 GiB data (0xC93) and a busy
		// 64-bit TSS (0x08B). The LDTR is left not-present (attrib 0).
		const CS_ATTRIB: u16 = 0xA9B;
		const DATA_ATTRIB: u16 = 0xC93;
		const TR_ATTRIB: u16 = 0x08B;
		// Code/data use page granularity (G = 1), so a 20-bit limit field covers
		// 4 GiB. TR is byte-granular with a 16-bit limit.
		const SEG_LIMIT: u32 = 0xF_FFFF;

		self.write_segment(SAVE_CS, code, CS_ATTRIB, SEG_LIMIT, 0);
		self.write_segment(SAVE_SS, data, DATA_ATTRIB, SEG_LIMIT, 0);
		self.write_segment(SAVE_DS, data, DATA_ATTRIB, SEG_LIMIT, 0);
		self.write_segment(SAVE_ES, data, DATA_ATTRIB, SEG_LIMIT, 0);
		self.write_segment(SAVE_FS, data, DATA_ATTRIB, SEG_LIMIT, 0);
		self.write_segment(SAVE_GS, data, DATA_ATTRIB, SEG_LIMIT, 0);
		self.write_segment(SAVE_TR, 0, TR_ATTRIB, 0xFFFF, 0);
		self.write_segment(SAVE_LDTR, 0, 0, 0xFFFF, 0);

		// Descriptor tables: the boot GDT lives in guest memory, the guest starts
		// without an IDT.
		self.write_segment(
			SAVE_GDTR,
			0,
			0,
			((core::mem::size_of::<u64>() * crate::BOOT_GDT_MAX) - 1) as u32,
			crate::GDT_OFFSET,
		);
		self.write_segment(SAVE_IDTR, 0, 0, 0xFFFF, 0);

		self.write_u8(SAVE_CPL, 0);

		// Control registers: protected mode + paging on and PAE for long mode. Only
		// PAE is set, matching the (working) VT-x backend's initial CR4: the hermit
		// kernel enables FSGSBASE itself during early boot ("Enable FSGSBASE
		// support"), so handing it a guest that already has CR4.FSGSBASE set diverges
		// from that path and trips an early-boot assertion before the console exists.
		const CR0: u64 = (1 << 0) | (1 << 4) | (1 << 5) | (1 << 31); // PE | ET | NE | PG
		const CR4: u64 = 1 << 5; // PAE
		self.write_u64(SAVE_CR0, CR0);
		self.write_u64(SAVE_CR3, crate::BOOT_PML4);
		self.write_u64(SAVE_CR4, CR4);
		self.write_u64(SAVE_CR2, 0);
		// Long mode enabled and active; SVME kept on for the consistency check.
		self.write_u64(SAVE_EFER, crate::EFER_LME | crate::EFER_LMA | EFER_SVME);

		self.write_u64(SAVE_DR7, 0x400);
		self.write_u64(SAVE_DR6, 0xFFFF_0FF0);
		self.write_u64(SAVE_RFLAGS, 0x2); // only the reserved bit 1 is set
		self.write_u64(SAVE_RIP, entry_point);
		self.write_u64(SAVE_RSP, rsp);
		self.write_u64(SAVE_RAX, 0);
		// Default PAT (the processor needs a sane G_PAT under nested paging).
		self.write_u64(SAVE_G_PAT, 0x0007_0406_0007_0406);
	}
}
