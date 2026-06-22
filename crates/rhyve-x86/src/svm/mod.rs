//! The AMD-V (SVM) backend of a single virtual CPU.
//!
//! The structural counterpart of the [`vmx`](crate::vmx) module. Where VT-x has a
//! VMXON region, a VMCS and Extended Page Tables, AMD-V has a host-save area, a
//! Virtual Machine Control Block ([`Vmcb`]) and Nested Page Tables ([`Npt`]). The
//! backend implements the same [`VcpuBackend`] trait, so [`Cpu`](crate) drives it
//! identically to the VT-x backend; the run loop only differs in the mechanics of
//! entering the guest and decoding the exit.

mod exitcode;
mod npt;
mod run;
mod vmcb;

use alloc::alloc::{Layout, alloc};
use alloc::boxed::Box;
use core::arch::asm;
use core::arch::x86_64::{__cpuid_count, _rdtsc};
use core::mem::MaybeUninit;

pub use npt::Npt;
use rhyve_core::error::HypervisorError;
use rhyve_core::{ExitReason, VcpuBackend};
pub use run::GuestRegisters;
use run::run_svm_vm;
use vmcb::Vmcb;
use x86_64::addr::VirtAddr;
use x86_64::registers::model_specific::Msr;
use x86_64::structures::paging::page::{PageSize, Size4KiB as BasePageSize};

use crate::apic::*;
use crate::fpu::FpuState;
use crate::virtual_to_physical;

/// IA32_EFER model-specific register.
const IA32_EFER: u32 = 0xC000_0080;
/// `EFER.SVME`: enables SVM instructions (`VMRUN`, `VMLOAD`, ...).
const EFER_SVME: u64 = 1 << 12;
/// VM_HSAVE_PA: physical address of the area `VMRUN` saves host state to.
const VM_HSAVE_PA: u32 = 0xC001_0117;

/// IA32_FS_BASE model-specific register.
const IA32_FS_BASE: u32 = 0xC000_0100;
/// IA32_GS_BASE model-specific register.
const IA32_GS_BASE: u32 = 0xC000_0101;

/// A 4 KiB permission bitmap (the I/O- or MSR-permission map).
///
/// Both maps are kept to a single page filled with `1` bits, intercepting every
/// port / MSR whose control bit lies in the first 4 KiB block. That covers all of
/// the standard I/O ports (`0x0000..0x7FFF`) and the standard plus `0xC000_xxxx`
/// MSR ranges — i.e. everything a booting kernel touches. (The architecture
/// permits the maps to span 12 KiB / 8 KiB; the remaining high ports / AMD MSRs
/// are not intercepted, matching what the guest never uses.)
#[repr(C, align(4096))]
struct Bitmap {
	data: [u8; BasePageSize::SIZE as usize],
}

impl Bitmap {
	/// Allocates a page-aligned bitmap with every bit set.
	fn all_ones() -> Result<Box<Self>, HypervisorError> {
		let layout = Layout::new::<Bitmap>();
		// SAFETY: the layout is non-zero sized; the whole page is initialized
		// immediately below before any read.
		let ptr = unsafe { alloc(layout) }.cast::<Bitmap>();
		if ptr.is_null() {
			return Err(HypervisorError::AllocationFailed);
		}
		// SAFETY: `ptr` is freshly allocated and aligned for `Bitmap`.
		unsafe {
			ptr.write(Bitmap {
				data: [0xFF; BasePageSize::SIZE as usize],
			})
		};
		// SAFETY: `ptr` now points at an initialized `Bitmap`.
		Ok(unsafe { Box::from_raw(ptr) })
	}

	fn host_physical(&self) -> Result<u64, HypervisorError> {
		host_physical((self as *const Bitmap).cast())
	}
}

/// A zeroed, page-aligned scratch region (the host-save area `VMRUN` writes to).
#[repr(C, align(4096))]
struct HostSaveArea {
	_data: [u8; BasePageSize::SIZE as usize],
}

/// Resolves the host-physical address backing a host-virtual pointer.
fn host_physical(ptr: *const u8) -> Result<u64, HypervisorError> {
	virtual_to_physical(VirtAddr::from_ptr(ptr))
		.map(|pa| pa.as_u64())
		.ok_or(HypervisorError::AllocationFailed)
}

/// The AMD-V backend of a single virtual CPU.
///
/// Holds the per-CPU SVM state. The Nested Page Tables are *not* owned here — they
/// belong to the virtual machine and are shared between all of its vCPUs, which
/// only need the nested CR3 (a physical address) passed at construction.
pub struct SvmCpu {
	/// The guest VMCB; boxed for a stable host-physical address, like a VMCS.
	vmcb: Box<Vmcb>,
	/// A scratch VMCB the trampoline `VMSAVE`s the host's hidden segment state
	/// into, so it can be restored after the guest runs. Owned here for its stable
	/// host-physical address ([`Self::host_vmcb_pa`]); never read directly.
	#[allow(dead_code)]
	host_vmcb: Box<Vmcb>,
	/// The area `VMRUN` saves the host's core register state to (`VM_HSAVE_PA`).
	/// Owned here for its stable host-physical address ([`Self::host_save_pa`]);
	/// never read directly.
	#[allow(dead_code)]
	host_save_area: Box<HostSaveArea>,
	/// I/O-permission map referenced by the VMCB control area.
	iopm: Box<Bitmap>,
	/// MSR-permission map referenced by the VMCB control area.
	msrpm: Box<Bitmap>,

	/// Cached host-physical addresses of the two VMCBs and the host-save area.
	vmcb_pa: u64,
	host_vmcb_pa: u64,
	host_save_pa: u64,

	/// Saved guest general-purpose registers.
	regs: GuestRegisters,

	/// Emulated local-APIC timer state.
	apic_timer: ApicTimer,

	/// Interrupt vector waiting to be injected into the guest (e.g. an expired
	/// timer), if the guest could not accept it at the time it became pending.
	pending_vector: Option<u8>,

	/// The guest's extended FP state, preserved across runs ([`fpu`](crate::fpu)).
	guest_fpu: Box<FpuState>,
	/// Scratch area holding the host's FP state while the guest runs.
	host_fpu: Box<FpuState>,
}

impl SvmCpu {
	/// Creates and fully initializes the AMD-V backend of a vCPU.
	///
	/// `ncr3` is the VM-wide nested CR3, `cpu_id` becomes the guest's RSI (CPU id)
	/// and `entry_point`/`guest_rsp` its initial RIP/RSP. The boot-info pointer is
	/// passed to the guest in RDI, following the hermit kernel's entry convention.
	/// AMD-V has no APIC-access-page mechanism, so `_apic_access_hpa` is unused (the
	/// local-APIC MMIO is backed as plain memory through the NPT); it is accepted
	/// only to share a signature with the VT-x backend.
	pub fn new(
		ncr3: u64,
		_apic_access_hpa: u64,
		cpu_id: u64,
		entry_point: u64,
		guest_rsp: u64,
	) -> Result<Self, HypervisorError> {
		let regs = GuestRegisters {
			rdi: crate::BOOT_INFO_OFFSET, // boot-info pointer (hermit entry arg 0)
			rsi: cpu_id,                  // CPU id (hermit entry arg 1)
			rip: entry_point,
			rsp: guest_rsp,
			rflags: 0x2, // only the reserved bit 1 is set
			..GuestRegisters::default()
		};

		let vmcb: Box<Vmcb> = Box::new(unsafe { MaybeUninit::zeroed().assume_init() });
		let host_vmcb: Box<Vmcb> = Box::new(unsafe { MaybeUninit::zeroed().assume_init() });
		let host_save_area: Box<HostSaveArea> =
			Box::new(unsafe { MaybeUninit::zeroed().assume_init() });
		let iopm = Bitmap::all_ones()?;
		let msrpm = Bitmap::all_ones()?;

		let vmcb_pa = host_physical((vmcb.as_ref() as *const Vmcb).cast())?;
		let host_vmcb_pa = host_physical((host_vmcb.as_ref() as *const Vmcb).cast())?;
		let host_save_pa = host_physical((host_save_area.as_ref() as *const HostSaveArea).cast())?;

		let mut cpu = Self {
			vmcb,
			host_vmcb,
			host_save_area,
			iopm,
			msrpm,
			vmcb_pa,
			host_vmcb_pa,
			host_save_pa,
			regs,
			apic_timer: ApicTimer::default(),
			pending_vector: None,
			guest_fpu: Box::new(FpuState::init_guest()),
			host_fpu: Box::new(FpuState::scratch()),
		};

		cpu.vmcb
			.setup_control(ncr3, cpu.iopm.host_physical()?, cpu.msrpm.host_physical()?);
		cpu.vmcb.setup_guest(entry_point, guest_rsp);
		cpu.setup_svm();

		Ok(cpu)
	}

	/// Enables SVM operation on the current physical core.
	///
	/// Sets `EFER.SVME` and points `VM_HSAVE_PA` at this vCPU's host-save area.
	/// Like the VT-x backend's `VMXON`, this is per physical core; because hermit
	/// does not expose a core id it is performed (idempotently) for every vCPU and
	/// refreshed before each run, since several vCPUs may share a core.
	fn setup_svm(&self) {
		let mut efer = Msr::new(IA32_EFER);
		let value = unsafe { efer.read() };
		if value & EFER_SVME == 0 {
			unsafe { efer.write(value | EFER_SVME) };
		}

		unsafe { Msr::new(VM_HSAVE_PA).write(self.host_save_pa) };
	}

	/// Emulates an MSR-read intercept by reflecting the guest's own VMCB-resident
	/// MSR state. Unknown MSRs read as zero, matching the VT-x backend.
	fn read_msr(&self, index: u32) -> u64 {
		match index {
			IA32_FS_BASE => self.vmcb.read_u64(vmcb::FS_BASE),
			IA32_GS_BASE => self.vmcb.read_u64(vmcb::GS_BASE),
			IA32_EFER => self.vmcb.read_u64(vmcb::EFER) & !vmcb::EFER_SVME,
			// Mirror back the armed local-APIC timer deadline.
			IA32_TSC_DEADLINE => self.apic_timer.deadline.unwrap_or(0),
			_ => 0,
		}
	}

	/// Writes a guest MSR back into the VMCB-resident state. Writes to unknown MSRs
	/// are ignored, matching the VT-x backend. `EFER.SVME` is kept set regardless,
	/// or the next `VMRUN` would fail its consistency check.
	fn write_msr(&mut self, index: u32, value: u64) {
		match index {
			IA32_FS_BASE => self.vmcb.write_u64(vmcb::FS_BASE, value),
			IA32_GS_BASE => self.vmcb.write_u64(vmcb::GS_BASE, value),
			IA32_EFER => self.vmcb.write_u64(vmcb::EFER, value | vmcb::EFER_SVME),
			// Local-APIC timer programming (x2APIC): record it and let the run loop
			// inject the interrupt when the deadline elapses.
			IA32_X2APIC_LVT_TIMER | IA32_TSC_DEADLINE | IA32_X2APIC_INIT_COUNT => {
				self.write_apic_timer_msr(index, value);
			}
			_ => {}
		}
	}

	/// Records a guest write to a local-APIC timer MSR, updating the emulated
	/// timer and its host-TSC deadline. As no TSC offsetting is configured, the
	/// guest's RDTSC reads the host TSC, so a programmed TSC deadline is directly
	/// comparable to the host's `_rdtsc()`. Mirrors the VT-x backend.
	fn write_apic_timer_msr(&mut self, msr: u32, value: u64) {
		match msr {
			IA32_X2APIC_LVT_TIMER => {
				self.apic_timer.vector = (value & 0xff) as u8;
				self.apic_timer.masked = value & LVT_MASKED != 0;
				let mode = value & LVT_TIMER_MODE_MASK;
				self.apic_timer.tsc_deadline_mode = mode == LVT_TIMER_MODE_TSC_DEADLINE;
				self.apic_timer.periodic = mode == LVT_TIMER_MODE_PERIODIC;
				// Masking the timer disarms any pending deadline.
				if self.apic_timer.masked {
					self.apic_timer.deadline = None;
					self.apic_timer.period = None;
				}
			}
			IA32_TSC_DEADLINE => {
				// Absolute TSC deadline; writing 0 disarms the timer.
				self.apic_timer.deadline = (value != 0 && !self.apic_timer.masked).then_some(value);
				self.apic_timer.period = None; // TSC-deadline mode is one-shot
			}
			IA32_X2APIC_INIT_COUNT => {
				// One-shot / periodic count mode. The APIC timer input frequency is
				// not recovered here; approximate one count as one TSC tick, which
				// keeps the interval in the right ballpark. Guests that support the
				// TSC-deadline timer (the common case) use the MSR above instead.
				if value == 0 || self.apic_timer.masked || self.apic_timer.tsc_deadline_mode {
					self.apic_timer.deadline = None;
					self.apic_timer.period = None;
				} else {
					let ticks = value;
					self.apic_timer.deadline = Some(unsafe { _rdtsc() } + ticks);
					self.apic_timer.period = self.apic_timer.periodic.then_some(ticks);
				}
			}
			_ => {}
		}
	}

	/// Services the emulated APIC timer before a `VMRUN`: if its deadline has
	/// elapsed, makes the timer vector pending and re-arms a periodic timer for the
	/// next period. Called on every exit, so a busy guest is polled through its
	/// natural exits and an idle one through the intercepted HLT exits.
	fn service_timer(&mut self) {
		let Some(deadline) = self.apic_timer.deadline else {
			return;
		};
		if unsafe { _rdtsc() } < deadline {
			return;
		}

		self.apic_timer.deadline = None;
		if !self.apic_timer.masked {
			self.pending_vector = Some(self.apic_timer.vector);
		}
		if let Some(period) = self.apic_timer.period {
			self.apic_timer.deadline = Some(deadline.saturating_add(period));
		}
	}

	/// Injects a pending interrupt vector into the guest via the VMCB event-
	/// injection field, but only once the guest is interruptible — an injected
	/// event ignores `RFLAGS.IF`, so delivering it while interrupts are masked
	/// would violate the guest's expectations. While not interruptible the vector
	/// stays pending and is retried on the next exit. The field is rewritten every
	/// entry (the processor does not clear it) so a stale event is never replayed.
	fn inject_pending(&mut self) {
		let interruptible = self.regs.rflags & (1 << 9) != 0 // RFLAGS.IF
			&& self.vmcb.read_u64(vmcb::INT_STATE) & 1 == 0; // no interrupt shadow

		match self.pending_vector {
			Some(vector) if interruptible => {
				// Valid (bit 31) | external-interrupt type (bits 10:8 = 0) | vector.
				let event = (1u64 << 31) | u64::from(vector);
				self.vmcb.write_u64(vmcb::EVENTINJ, event);
				self.pending_vector = None;
			}
			_ => self.vmcb.write_u64(vmcb::EVENTINJ, 0),
		}
	}
}

impl VcpuBackend<GuestRegisters> for SvmCpu {
	/// Executes the vCPU until a `#VMEXIT` occurs.
	///
	/// Refreshes the per-core SVM state, mirrors the cached RAX/RIP/RSP/RFLAGS into
	/// the VMCB, performs the `VMRUN` and decodes the exit reason.
	fn run(&mut self) -> Result<ExitReason, HypervisorError> {
		// Make sure SVM is enabled and VM_HSAVE_PA points here on this core.
		self.setup_svm();

		// Mirror the cached RAX/RIP/RSP/RFLAGS into the VMCB so VMRUN continues
		// where the previous #VMEXIT left off.
		self.vmcb.write_u64(vmcb::RAX, self.regs.rax);
		self.vmcb.write_u64(vmcb::RIP, self.regs.rip);
		self.vmcb.write_u64(vmcb::RSP, self.regs.rsp);
		self.vmcb.write_u64(vmcb::RFLAGS, self.regs.rflags);

		// Emulated local-APIC timer: AMD-V has no preemption timer, so poll the
		// deadline here and inject the expired timer interrupt once the guest can
		// take it (the vector stays pending otherwise).
		self.service_timer();
		self.inject_pending();

		// Swap the host's live FP state out for the guest's so the guest's
		// register file survives the run (VMRUN preserves none of it); see
		// [`fpu`](crate::fpu).
		self.host_fpu.save();
		self.guest_fpu.restore();

		// Enter the guest. The trampoline saves/restores the remaining GPRs.
		unsafe { run_svm_vm(&mut self.regs, self.vmcb_pa, self.host_vmcb_pa) };

		// Capture the guest's FP state and put the host's back before any host
		// code (which may use FP) runs again.
		self.guest_fpu.save();
		self.host_fpu.restore();

		// #VMEXIT occurred: refresh the cached RAX/RIP/RSP/RFLAGS and decode.
		self.regs.rax = self.vmcb.read_u64(vmcb::RAX);
		self.regs.rip = self.vmcb.read_u64(vmcb::RIP);
		self.regs.rsp = self.vmcb.read_u64(vmcb::RSP);
		self.regs.rflags = self.vmcb.read_u64(vmcb::RFLAGS);

		let exitcode = self.vmcb.read_u64(vmcb::EXITCODE);

		match exitcode {
			exitcode::INTR => {
				// A physical interrupt exited the guest; the host already serviced
				// it (the trampoline's `stgi` delivered it). Resume the guest at the
				// same RIP — the interrupt was asynchronous, so nothing to advance.
				Ok(ExitReason::Success)
			}
			exitcode::PAUSE => {
				self.regs.rip += 2;
				Ok(ExitReason::Success)
			}
			exitcode::HLT => {
				// The guest halted waiting for an interrupt. Advance past the HLT
				// (`F4`, one byte) and resume: the emulated APIC timer is polled and
				// injected before the next entry, so an idle guest keeps making
				// progress instead of stalling for want of a timed exit.
				self.regs.rip += 1;
				Ok(ExitReason::Success)
			}
			exitcode::CPUID => {
				// Execute CPUID on behalf of the guest and pass the result back.
				// `eax` selects the leaf, `ecx` the sub-leaf. (As with the VT-x
				// backend the host values are forwarded verbatim for now.)
				let result = __cpuid_count(self.regs.rax as u32, self.regs.rcx as u32);
				self.regs.rax = u64::from(result.eax);
				self.regs.rbx = u64::from(result.ebx);
				self.regs.rcx = u64::from(result.ecx);
				self.regs.rdx = u64::from(result.edx);

				// CPUID is a fixed two-byte instruction (`0F A2`).
				self.regs.rip += 2;
				Ok(ExitReason::Success)
			}
			exitcode::MSR => {
				// EXITINFO1 bit 0: 0 = RDMSR, 1 = WRMSR. `rcx` holds the MSR index.
				let is_write = self.vmcb.read_u64(vmcb::EXITINFO1) & 1 != 0;
				let index = self.regs.rcx as u32;
				if is_write {
					let value = (self.regs.rdx << 32) | (self.regs.rax & 0xffff_ffff);
					self.write_msr(index, value);
				} else {
					let value = self.read_msr(index);
					self.regs.rax = value & 0xffff_ffff; // EAX = low
					self.regs.rdx = value >> 32; // EDX = high
				}

				// RDMSR/WRMSR are fixed two-byte instructions (`0F 32` / `0F 30`).
				self.regs.rip += 2;
				Ok(ExitReason::Success)
			}
			exitcode::XSETBV => {
				// XSETBV is intercepted, so execute it on behalf of the guest:
				// `ecx` selects the XCR, `edx:eax` is the value.
				let value = (self.regs.rdx << 32) | (self.regs.rax & 0xffff_ffff);
				unsafe {
					asm!(
						"xsetbv",
						in("ecx") self.regs.rcx as u32,
						in("eax") value as u32,
						in("edx") (value >> 32) as u32,
					);
				}

				// XSETBV is a fixed three-byte instruction (`0F 01 D1`).
				self.regs.rip += 3;
				Ok(ExitReason::Success)
			}
			exitcode::IOIO => {
				// EXITINFO1 describes the access, EXITINFO2 holds the RIP of the
				// instruction following the I/O. Repackage the access into the same
				// layout the VT-x exit qualification uses (bit 3 = IN, bits 16:31 =
				// port), so the architecture-independent `Cpu` decodes it uniformly.
				let info1 = self.vmcb.read_u64(vmcb::EXITINFO1);
				let is_in = info1 & 1; // bit 0: 1 = IN, 0 = OUT
				let port = (info1 >> 16) & 0xFFFF;
				let qualification = (port << 16) | (is_in << 3);

				self.regs.rip = self.vmcb.read_u64(vmcb::EXITINFO2);
				Ok(ExitReason::IoInstruction(qualification))
			}
			exitcode::SHUTDOWN => {
				// A triple fault is used to shut the system down.
				Ok(ExitReason::Shutdown)
			}
			exitcode::INVALID => {
				// A VMCB consistency check failed, so the guest never started.
				Err(HypervisorError::VMEntryFailed(0))
			}
			_ => {
				warn!("Unhandled exit code at {:x}: {exitcode:#x}", self.regs.rip);
				Err(HypervisorError::UnknownVMExitReason)
			}
		}
	}

	fn guest_registers(&self) -> &GuestRegisters {
		&self.regs
	}

	fn guest_registers_mut(&mut self) -> &mut GuestRegisters {
		&mut self.regs
	}
}
