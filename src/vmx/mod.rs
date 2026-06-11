mod ept;
mod run;
mod vmcs;
mod vmerror;
mod vmxon;

pub use ept::Ept;
pub use run::GuestRegisters;
pub use vmerror::VmxBasicExitReason;

use alloc::boxed::Box;
use core::arch::asm;
use core::mem::MaybeUninit;

use hermit::mm::{VirtAddr, virtual_to_physical};
use raw_cpuid::CpuId;
use x86_64::registers::control::{Cr0, Cr0Flags, Cr4, Cr4Flags};
use x86_64::registers::model_specific::Msr;
use x86_64::registers::rflags::{self, RFlags};

use crate::error::HypervisorError;
use crate::vmx::run::run_vmx_vm;
use crate::vmx::vmcs::Vmcs;
use crate::vmx::vmxon::Vmxon;

/// Reporting Register of Basic VMX Capabilities (R/O) See Table 35-2. See Appendix A.1, Basic VMX Information (If CPUID.01H:ECX.\[bit 9\])
const IA32_VMX_BASIC: u32 = 0x480;
/// Capability Reporting Register of CR0 Bits Fixed to 0 (R/O) See Appendix A.7, VMX-Fixed Bits in CR0 (If CPUID.01H:ECX.\[bit 9\])
const IA32_VMX_CR0_FIXED0: u32 = 0x486;
/// Capability Reporting Register of CR0 Bits Fixed to 1 (R/O) See Appendix A.7, VMX-Fixed Bits in CR0 (If CPUID.01H:ECX.\[bit 9\])
const IA32_VMX_CR0_FIXED1: u32 = 0x487;
/// Capability Reporting Register of CR4 Bits Fixed to 0 (R/O) See Appendix A.8, VMX-Fixed Bits in CR4 (If CPUID.01H:ECX.\[bit 9\])
const IA32_VMX_CR4_FIXED0: u32 = 0x488;
/// Capability Reporting Register of CR4 Bits Fixed to 1 (R/O) See Appendix A.8, VMX-Fixed Bits in CR4 (If CPUID.01H:ECX.\[bit 9\])
const IA32_VMX_CR4_FIXED1: u32 = 0x489;

/* desired control word constrained by hardware/hypervisor capabilities */
/*#[inline(always)]
fn cap2ctrl(cap: u64, ctrl: u64) -> u64 {
	(ctrl | (cap & 0xffffffff)) & (cap >> 32)
}*/

/// The Intel VT-x backend of a single virtual CPU.
///
/// Holds the per-CPU VMX state: the VMXON region (enabling VMX operation on the
/// physical core the vCPU runs on), the VMCS, the cached guest registers and the
/// launch state. The Extended Page Tables are *not* owned here — they belong to
/// the virtual machine and are shared between all of its vCPUs, which only need
/// the EPT pointer (a physical address) passed at construction.
pub struct VmxCpu {
	/// The VMXON (Virtual Machine Extensions On) region.
	/// - Aligned to 4096 bytes (0x1000)
	/// - Boxed so the region keeps a stable physical address: a VMXON/VMCS region
	///   is bound to the address it was activated at and must never be moved.
	vmxon_region: Box<Vmxon>,

	/// The VMCS (Virtual Machine Control Structure) for this vCPU.
	/// - Aligned to 4096 bytes (0x1000)
	/// - Boxed for the same reason as [`Self::vmxon_region`].
	vmcs_region: Box<Vmcs>,

	/// Saved guest general-purpose registers.
	regs: GuestRegisters,

	/// Whether the VMCS has already been launched (selects VMLAUNCH vs VMRESUME).
	launched: bool,

	/// Guest entry point (initial RIP).
	entry_point: u64,

	/// Guest initial stack pointer.
	guest_rsp: u64,
}

impl VmxCpu {
	/// Initializes the VMXON and VMCS regions with their revision identifiers.
	fn init(&mut self) -> Result<(), HypervisorError> {
		self.vmxon_region.init();
		self.vmcs_region.init();

		Ok(())
	}

	#[inline(always)]
	fn vmx_capture_status(&self) -> Result<(), HypervisorError> {
		let flags = rflags::read();
		if flags.contains(RFlags::ZERO_FLAG) || flags.contains(RFlags::CARRY_FLAG) {
			Err(HypervisorError::VMCLEARFailed)
		} else {
			Ok(())
		}
	}

	/// Enables VMX operation on the current physical core and executes VMXON.
	///
	/// VMXON is a *per physical core* operation: once a core is in VMX root
	/// operation, executing VMXON again on it fails with VM-instruction error 15
	/// ("VMXON in VMX root operation"). Because hermit does not expose a core id
	/// on x86-64 (only on riscv64), this is performed lazily in each vCPU thread
	/// and such a repeated-VMXON failure is treated as success — the core is
	/// already enabled. A proper per-core guard is future work and is tied to
	/// pinning vCPU threads to cores.
	fn setup_vmxon(&self) -> Result<(), HypervisorError> {
		const IA32_FEATURE_CONTROL: u32 = 0x3a;
		const VMX_LOCK_BIT: u64 = 1 << 0;
		const VMXON_OUTSIDE_SMX: u64 = 1 << 2;
		/// VM-instruction error: "VMXON executed in VMX root operation".
		const VMXON_IN_ROOT: u64 = 15;

		unsafe {
			Cr4::update(|flags| {
				flags.insert(Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS);
			});
		}

		let mut feature_control_register = Msr::new(IA32_FEATURE_CONTROL);
		let mut features = unsafe { feature_control_register.read() };
		if features & VMX_LOCK_BIT == 0 {
			features |= VMX_LOCK_BIT | VMXON_OUTSIDE_SMX;
			unsafe { feature_control_register.write(features) };
		} else if features & VMXON_OUTSIDE_SMX == 0 {
			panic!("Unable to initialize VMX");
		}

		let ia32_vmx_cr0_fixed0 = unsafe { Msr::new(IA32_VMX_CR0_FIXED0).read() };
		let ia32_vmx_cr0_fixed1 = unsafe { Msr::new(IA32_VMX_CR0_FIXED1).read() };

		unsafe {
			Cr0::update(|flags| {
				flags.insert(Cr0Flags::from_bits(ia32_vmx_cr0_fixed0).unwrap());
				flags.intersects(Cr0Flags::from_bits_retain(ia32_vmx_cr0_fixed1));
			});
		}

		let ia32_vmx_cr4_fixed0 = unsafe { Msr::new(IA32_VMX_CR4_FIXED0).read() };
		let ia32_vmx_cr4_fixed1 = unsafe { Msr::new(IA32_VMX_CR4_FIXED1).read() };

		unsafe {
			Cr4::update(|flags| {
				flags.insert(Cr4Flags::from_bits(ia32_vmx_cr4_fixed0).unwrap());
				flags.intersects(Cr4Flags::from_bits_retain(ia32_vmx_cr4_fixed1));
			});
		}

		let vmxon_addr =
			virtual_to_physical(VirtAddr::from_ptr(self.vmxon_region.as_ref() as *const Vmxon))
				.unwrap()
				.as_u64();
		unsafe {
			asm!("vmxon [{0}]", in(reg) &vmxon_addr);
		}

		if self.vmx_capture_status().is_err() {
			// The core may already be in VMX root operation (another vCPU enabled
			// it). If a previous vCPU's VMCS is still current we can confirm this
			// via the VM-instruction error; otherwise report the failure.
			if matches!(self.vmcs_region.instruction_error(), Ok(VMXON_IN_ROOT)) {
				debug!("VMX is already enabled on this core");
			} else {
				return Err(HypervisorError::VMCLEARFailed);
			}
		}

		Ok(())
	}

	/// Clears and loads the VMCS, then configures its control, host and guest
	/// areas. `eptp` is the VM-wide EPT pointer shared by all vCPUs.
	fn setup_vmcs(&mut self, eptp: u64) -> Result<(), HypervisorError> {
		let vmcs_addr =
			virtual_to_physical(VirtAddr::from_ptr(self.vmcs_region.as_ref() as *const Vmcs))
				.unwrap()
				.as_u64();
		// Clear the VMCS region.
		unsafe {
			asm!("vmclear [{0}]", in(reg) &vmcs_addr);
		}
		self.vmx_capture_status()?;

		// Load current VMCS pointer.
		unsafe {
			asm!("vmptrld [{0}]", in(reg) &vmcs_addr);
		}
		self.vmx_capture_status()?;

		self.vmcs_region.setup_controls(eptp)?;
		self.vmcs_region.setup_host()?;
		self.vmcs_region.setup_guest(self.entry_point, self.guest_rsp)
	}

	/// Makes this vCPU's VMCS the current one on the executing core.
	///
	/// Needed before every entry because several vCPUs may share a physical core
	/// and each VMLAUNCH/VMRESUME operates on whichever VMCS is currently loaded.
	fn activate(&self) -> Result<(), HypervisorError> {
		let vmcs_addr =
			virtual_to_physical(VirtAddr::from_ptr(self.vmcs_region.as_ref() as *const Vmcs))
				.unwrap()
				.as_u64();
		unsafe {
			asm!("vmptrld [{0}]", in(reg) &vmcs_addr);
		}
		self.vmx_capture_status()
	}

	/// Executes the vCPU until a VM-exit occurs.
	///
	/// Loads this vCPU's VMCS, restores the guest registers and performs a
	/// VMLAUNCH/VMRESUME. On VM-exit the guest register state is captured and the
	/// exit reason is decoded.
	///
	/// # Returns
	///
	/// Returns `Ok(VmxBasicExitReason)` indicating the reason for the VM-exit, or an `Err(HypervisorError)`
	/// if the VM fails to launch or an unknown exit reason is encountered.
	pub fn run(&mut self) -> Result<VmxBasicExitReason, HypervisorError> {
		// Make sure this vCPU's VMCS is the current one on this core.
		self.activate()?;

		// Mirror the cached RIP/RSP/RFLAGS into the VMCS so a VMRESUME continues
		// where the previous VM-exit left off.
		self.vmcs_region.load_guest_registers(&self.regs)?;

		// Enter the guest. Returns the RFLAGS produced by VMLAUNCH/VMRESUME.
		let flags = unsafe { run_vmx_vm(&mut self.regs) };
		self.launched = true;

		// A set ZF (VMfailValid) or CF (VMfailInvalid) means VM-entry failed and
		// no VM-exit occurred.
		let rflags = RFlags::from_bits_truncate(flags);
		if rflags.contains(RFlags::ZERO_FLAG) || rflags.contains(RFlags::CARRY_FLAG) {
			let error = self.vmcs_region.instruction_error().unwrap_or(0);
			return Err(HypervisorError::VMEntryFailed(error));
		}

		// VM-exit occurred: refresh the cached RIP/RSP/RFLAGS and decode the reason.
		self.vmcs_region.save_guest_registers(&mut self.regs)?;
		let reason = self.vmcs_region.exit_reason()?;
		VmxBasicExitReason::from_u32(reason).ok_or(HypervisorError::UnknownVMExitReason)
	}

	/// Returns the cached guest register state captured at the last VM-exit.
	pub fn guest_registers(&self) -> &GuestRegisters {
		&self.regs
	}

	/// Creates and fully initializes the VT-x backend of a vCPU.
	///
	/// `eptp` is the VM-wide EPT pointer, `cpu_id` becomes the guest's RSI (CPU
	/// id) and `entry_point`/`guest_rsp` its initial RIP/RSP. The boot-info
	/// pointer is passed to the guest in RDI, following the hermit kernel's entry
	/// convention.
	pub fn new(
		eptp: u64,
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

		let mut cpu: VmxCpu = Self {
			vmcs_region: Box::new(unsafe { MaybeUninit::zeroed().assume_init() }),
			vmxon_region: Box::new(unsafe { MaybeUninit::zeroed().assume_init() }),
			regs,
			launched: false,
			entry_point,
			guest_rsp,
		};

		cpu.init()?;
		// Enable VMX on this core and configure the VMCS.
		cpu.setup_vmxon()?;
		cpu.setup_vmcs(eptp)?;

		Ok(cpu)
	}
}

/// Checks if the CPU is supported for hypervisor operation.
///
/// Verifies the CPU is Intel with VMX support and Memory Type Range Registers (MTRRs) support.
///
/// # Returns
///
/// Returns `Ok(())` if the CPU meets all requirements, otherwise returns `Err(HypervisorError)`.
pub fn check_supported_cpu() -> Result<(), HypervisorError> {
	let cpuid = CpuId::new();

	if let Some(vf) = cpuid.get_vendor_info() {
		if vf.as_str() != "GenuineIntel" {
			return Err(HypervisorError::CPUUnsupported);
		}
	} else {
		return Err(HypervisorError::CPUUnsupported);
	}

	if !cpuid
		.get_feature_info()
		.is_some_and(|finfo| finfo.has_vmx())
	{
		Err(HypervisorError::VMXUnsupported)
	} else {
		Ok(())
	}
}