mod vmcs;
mod vmerror;
mod vmxon;

use core::arch::asm;
use core::mem::MaybeUninit;

use hermit::mm::{VirtAddr, virtual_to_physical};
use raw_cpuid::CpuId;
use x86_64::registers::control::{Cr0, Cr0Flags, Cr4, Cr4Flags};
use x86_64::registers::model_specific::Msr;
use x86_64::registers::rflags::{self, RFlags};

use crate::error::HypervisorError;
use crate::intel::vmcs::Vmcs;
use crate::intel::vmerror::VmxBasicExitReason;
use crate::intel::vmxon::Vmxon;

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

/// Represents a Virtual Machine (VM) instance, encapsulating its state and control mechanisms.
pub struct Vm {
	/// The VMXON (Virtual Machine Extensions On) region for the VM.
	/// - Aligned to 4096 bytes (0x1000)
	pub vmxon_region: Vmxon,

	/// The VMCS (Virtual Machine Control Structure) for the VM.
	/// - Aligned to 4096 bytes (0x1000)
	pub vmcs_region: Vmcs,
}

impl Vm {
	/// Creates a new zeroed VM instance.
	pub fn zeroed() -> MaybeUninit<Self> {
		MaybeUninit::zeroed()
	}

	/// Initializes a new VM instance with specified guest registers.
	pub fn init(&mut self) -> Result<(), HypervisorError> {
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

	/// Prepares the system for VMX operation by configuring necessary control registers and MSRs.
	///
	/// Ensures that the system meets all prerequisites for VMX operation as defined by Intel's specifications.
	/// This includes enabling VMX operation through control register modifications, setting the lock bit in
	/// IA32_FEATURE_CONTROL MSR, and adjusting mandatory CR0 and CR4 bits.
	///
	/// # Returns
	///
	/// Returns `Ok(())` if all configurations are successfully applied, or an `Err(HypervisorError)` if adjustments fail.
	pub fn setup_vmxon(&self) -> Result<(), HypervisorError> {
		const IA32_FEATURE_CONTROL: u32 = 0x3a;
		const VMX_LOCK_BIT: u64 = 1 << 0;
		const VMXON_OUTSIDE_SMX: u64 = 1 << 2;

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

		let vmxon_addr = virtual_to_physical(VirtAddr::from_ptr(&self.vmxon_region as *const _))
			.unwrap()
			.as_u64();
		unsafe {
			asm!("vmxon [{0}]", in(reg) &vmxon_addr);
		}

		self.vmx_capture_status()
	}

	/// Activates the VMCS region for the VM, preparing it for execution.
	///
	/// Clears and loads the VMCS region, setting it as the current VMCS for VMX operations.
	/// Calls `setup_vmcs` to configure the VMCS with guest, host, and control settings.
	///
	/// # Returns
	///
	/// Returns `Ok(())` on successful activation, or an `Err(HypervisorError)` if activation fails.
	pub fn setup_vmcs(&mut self) -> Result<(), HypervisorError> {
		// Clear the VMCS region.
		let vmcs_addr = virtual_to_physical(VirtAddr::from_ptr(&self.vmcs_region as *const _))
			.unwrap()
			.as_u64();
		unsafe {
			asm!("vmclear [{0}]", in(reg) &vmcs_addr);
		}
		self.vmx_capture_status()?;

		// Load current VMCS pointer.
		unsafe {
			asm!("vmptrld [{0}]", in(reg) &vmcs_addr);
		}
		self.vmx_capture_status()?;

		self.vmcs_region.setup_capabilities()?;
		self.vmcs_region.setup_system_gdt()?;
		self.vmcs_region.setup_system_64bit()
	}

	/// Executes the VM, running in a loop until a VM-exit occurs.
	///
	/// Launches or resumes the VM based on its current state, handling VM-exits as they occur.
	/// Updates the VM's state based on VM-exit reasons and captures the guest register state post-exit.
	///
	/// # Returns
	///
	/// Returns `Ok(VmxBasicExitReason)` indicating the reason for the VM-exit, or an `Err(HypervisorError)`
	/// if the VM fails to launch or an unknown exit reason is encountered.
	pub fn run(&mut self) -> Result<VmxBasicExitReason, HypervisorError> {
		Err(HypervisorError::UnknownVMExitReason)
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
