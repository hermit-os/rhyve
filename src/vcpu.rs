use crate::HypervisorError;
use crate::vm::VmId;
use crate::vmx::{GuestRegisters, VmxBasicExitReason, VmxCpu};

pub type VCpuId = usize;

/// Initial configuration of a virtual CPU.
///
/// Describes the architectural state a [`Cpu`] starts executing with. The
/// guest-physical address space is described once at the VM level; a vCPU only
/// needs the EPT pointer that addresses it.
#[derive(Debug, Clone, Copy)]
pub struct CpuConfig {
	/// EPT pointer (EPTP) of the owning VM, shared by all of its vCPUs.
	pub eptp: u64,
	/// Initial instruction pointer (the guest entry point).
	pub entry_point: u64,
	/// Initial stack pointer.
	pub stack_pointer: u64,
}

/// A virtual CPU belonging to a virtual machine.
///
/// `Cpu` is the architecture-independent handle that implements [`VCpu`]. The
/// Intel VT-x backend [`VmxCpu`] carries the per-CPU VMXON region, VMCS and guest
/// registers; `Cpu` owns that backend and drives it. The EPT is owned by the VM
/// and shared with every `Cpu` via [`CpuConfig::eptp`].
pub struct Cpu {
	/// Id of the virtual machine this vCPU belongs to.
	vm_id: VmId,
	/// Id of this vCPU within its virtual machine.
	id: VCpuId,
	/// The VT-x backend holding the VMXON region, VMCS and guest register state.
	backend: VmxCpu,
}

pub trait VCpu: Sized {
	type VCpuConfig;
	type VCpuExitReasons;

	// Create new Intialization instance
	fn new(vm_id: VmId, vcpu_id: VCpuId, config: Self::VCpuConfig) -> Self;

	/// Executes the VM, running in a loop until a VM-exit occurs.
	///
	/// Launches or resumes the VM based on its current state, handling VM-exits as they occur.
	/// Updates the VM's state based on VM-exit reasons and captures the guest register state post-exit.
	///
	/// # Returns
	///
	/// Returns `Ok(VmxBasicExitReason)` indicating the reason for the VM-exit, or an `Err(HypervisorError)`
	/// if the VM fails to launch or an unknown exit reason is encountered.
	fn run(&mut self) -> Result<Self::VCpuExitReasons, HypervisorError>;
}

impl VCpu for Cpu {
	type VCpuConfig = CpuConfig;
	type VCpuExitReasons = VmxBasicExitReason;

	/// Creates a virtual CPU and its VT-x backend. The backend enables VMX,
	/// configures the VMCS with the VM's EPT pointer and seeds the guest registers
	/// from `config` (RIP at the entry point, RSP at the initial stack).
	fn new(vm_id: VmId, vcpu_id: VCpuId, config: Self::VCpuConfig) -> Self {
		let backend = VmxCpu::new(
			config.eptp,
			vcpu_id as u64,
			config.entry_point,
			config.stack_pointer,
		)
		.expect("Failed to create the VT-x backend of the vCPU");

		Self {
			vm_id,
			id: vcpu_id,
			backend,
		}
	}

	fn run(&mut self) -> Result<Self::VCpuExitReasons, HypervisorError> {
		self.backend.run()
	}
}

#[allow(dead_code)]
impl Cpu {
	/// Returns the id of the virtual machine this vCPU belongs to.
	pub fn vm_id(&self) -> VmId {
		self.vm_id
	}

	/// Returns the id of this vCPU within its virtual machine.
	pub fn id(&self) -> VCpuId {
		self.id
	}

	/// Returns the guest register state captured at the last VM-exit.
	pub fn registers(&self) -> &GuestRegisters {
		self.backend.guest_registers()
	}
}
