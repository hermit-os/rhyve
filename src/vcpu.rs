use alloc::boxed::Box;
use core::fmt;

use crate::HypervisorError;
use crate::vm::VmId;
use crate::vmx::{GuestRegisters, VmxBasicExitReason, VmxCpu};

pub type VCpuId = usize;

/// Initial configuration of a virtual CPU.
///
/// Describes the architectural state a [`Cpu`] starts executing with. The
/// guest-physical address space is described once at the VM level; a vCPU only
/// needs the nested-paging pointer that addresses it.
#[derive(Debug, Clone, Copy)]
pub struct CpuConfig {
	/// Nested-paging pointer of the owning VM, shared by all of its vCPUs (the
	/// EPT pointer on Intel VT-x, the nested CR3 on AMD-V).
	pub nested_paging_pointer: u64,
	/// Initial instruction pointer (the guest entry point).
	pub entry_point: u64,
	/// Initial stack pointer.
	pub stack_pointer: u64,
}

/// Reason a vCPU returned control to the hypervisor, independent of the
/// underlying virtualization extension.
///
/// Each backend contributes its own variant; this keeps [`Cpu`] and its callers
/// backend-agnostic while still exposing the native, fully detailed reason.
#[derive(Debug, Clone, Copy)]
pub enum ExitReason {
	/// A VM-exit reported by the Intel VT-x backend.
	Vmx(VmxBasicExitReason),
}

impl fmt::Display for ExitReason {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			ExitReason::Vmx(reason) => write!(f, "{reason}"),
		}
	}
}

/// A swappable per-vCPU virtualization backend.
///
/// Implemented by the concrete hardware backends (e.g. [`VmxCpu`] for Intel
/// VT-x; an AMD-V backend would implement it likewise). The trait is
/// object-safe so [`Cpu`] can hold a `Box<dyn VcpuBackend>` and the backend can
/// be chosen — even at runtime, based on the CPU vendor — without changing
/// `Cpu` or `Vm`.
pub trait VcpuBackend {
	/// Runs the vCPU until the next VM-exit.
	fn run(&mut self) -> Result<ExitReason, HypervisorError>;

	/// Returns the guest register state captured at the last VM-exit.
	fn guest_registers(&self) -> &GuestRegisters;
}

/// A virtual CPU belonging to a virtual machine.
///
/// `Cpu` is the architecture-independent handle that implements [`VCpu`]. The
/// actual virtualization work is delegated to a [`VcpuBackend`] held behind a
/// trait object, so the backend (Intel VT-x today, other extensions later) can
/// be swapped without touching `Cpu` or [`Vm`](crate::vm::Vm). The nested page
/// tables are owned by the VM and shared with every `Cpu` via
/// [`CpuConfig::eptp`].
pub struct Cpu {
	/// Id of the virtual machine this vCPU belongs to.
	vm_id: VmId,
	/// Id of this vCPU within its virtual machine.
	id: VCpuId,
	/// The virtualization backend driving this vCPU.
	backend: Box<dyn VcpuBackend>,
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
	/// Returns `Ok(ExitReason)` indicating the reason for the VM-exit, or an `Err(HypervisorError)`
	/// if the VM fails to launch or an unknown exit reason is encountered.
	fn run(&mut self) -> Result<Self::VCpuExitReasons, HypervisorError>;
}

impl VCpu for Cpu {
	type VCpuConfig = CpuConfig;
	type VCpuExitReasons = ExitReason;

	/// Creates a virtual CPU and its virtualization backend. The backend enables
	/// virtualization, configures the VM control structures with the VM's
	/// nested-paging pointer and seeds the guest registers from `config` (RIP at
	/// the entry point, RSP at the initial stack).
	fn new(vm_id: VmId, vcpu_id: VCpuId, config: Self::VCpuConfig) -> Self {
		// Backend-selection point: today only Intel VT-x is supported. A future
		// implementation can pick the backend here based on the CPU vendor.
		let backend: Box<dyn VcpuBackend> = Box::new(
			VmxCpu::new(
				config.nested_paging_pointer,
				vcpu_id as u64,
				config.entry_point,
				config.stack_pointer,
			)
			.expect("Failed to create the VT-x backend of the vCPU"),
		);

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
