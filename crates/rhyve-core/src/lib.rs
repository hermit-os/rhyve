#![no_std]

pub mod error;

use crate::error::HypervisorError;

/// Reason a vCPU returned control to the hypervisor, independent of the
/// underlying virtualization extension.
#[derive(Debug, Clone, Copy)]
pub enum ExitReason {
	/// Exit reason is already handled
	Success,
	/// I/O ports access
	IoInstruction(u64),
	/// Shutdown system
	Shutdown,
}

/// A swappable per-vCPU virtualization backend.
///
/// Implemented by the concrete hardware backends (e.g. [`VmxCpu`] for Intel
/// VT-x; an AMD-V backend would implement it likewise). The trait is
/// object-safe so [`Cpu`] can hold a `Box<dyn VcpuBackend>` and the backend can
/// be chosen — even at runtime, based on the CPU vendor — without changing
/// `Cpu` or `Vm`.
pub trait VcpuBackend<T> {
	/// Runs the vCPU until the next VM-exit.
	fn run(&mut self) -> Result<ExitReason, HypervisorError>;

	/// Returns the guest register state captured at the last VM-exit.
	fn guest_registers(&self) -> &T;

	/// Returns the mutable guest register state captured at the last VM-exit.
	fn guest_registers_mut(&mut self) -> &mut T;
}
