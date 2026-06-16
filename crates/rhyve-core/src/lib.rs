#![no_std]

pub mod error;

use hermit_sync::OnceCell;

use crate::error::HypervisorError;

/// Host memory services the hypervisor backends need but cannot provide
/// themselves, because they depend on the environment rhyve runs in.
///
/// The crate user registers an implementation once via [`set_host_memory`];
/// the backends then use the free [`virtual_to_physical`] function. This keeps
/// the translation injectable (and mockable in tests) without threading it
/// through every backend API.
pub trait HostMemory: Sync {
	/// Translates a host-virtual address to a host-physical address, or returns
	/// `None` if the address is not mapped.
	fn virtual_to_physical(&self, vaddr: u64) -> Option<u64>;
}

static HOST_MEMORY: OnceCell<&'static dyn HostMemory> = OnceCell::new();

/// Registers the [`HostMemory`] implementation. Must be called once before any
/// virtual machine is created; further calls have no effect.
pub fn set_host_memory(host: &'static dyn HostMemory) {
	let _ = HOST_MEMORY.set(host);
}

/// Translates a host-virtual to a host-physical address via the registered
/// [`HostMemory`].
///
/// # Panics
///
/// Panics if [`set_host_memory`] has not been called yet.
pub fn virtual_to_physical(vaddr: u64) -> Option<u64> {
	HOST_MEMORY
		.get()
		.expect("rhyve_core::set_host_memory has not been called")
		.virtual_to_physical(vaddr)
}

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
