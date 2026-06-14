use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use hermit_sync::SpinMutex;

use crate::error::HypervisorError;
use crate::uart::Uart;
use crate::vcpu::{Cpu, CpuConfig, VCpu};
use crate::vmx::Ept;

pub type VmId = usize;

/// Configuration describing a virtual machine's guest memory.
#[derive(Debug, Clone, Copy)]
pub struct VmConfig {
	/// Host-virtual base address of the guest memory buffer; its offsets are the
	/// guest-physical addresses.
	pub guest_base: *mut u8,
	/// Size of the guest memory buffer in bytes.
	pub guest_size: usize,
}

/// Nested paging that maps a guest-physical address space onto host-physical
/// memory, independent of the virtualization extension.
///
/// Implemented by the concrete structures of each backend ([`Ept`] for Intel
/// VT-x; AMD-V nested page tables would implement it likewise). The trait is
/// object-safe so a [`Vm`] can own a `Box<dyn NestedPaging>` and the paging
/// scheme can be chosen alongside the vCPU backend.
pub trait NestedPaging {
	/// Returns the nested-paging pointer addressing this guest-physical address
	/// space (the EPT pointer on Intel VT-x, the nested CR3 on AMD-V), ready to
	/// be stored in a vCPU's control structure.
	fn pointer(&self) -> Result<u64, HypervisorError>;
}

/// A virtual machine: the guest's physical address space plus the set of virtual
/// CPUs that execute within it.
///
/// The nested page tables are built once and owned here; every [`Cpu`] of this
/// VM shares them through the nested-paging pointer ([`NestedPaging::pointer`]).
/// This is what lets a single VM manage several vCPUs over one guest-physical
/// address space, regardless of the active backend.
pub struct Vm {
	/// Id of the virtual machine instance.
	id: VmId,
	/// Nested page tables mapping the guest-physical address space, shared by all
	/// vCPUs of this VM.
	paging: Box<dyn NestedPaging>,
	/// Emulated serial port (16550 UART).
	uart: Arc<SpinMutex<Uart>>,
	/// The virtual CPUs managed by this VM.
	cpus: Vec<Cpu>,
}

pub trait VirtualMachine: Sized {
	type Config;

	// Initialize new virtual machine
	fn new(vm_id: VmId, config: Self::Config) -> Result<Self, HypervisorError>;
}

impl VirtualMachine for Vm {
	type Config = VmConfig;

	/// Creates a virtual machine and builds the nested page tables that map its
	/// guest-physical address space. No vCPU exists yet; add them with
	/// [`Vm::create_cpu`].
	fn new(id: VmId, config: Self::Config) -> Result<Self, HypervisorError> {
		// Backend-selection point: today only Intel VT-x EPT is supported. A
		// future implementation can pick the paging scheme here based on the CPU
		// vendor, matching the vCPU backend chosen in `Cpu::new`.
		let paging: Box<dyn NestedPaging> =
			Box::new(Ept::new(config.guest_base, config.guest_size)?);

		Ok(Self {
			id,
			paging,
			uart: Arc::new(SpinMutex::new(Uart::new())),
			cpus: Vec::new(),
		})
	}
}

#[allow(dead_code)]
impl Vm {
	/// Returns the id of this virtual machine.
	pub fn id(&self) -> VmId {
		self.id
	}

	/// Adds a virtual CPU to this VM and returns a mutable reference to it.
	///
	/// The new vCPU shares this VM's nested page tables and starts executing at
	/// `entry_point` with `stack_pointer` as its initial stack. Its id is assigned
	/// in creation order, starting at 0 for the boot CPU.
	pub fn create_cpu(
		&mut self,
		entry_point: u64,
		stack_pointer: u64,
	) -> Result<&mut Cpu, HypervisorError> {
		let config = CpuConfig {
			nested_paging_pointer: self.paging.pointer()?,
			entry_point,
			stack_pointer,
		};
		let vcpu_id = self.cpus.len();
		self.cpus
			.push(Cpu::new(self.id, vcpu_id, self.uart.clone(), config));

		Ok(self.cpus.last_mut().unwrap())
	}

	/// Returns the virtual CPUs managed by this VM.
	pub fn cpus(&self) -> &[Cpu] {
		&self.cpus
	}

	/// Returns the virtual CPUs managed by this VM mutably.
	pub fn cpus_mut(&mut self) -> &mut [Cpu] {
		&mut self.cpus
	}
}
