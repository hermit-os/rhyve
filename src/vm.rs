use alloc::vec::Vec;

use crate::error::HypervisorError;
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

/// A virtual machine: the guest's physical address space plus the set of virtual
/// CPUs that execute within it.
///
/// The Extended Page Tables are built once and owned here; every [`Cpu`] of this
/// VM shares them through the EPT pointer ([`Ept::eptp`]). This is what lets a
/// single VM manage several vCPUs over one guest-physical address space.
pub struct Vm {
	/// Id of the virtual machine instance.
	id: VmId,
	/// Extended Page Tables mapping the guest-physical address space, shared by
	/// all vCPUs of this VM.
	ept: Ept,
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

	/// Creates a virtual machine and builds the Extended Page Tables that map its
	/// guest-physical address space. No vCPU exists yet; add them with
	/// [`Vm::create_cpu`].
	fn new(id: VmId, config: Self::Config) -> Result<Self, HypervisorError> {
		let ept = Ept::new(config.guest_base, config.guest_size)?;

		Ok(Self {
			id,
			ept,
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
	/// The new vCPU shares this VM's EPT and starts executing at `entry_point`
	/// with `stack_pointer` as its initial stack. Its id is assigned in creation
	/// order, starting at 0 for the boot CPU.
	pub fn create_cpu(
		&mut self,
		entry_point: u64,
		stack_pointer: u64,
	) -> Result<&mut Cpu, HypervisorError> {
		let config = CpuConfig {
			eptp: self.ept.eptp()?,
			entry_point,
			stack_pointer,
		};
		let vcpu_id = self.cpus.len();
		self.cpus.push(Cpu::new(self.id, vcpu_id, config));

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
