use crate::error::HypervisorError;

pub type VmId = usize;

pub trait VirtualMachine: Sized {
	// Required methods
	fn new(vm_id: VmId) -> Result<Self, HypervisorError>;
}
