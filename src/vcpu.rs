use crate::vm::VmId;

pub type VCpuId = usize;

pub struct Cpu;
pub struct CpuConfig;

pub trait VCpu: Sized {
	type VCpuConfig;

	// Required methods
	fn new(vm_id: VmId, vcpu_id: VCpuId, config: Self::VCpuConfig) -> Self;
}
