use thiserror_no_std::Error;

#[derive(Error, Debug)]
pub enum HypervisorError {
	#[error("IO error")]
	IoError(hermit::errno::Errno),
	#[error("Unable to parse ELF file")]
	ParseError,
	#[error("Intel CPU not found")]
	CPUUnsupported,
	#[error("VMX is not supported")]
	VMXUnsupported,
	#[error("Failed to execute VMCLEAR")]
	VMCLEARFailed,
	#[error("Failed to execute VMREAD")]
	VMREADFailed,
	#[error("Failed to execute VMWRITE")]
	VMWRITEFailed,
	#[error("Failed to execute VMLAUNCH/VMRESUME (VM-instruction error {0})")]
	VMEntryFailed(u64),
	#[error("Failed to allocate memory")]
	AllocationFailed,
	#[error("Hardware support is missing")]
	VmUnsupported,
	#[error("Unknown VM exit basic reason")]
	UnknownVMExitReason,
}
