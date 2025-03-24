use thiserror_no_std::Error;

#[derive(Error, Debug)]
pub enum HypervisorError {
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
	#[error("Unknown VM exit basic reason")]
	UnknownVMExitReason,
}
