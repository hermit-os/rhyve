use bitfield::BitMut;
use hermit::arch::{BasePageSize, PageSize};
use x86_64::registers::model_specific::Msr;

use crate::vmx::IA32_VMX_BASIC;

/// A representation of the VMXON region in memory.
///
/// The VMXON region is essential for enabling VMX operations on the CPU.
/// This structure offers methods for setting up the VMXON region, enabling VMX operations,
/// and performing related tasks.
///
/// Reference: Intel® 64 and IA-32 Architectures Software Developer's Manual: 25.11.5 VMXON Region
#[repr(C, align(4096))]
pub struct Vmxon {
	/// Revision ID required for VMXON.
	pub revision_id: u32,

	/// Data array constituting the rest of the VMXON region.
	pub data: [u8; BasePageSize::SIZE as usize - 4],
}

impl Vmxon {
	/// Initializes the VMXON region.
	pub fn init(&mut self) {
		let msr = Msr::new(IA32_VMX_BASIC);
		self.revision_id = unsafe { msr.read() as u32 };
		self.revision_id.set_bit(31, false);
	}
}
