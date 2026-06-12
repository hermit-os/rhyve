//! Extended Page Tables (EPT) for the guest.
//!
//! Unlike an in-place hypervisor that virtualizes the running host with an
//! identity-mapped EPT (guest-physical == host-physical), `rhyve` boots a
//! *separate* guest image that lives in a buffer allocated from the host heap.
//! The guest therefore sees a physical address space starting at 0, and EPT has
//! to translate each guest-physical page to the host-physical page that backs
//! the corresponding offset of the guest buffer.
//!
//! The guest buffer is not guaranteed to be physically contiguous, so the whole
//! range is mapped with 4 KiB EPT leaf entries whose host-physical frame numbers
//! are resolved page by page via [`virtual_to_physical`].
//!
//! Reference: Intel® 64 and IA-32 Architectures Software Developer's Manual,
//! Section 29.3 "The Extended Page Table Mechanism (EPT)".

use alloc::alloc::{Layout, alloc_zeroed};
use alloc::boxed::Box;
use alloc::vec::Vec;

use hermit::arch::{BasePageSize, PageSize};
use hermit::mm::{VirtAddr, virtual_to_physical};

use crate::error::HypervisorError;
use crate::vm::NestedPaging;

/// Number of entries in a paging-structure table.
const ENTRY_COUNT: usize = 512;

/// EPT entry: read access.
const EPT_READ: u64 = 1 << 0;
/// EPT entry: write access.
const EPT_WRITE: u64 = 1 << 1;
/// EPT entry: execute access.
const EPT_EXECUTE: u64 = 1 << 2;
/// EPT leaf entry: write-back (WB) memory type, encoded in bits 5:3.
const EPT_MEMORY_TYPE_WB: u64 = 6 << 3;

/// EPTP: 4-level page-walk length, encoded in bits 5:3 (value 3).
const EPT_WALK_LENGTH_4: u64 = 3 << 3;
/// EPTP: write-back paging-structure memory type.
const EPT_POINTER_MEMORY_TYPE_WB: u64 = 6;

/// A 4 KiB-aligned paging-structure table of 512 64-bit entries.
#[repr(C, align(4096))]
struct Table {
	entries: [u64; ENTRY_COUNT],
}

/// Allocates a zeroed, page-aligned [`Table`] on the heap.
fn alloc_table() -> Result<Box<Table>, HypervisorError> {
	let layout = Layout::new::<Table>();
	// SAFETY: `Table` is a plain array of integers for which an all-zero bit
	// pattern is valid, and the layout is non-zero sized.
	let ptr = unsafe { alloc_zeroed(layout) }.cast::<Table>();
	if ptr.is_null() {
		return Err(HypervisorError::AllocationFailed);
	}
	// SAFETY: `ptr` is a freshly allocated, properly aligned and initialized
	// (zeroed) allocation matching `Layout::new::<Table>()`.
	Ok(unsafe { Box::from_raw(ptr) })
}

/// Resolves the host-physical address backing a host-virtual pointer.
fn host_physical(ptr: *const u8) -> Result<u64, HypervisorError> {
	virtual_to_physical(VirtAddr::from_ptr(ptr))
		.map(|pa| pa.as_u64())
		.ok_or(HypervisorError::AllocationFailed)
}

/// Extended Page Tables describing a guest's physical address space.
///
/// The lower-level tables are referenced by the hardware through host-physical
/// addresses, not by Rust, so the `Box`es are never read after construction.
/// They must nonetheless stay owned here: dropping them would free page tables
/// the processor still walks on every guest memory access.
#[allow(dead_code)]
pub struct Ept {
	pml4: Box<Table>,
	pdpt: Box<Table>,
	pd: Box<Table>,
	/// Leaf page tables; one per 2 MiB of guest-physical memory.
	pts: Vec<Box<Table>>,
}

impl Ept {
	/// Builds EPT mapping guest-physical `[0, size)` onto the host-physical
	/// pages that back `guest_base .. guest_base + size`.
	///
	/// `guest_base` must be page-aligned and `size` a multiple of the base page
	/// size. The whole range has to fit into a single page-directory (1 GiB).
	pub fn new(guest_base: *const u8, size: usize) -> Result<Self, HypervisorError> {
		let page_size = BasePageSize::SIZE as usize;
		assert_eq!(guest_base as usize % page_size, 0, "guest base must be page-aligned");
		assert_eq!(size % page_size, 0, "guest size must be a multiple of the page size");

		let num_pages = size / page_size;
		let num_pts = num_pages.div_ceil(ENTRY_COUNT);
		assert!(num_pts <= ENTRY_COUNT, "guest larger than 1 GiB is not supported");

		let mut pml4 = alloc_table()?;
		let mut pdpt = alloc_table()?;
		let mut pd = alloc_table()?;
		let mut pts = Vec::with_capacity(num_pts);

		// PML4[0] -> PDPT, PDPT[0] -> PD: the whole guest fits in the first 1 GiB.
		pml4.entries[0] = host_physical((&*pdpt as *const Table).cast())? | EPT_READ | EPT_WRITE | EPT_EXECUTE;
		pdpt.entries[0] = host_physical((&*pd as *const Table).cast())? | EPT_READ | EPT_WRITE | EPT_EXECUTE;

		for i in 0..num_pts {
			let mut pt = alloc_table()?;
			for j in 0..ENTRY_COUNT {
				let page_idx = i * ENTRY_COUNT + j;
				if page_idx >= num_pages {
					break;
				}
				let gpa = page_idx * page_size;
				// SAFETY: `gpa < size`, so the resulting pointer stays inside the
				// guest buffer; it is only used to query its physical address.
				let hpa = host_physical(unsafe { guest_base.add(gpa) })?;
				pt.entries[j] = hpa | EPT_READ | EPT_WRITE | EPT_EXECUTE | EPT_MEMORY_TYPE_WB;
			}
			pd.entries[i] = host_physical((&*pt as *const Table).cast())? | EPT_READ | EPT_WRITE | EPT_EXECUTE;
			pts.push(pt);
		}

		Ok(Self { pml4, pdpt, pd, pts })
	}

	/// Returns the EPT pointer (EPTP) value to store in the VMCS, i.e. the
	/// host-physical address of the PML4 table combined with the 4-level
	/// page-walk length and write-back memory type.
	pub fn eptp(&self) -> Result<u64, HypervisorError> {
		let pml4 = host_physical((&*self.pml4 as *const Table).cast())?;
		Ok(pml4 | EPT_WALK_LENGTH_4 | EPT_POINTER_MEMORY_TYPE_WB)
	}
}

impl NestedPaging for Ept {
	fn pointer(&self) -> Result<u64, HypervisorError> {
		self.eptp()
	}
}
