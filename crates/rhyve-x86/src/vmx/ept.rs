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

	/// Maps a single 4 KiB guest-physical page to a host-physical page, e.g. to
	/// back an MMIO region such as the APIC pages.
	fn map_mmio(&mut self, gpa: u64, hpa: u64) -> Result<(), HypervisorError>;
}

/// A 4 KiB-aligned paging-structure table of 512 64-bit entries.
#[repr(C, align(4096))]
struct Table {
	entries: [u64; ENTRY_COUNT],
}

/// A 4 KiB-aligned page, e.g. a backing page for an MMIO region (the APIC
/// pages). Boxed so it keeps a stable host-physical address.
#[repr(C, align(4096))]
pub struct Page {
	_data: [u8; BasePageSize::SIZE as usize],
}

impl Page {
	/// Allocates a zeroed, page-aligned page on the heap.
	pub fn zeroed() -> Result<Box<Self>, HypervisorError> {
		let layout = Layout::new::<Page>();
		// SAFETY: an all-zero byte pattern is valid for `Page` and the layout is
		// non-zero sized.
		let ptr = unsafe { alloc_zeroed(layout) }.cast::<Page>();
		if ptr.is_null() {
			return Err(HypervisorError::AllocationFailed);
		}
		// SAFETY: freshly allocated, aligned, zero-initialized allocation matching
		// `Layout::new::<Page>()`.
		Ok(unsafe { Box::from_raw(ptr) })
	}

	/// Returns the host-physical address backing this page.
	pub fn host_physical(&self) -> Result<u64, HypervisorError> {
		host_physical((self as *const Page).cast())
	}
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
	/// Tables allocated on demand by [`Ept::map_page`] (e.g. for MMIO regions
	/// such as the APIC pages), paired with their host-physical address.
	extra: Vec<(u64, Box<Table>)>,
}

impl Ept {
	/// # Safety
	///
	/// Builds EPT mapping guest-physical `[0, size)` onto the host-physical
	/// pages that back `guest_base .. guest_base + size`.
	///
	/// `guest_base` must be page-aligned and `size` a multiple of the base page
	/// size. The whole range has to fit into a single page-directory (1 GiB).
	pub unsafe fn new(guest_base: *const u8, size: usize) -> Result<Self, HypervisorError> {
		let page_size = BasePageSize::SIZE as usize;
		assert_eq!(
			guest_base as usize % page_size,
			0,
			"guest base must be page-aligned"
		);
		assert_eq!(
			size % page_size,
			0,
			"guest size must be a multiple of the page size"
		);

		let num_pages = size / page_size;
		let num_pts = num_pages.div_ceil(ENTRY_COUNT);
		assert!(
			num_pts <= ENTRY_COUNT,
			"guest larger than 1 GiB is not supported"
		);

		let mut pml4 = alloc_table()?;
		let mut pdpt = alloc_table()?;
		let mut pd = alloc_table()?;
		let mut pts = Vec::with_capacity(num_pts);

		// PML4[0] -> PDPT, PDPT[0] -> PD: the whole guest fits in the first 1 GiB.
		pml4.entries[0] =
			host_physical((&*pdpt as *const Table).cast())? | EPT_READ | EPT_WRITE | EPT_EXECUTE;
		pdpt.entries[0] =
			host_physical((&*pd as *const Table).cast())? | EPT_READ | EPT_WRITE | EPT_EXECUTE;

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
			pd.entries[i] =
				host_physical((&*pt as *const Table).cast())? | EPT_READ | EPT_WRITE | EPT_EXECUTE;
			pts.push(pt);
		}

		Ok(Self {
			pml4,
			pdpt,
			pd,
			pts,
			extra: Vec::new(),
		})
	}

	/// Returns the child table referenced by `parent.entries[idx]`, allocating it
	/// if the entry is not present yet. Used to build paths for MMIO mappings.
	fn child_or_alloc(
		&mut self,
		parent: *mut Table,
		idx: usize,
	) -> Result<*mut Table, HypervisorError> {
		let entry = unsafe { &mut (*parent).entries[idx] };
		if *entry & EPT_READ != 0 {
			let child_hpa = *entry & 0x000F_FFFF_FFFF_F000;
			for (hpa, table) in &self.extra {
				if *hpa == child_hpa {
					return Ok((table.as_ref() as *const Table).cast_mut());
				}
			}
			return Err(HypervisorError::AllocationFailed);
		}

		let table = alloc_table()?;
		let ptr = (table.as_ref() as *const Table).cast_mut();
		let hpa = host_physical(ptr.cast())?;
		*entry = hpa | EPT_READ | EPT_WRITE | EPT_EXECUTE;
		self.extra.push((hpa, table));
		Ok(ptr)
	}

	/// Maps a single 4 KiB guest-physical page `gpa` to the host-physical page
	/// `hpa`, allocating the intermediate tables as needed.
	///
	/// Intended for MMIO regions (e.g. the APIC pages) that lie *outside* the
	/// contiguous guest RAM mapped by [`Ept::new`]: `gpa` must be below 512 GiB
	/// and not within the first 1 GiB (whose PD is owned separately).
	pub fn map_page(&mut self, gpa: u64, hpa: u64) -> Result<(), HypervisorError> {
		assert_eq!(
			(gpa >> 39) & 0x1ff,
			0,
			"guest-physical address must be below 512 GiB"
		);
		assert_ne!(
			(gpa >> 30) & 0x1ff,
			0,
			"map_page is for addresses outside the first 1 GiB"
		);

		// pml4[0] -> pdpt already exists; start the walk at the PDPT.
		let pdpt = (self.pdpt.as_ref() as *const Table).cast_mut();
		let i3 = ((gpa >> 30) & 0x1ff) as usize;
		let pd = self.child_or_alloc(pdpt, i3)?;
		let i2 = ((gpa >> 21) & 0x1ff) as usize;
		let pt = self.child_or_alloc(pd, i2)?;
		let i1 = ((gpa >> 12) & 0x1ff) as usize;
		unsafe {
			(*pt).entries[i1] = hpa | EPT_READ | EPT_WRITE | EPT_EXECUTE | EPT_MEMORY_TYPE_WB;
		}
		Ok(())
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

	fn map_mmio(&mut self, gpa: u64, hpa: u64) -> Result<(), HypervisorError> {
		self.map_page(gpa, hpa)
	}
}
