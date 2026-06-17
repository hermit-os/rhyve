//! Nested Page Tables (NPT) for the guest.
//!
//! The AMD-V counterpart of the [`Ept`](crate::vmx::Ept) used by the VT-x
//! backend. Like EPT, NPT maps the guest-physical address space onto the
//! host-physical pages that back `rhyve`'s guest buffer; unlike EPT, its entries
//! use the *ordinary* x86-64 paging-structure format, so the same `PRESENT`,
//! `WRITE` and `USER` bits a normal page table would use apply here. The nested
//! CR3 (`nCR3`) the processor walks is simply the host-physical address of the
//! top-level table.
//!
//! The guest buffer is not guaranteed to be physically contiguous, so the whole
//! range is mapped with 4 KiB leaf entries whose host-physical frame numbers are
//! resolved page by page via [`virtual_to_physical`].
//!
//! Reference: AMD64 Architecture Programmer's Manual, Volume 2, Section 15.25
//! "Nested Paging".

use alloc::alloc::{Layout, alloc_zeroed};
use alloc::boxed::Box;
use alloc::vec::Vec;

use rhyve_core::error::HypervisorError;
use x86_64::addr::VirtAddr;
use x86_64::structures::paging::page::{PageSize, Size4KiB as BasePageSize};

use crate::virtual_to_physical;
use crate::vmx::NestedPaging;

/// Number of entries in a paging-structure table.
const ENTRY_COUNT: usize = 512;

/// Mask selecting the host-physical frame address out of a paging entry.
const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

/// Paging entry: present.
const NPT_PRESENT: u64 = 1 << 0;
/// Paging entry: writable.
const NPT_WRITE: u64 = 1 << 1;
/// Paging entry: user-accessible. Nested page tables are walked as if from user
/// mode, so every entry must allow user access for the guest to use it.
const NPT_USER: u64 = 1 << 2;
/// Leaf entry: accessed. Set up front so the processor never has to update it.
const NPT_ACCESSED: u64 = 1 << 5;
/// Leaf entry: dirty. Set up front for the same reason.
const NPT_DIRTY: u64 = 1 << 6;

/// Flags shared by every non-leaf entry (pointing at the next-level table).
const NPT_TABLE: u64 = NPT_PRESENT | NPT_WRITE | NPT_USER;
/// Flags shared by every 4 KiB leaf entry (mapping guest RAM or MMIO).
const NPT_LEAF: u64 = NPT_PRESENT | NPT_WRITE | NPT_USER | NPT_ACCESSED | NPT_DIRTY;

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

/// Nested Page Tables describing a guest's physical address space.
///
/// The lower-level tables are referenced by the hardware through host-physical
/// addresses, not by Rust, so the `Box`es are never read after construction.
/// They must nonetheless stay owned here: dropping them would free page tables
/// the processor still walks on every guest memory access.
#[allow(dead_code)]
pub struct Npt {
	pml4: Box<Table>,
	pdpt: Box<Table>,
	pd: Box<Table>,
	/// Leaf page tables; one per 2 MiB of guest-physical memory.
	pts: Vec<Box<Table>>,
	/// Tables allocated on demand by [`Npt::map_page`] (e.g. for MMIO regions
	/// such as the APIC pages), paired with their host-physical address.
	extra: Vec<(u64, Box<Table>)>,
}

impl Npt {
	/// # Safety
	///
	/// Builds NPT mapping guest-physical `[0, size)` onto the host-physical pages
	/// that back `guest_base .. guest_base + size`.
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
		pml4.entries[0] = host_physical((&*pdpt as *const Table).cast())? | NPT_TABLE;
		pdpt.entries[0] = host_physical((&*pd as *const Table).cast())? | NPT_TABLE;

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
				pt.entries[j] = hpa | NPT_LEAF;
			}
			pd.entries[i] = host_physical((&*pt as *const Table).cast())? | NPT_TABLE;
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
		if *entry & NPT_PRESENT != 0 {
			let child_hpa = *entry & ADDR_MASK;
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
		*entry = hpa | NPT_TABLE;
		self.extra.push((hpa, table));
		Ok(ptr)
	}

	/// Maps a single 4 KiB guest-physical page `gpa` to the host-physical page
	/// `hpa`, allocating the intermediate tables as needed.
	///
	/// Intended for MMIO regions (e.g. the APIC pages) that lie *outside* the
	/// contiguous guest RAM mapped by [`Npt::new`]: `gpa` must be below 512 GiB
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
			(*pt).entries[i1] = hpa | NPT_LEAF;
		}
		Ok(())
	}

	/// Returns the nested CR3 (`nCR3`) value to store in the VMCB: the
	/// host-physical address of the top-level (PML4) table. Unlike the EPT
	/// pointer, no walk-length or memory-type bits are encoded.
	pub fn ncr3(&self) -> Result<u64, HypervisorError> {
		host_physical((&*self.pml4 as *const Table).cast())
	}
}

impl NestedPaging for Npt {
	fn pointer(&self) -> Result<u64, HypervisorError> {
		self.ncr3()
	}

	fn map_mmio(&mut self, gpa: u64, hpa: u64) -> Result<(), HypervisorError> {
		self.map_page(gpa, hpa)
	}
}
