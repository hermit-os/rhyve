use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use hermit_sync::SpinMutex;
use rhyve_x86::error::*;

use crate::uart::Uart;
use crate::vcpu::{Cpu, CpuConfig, VCpu};
use crate::vmx::{Ept, NestedPaging, Page};

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

/// Guest-physical base of the local APIC MMIO (default `IA32_APIC_BASE`).
pub const LAPIC_BASE: u64 = 0xFEE0_0000;
/// Guest-physical base of the I/O APIC MMIO.
pub const IOAPIC_BASE: u64 = 0xFEC0_0000;

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
	/// APIC-access page for hardware APIC virtualization, mapped at [`LAPIC_BASE`]
	/// in the guest. Owned here so it stays alive while the EPT references it.
	#[allow(dead_code)]
	apic_access_page: Box<Page>,
	/// Dummy page backing the (not yet emulated) I/O APIC MMIO, so guest accesses
	/// do not fault. Owned here for the same reason.
	#[allow(dead_code)]
	ioapic_page: Box<Page>,
	/// Host-physical address of the APIC-access page, handed to each vCPU.
	apic_access_hpa: u64,
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
		let mut paging: Box<dyn NestedPaging> =
			Box::new(unsafe { Ept::new(config.guest_base, config.guest_size)? });

		// Back the APIC MMIO regions: the local APIC via a dedicated APIC-access
		// page (used by hardware APIC virtualization), the I/O APIC via a plain
		// page so its accesses do not fault until it is emulated.
		let apic_access_page = Page::zeroed()?;
		let ioapic_page = Page::zeroed()?;
		let apic_access_hpa = apic_access_page.host_physical()?;
		paging.map_mmio(LAPIC_BASE, apic_access_hpa)?;
		paging.map_mmio(IOAPIC_BASE, ioapic_page.host_physical()?)?;

		Ok(Self {
			id,
			paging,
			uart: Arc::new(SpinMutex::new(Uart::new())),
			apic_access_page,
			ioapic_page,
			apic_access_hpa,
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
			apic_access_hpa: self.apic_access_hpa,
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
