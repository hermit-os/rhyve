#![feature(allocator_api)]
#![feature(ptr_as_uninit)]

#[macro_use]
extern crate log;
extern crate alloc;

use hermit as _;

mod fdt;
mod uart;
mod vcpu;
mod vm;

use std::alloc::*;
use std::ffi::CStr;
use std::fs::{self, File, create_dir};
use std::io::{Read, Write};
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::time::SystemTime;

use hermit_entry::boot_info::*;
use hermit_entry::elf::{KernelObject, LoadedKernel};
use rhyve_core::error::HypervisorError;
use rhyve_x86::*;
use time::OffsetDateTime;
use x86_64::instructions::interrupts;
use x86_64::structures::paging::page::{
	PageSize, Size2MiB as LargePageSize, Size4KiB as BasePageSize,
};

use crate::fdt::Fdt;
use crate::vcpu::VCpu;
use crate::vm::{VirtualMachine, Vm, VmConfig};

static GUEST: &[u8] = include_bytes!("../data/x86_64/hello_world");

// I/O port base of the guest's emulated serial port.
const SERIAL_BASE: u16 = 0x800;
/// Guest-physical address of the flattened device tree.
const FDT_OFFSET: u64 = 0x5000;
/// Initial guest stack pointer (grows down, below the loaded kernel).
const BOOT_STACK_TOP: u64 = 0x70000;

unsafe extern "C" {
	safe fn sys_get_processor_frequency() -> u16;
	safe fn sys_virt_addr_to_phys_addr(virt_addr: usize) -> usize;
}

/// Host-memory services backed by the hermit kernel, handed to `rhyve-core`.
struct HermitHostMemory;

impl rhyve_core::HostMemory for HermitHostMemory {
	fn virtual_to_physical(&self, vaddr: u64) -> Option<u64> {
		Some(sys_virt_addr_to_phys_addr(vaddr as usize) as u64)
	}
}

static HOST_MEMORY: HermitHostMemory = HermitHostMemory;

fn mount_guest_image() {
	create_dir("/image").expect("Unable to create directory /image");

	let mut file = File::create("/image/guest").expect("Unable to create_file");
	file.write_all(GUEST).expect("Unable to write to file");
}

fn load_guest_image(
	image: &str,
	guest_slice: &mut [MaybeUninit<u8>],
) -> Result<LoadedKernel, HypervisorError> {
	let meta = fs::metadata(image).map_err(|_| HypervisorError::IoError)?;
	let len = meta.len();
	let mut file = File::open(image).map_err(|_| HypervisorError::IoError)?;

	let mut buffer = vec![0; len.try_into().unwrap()];
	file.read(&mut buffer)
		.map_err(|_| HypervisorError::IoError)?;

	let elf_kernel = KernelObject::parse(&buffer).map_err(|_| HypervisorError::ParseError)?;
	let kernel_offset = 128 * BasePageSize::SIZE;
	let loaded_kernel = elf_kernel.load_kernel(
		&mut guest_slice[kernel_offset as usize..kernel_offset as usize + elf_kernel.mem_size()],
		kernel_offset,
	);

	Ok(loaded_kernel)
}

fn init_hypervisor(image: &CStr) -> Result<(), HypervisorError> {
	let image = image.to_str().expect("Invalid UTF-8 in application path");

	info!("Using image {image:?}");

	// Create slice for the guest
	let guest_size = 256 << 20; // create VM with a memory size of 256 MiB
	let layout =
		unsafe { Layout::from_size_align_unchecked(guest_size, LargePageSize::SIZE as usize) };
	let guest_slice = unsafe { Global.allocate(layout).unwrap().as_uninit_slice_mut() };
	// `Global.allocate` returns uninitialized memory; the guest expects its RAM
	// (BSS, gaps, stack) to read as zero, so clear the whole buffer before loading.
	guest_slice.fill(MaybeUninit::new(0));

	// load guest
	let loaded_kernel = load_guest_image(image, guest_slice)?;
	debug!("Kernel entry point 0x{:x}", loaded_kernel.entry_point);
	debug!("LoadInfo {:x?}", loaded_kernel.load_info);

	let load_info = loaded_kernel.load_info;
	let duration = SystemTime::now()
		.duration_since(SystemTime::UNIX_EPOCH)
		.expect("Unable to create time sinde UNIX_EPOCH");
	let cpu_freq: u32 = sys_get_processor_frequency().into();
	let boot_info = BootInfo {
		hardware_info: HardwareInfo {
			phys_addr_range: 0..guest_size as u64,
			serial_port_base: SerialPortBase::new(SERIAL_BASE),
			device_tree: Some(NonZero::new(FDT_OFFSET).unwrap()),
		},
		load_info,
		platform_info: PlatformInfo::Uhyve {
			has_pci: false,
			num_cpus: NonZero::new(1).unwrap(),
			cpu_freq: Some(NonZero::new(cpu_freq * 1000).unwrap()),
			boot_time: OffsetDateTime::from_unix_timestamp_nanos(
				duration.as_nanos().try_into().unwrap(),
			)
			.unwrap(),
		},
	};

	let fdt = Fdt::new("uhyve")
		.unwrap()
		.memory(0..guest_size as u64)
		.unwrap()
		.finish()
		.unwrap();

	unsafe {
		guest_slice[FDT_OFFSET as usize..FDT_OFFSET as usize + fdt.len()]
			.assume_init_mut()
			.copy_from_slice(fdt.as_slice());

		let raw_boot_info_ptr =
			&mut guest_slice[BOOT_INFO_OFFSET as usize] as *mut _ as *mut RawBootInfo;
		*raw_boot_info_ptr = RawBootInfo::from(boot_info);
	}

	// Set up the guest's boot GDT and page tables in guest memory.
	init_guest_memory(guest_slice);

	interrupts::disable();

	// Create the VM (building the shared EPT) and add the boot vCPU to it.
	let mut vm = Vm::new(
		0,
		VmConfig {
			guest_base: guest_slice.as_mut_ptr() as *mut u8,
			guest_size,
		},
	)?;
	let cpu = vm.create_cpu(loaded_kernel.entry_point, BOOT_STACK_TOP)?;

	cpu.run()?;

	Ok(())
}

pub fn main() {
	info!("Initialize rhyve");

	match check_supported_cpu() {
		Ok(HypervisorExtension::Vmx) => info!("Using the Intel VT-x (VMX) backend"),
		Ok(HypervisorExtension::Svm) => info!("Using the AMD-V (SVM) backend"),
		Err(_) => panic!("CPU doesn't support any virtualization extensions!"),
	}

	// Provide the host's virtual-to-physical translation to the backend.
	rhyve_core::set_host_memory(&HOST_MEMORY);

	// mount guest image
	mount_guest_image();

	debug!("Start the hypervisor");

	init_hypervisor(c"/image/guest").unwrap();
}
