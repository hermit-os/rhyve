#![no_std] // don't link the Rust standard library
#![no_main]

#[macro_use]
extern crate log;
extern crate alloc;
extern crate hermit;

mod fdt;
mod uart;
mod vcpu;
mod vm;

use alloc::borrow::ToOwned;
use alloc::vec;
use core::ffi::CStr;
use core::mem::MaybeUninit;
use core::num::NonZero;
use core::ptr::slice_from_raw_parts_mut;

use embedded_io::Read;
use hermit::arch::{BasePageSize, PageSize};
use hermit::fd::AccessPermission;
use hermit::fs::{self, File, create_dir, create_file};
use hermit::scheduler::task::NORMAL_PRIO;
use hermit::scheduler::{join, shutdown, spawn};
use hermit::syscalls::{sys_alloc, sys_get_processor_frequency};
use hermit::time::SystemTime;
use hermit_entry::boot_info::*;
use hermit_entry::elf::{KernelObject, LoadedKernel};
use rhyve_core::error::HypervisorError;
use rhyve_x86::*;
use time::OffsetDateTime;
use x86_64::instructions::interrupts;

use crate::fdt::Fdt;
use crate::vcpu::VCpu;
use crate::vm::{VirtualMachine, Vm, VmConfig};

static GUEST: &[u8] = include_bytes!("../data/x86_64/hello_world");

// I/O port base of the guest's emulated serial port.
pub const SERIAL_BASE: u16 = 0x800;
/// Guest-physical address of the flattened device tree.
pub const FDT_OFFSET: u64 = 0x5000;

fn mount_guest_image() {
	create_dir("/image", AccessPermission::from_bits(0o777).unwrap())
		.expect("Unable to create directory /image");

	// Mount in-memory file
	if create_file(
		"/image/guest",
		GUEST,
		AccessPermission::S_IRUSR
			| AccessPermission::S_IRGRP
			| AccessPermission::S_IROTH
			| AccessPermission::S_IXUSR
			| AccessPermission::S_IXGRP
			| AccessPermission::S_IXOTH,
	)
	.is_err()
	{
		error!("Unable to mount file");
	}
}

fn load_guest_image(
	image: &str,
	guest_slice: &mut [MaybeUninit<u8>],
) -> Result<LoadedKernel, HypervisorError> {
	let meta = fs::metadata(image).map_err(|_| HypervisorError::IoError)?;
	let len = meta.len();
	let mut file = File::open(image).map_err(|_| HypervisorError::IoError)?;

	let mut buffer = vec![0; len];
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
	let guest_size = 256 << 20; // create image with a memory size of 256 MiB
	let guest_ptr = sys_alloc(guest_size, BasePageSize::SIZE as usize) as *mut MaybeUninit<u8>;
	let guest_slice: &mut [MaybeUninit<u8>] =
		unsafe { &mut *slice_from_raw_parts_mut(guest_ptr, guest_size) };

	// load guest
	let loaded_kernel = load_guest_image(image, guest_slice)?;
	debug!("Kernel entry point 0x{:x}", loaded_kernel.entry_point);
	debug!("LoadInfo {:x?}", loaded_kernel.load_info);

	let load_info = loaded_kernel.load_info;
	let duration = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
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
			guest_base: guest_ptr.cast::<u8>(),
			guest_size,
		},
	)?;
	let cpu = vm.create_cpu(loaded_kernel.entry_point, BOOT_STACK_TOP)?;

	cpu.run()?;

	Ok(())
}

extern "C" fn start_hypervisor(path: usize) {
	let image = unsafe { CStr::from_ptr(core::ptr::with_exposed_provenance(path)) };
	let _ = init_hypervisor(image)
		.map_err(|e| error!("Unable to start hypervisor with image {image:?}: {e:?}"));
}

#[unsafe(no_mangle)] // don't mangle the name of this function
pub extern "C" fn runtime_entry(_argc: i32, _argv: *const *const u8, _env: *const *const u8) -> ! {
	info!("Initialize rhyve");

	if let Ok(result) = check_supported_cpu() {
		if result == HypervisorExtension::Svm {
			panic!("AMD-V is currently not supportedt");
		}
	} else {
		panic!("CPU doesn't support any virtualization extensions!")
	}

	// mount guest image
	mount_guest_image();

	debug!("Spawn thread to start the hypervisor");

	let image = c"/image/guest".to_owned();
	let id = unsafe {
		spawn(
			start_hypervisor,
			image.into_raw() as usize,
			NORMAL_PRIO,
			hermit::DEFAULT_STACK_SIZE,
			-1,
		)
	};
	let _ = join(id);

	shutdown(0);
}
