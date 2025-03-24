#![no_std] // don't link the Rust standard library
#![no_main]

#[macro_use]
extern crate log;
extern crate alloc;
extern crate hermit;

mod error;
mod intel;

use alloc::vec;
use core::mem::MaybeUninit;
use core::num::NonZero;
use core::ptr::slice_from_raw_parts_mut;

use hermit::arch::{BasePageSize, PageSize};
use hermit::fd::AccessPermission;
use hermit::fs::{self, File, create_dir, create_file};
use hermit::io::Read;
use hermit::scheduler::task::NORMAL_PRIO;
use hermit::scheduler::{join, shutdown, spawn};
use hermit::syscalls::{sys_alloc, sys_get_processor_frequency};
use hermit::time::SystemTime;
use hermit_entry::boot_info::*;
use hermit_entry::elf::{KernelObject, LoadedKernel};
use time::OffsetDateTime;
use x86_64::instructions::interrupts;

static GUEST: &[u8] = include_bytes!("../data/x86_64/hello_world");

pub const BOOT_PML4: u64 = 0x10000;
pub const BOOT_INFO_OFFSET: u64 = 0x9000;
pub const BOOT_GDT_NULL: usize = 0;
pub const BOOT_GDT_CODE: usize = 1;
pub const BOOT_GDT_DATA: usize = 2;
pub const BOOT_GDT_MAX: usize = 3;
pub const GDT_KERNEL_CODE: u16 = 1;
pub const GDT_KERNEL_DATA: u16 = 2;
pub const GDT_OFFSET: u64 = 0x1000;
pub const EFER_SCE: u64 = 1; /* System Call Extensions */
pub const EFER_LME: u64 = 1 << 8; /* Long mode enable */
pub const EFER_LMA: u64 = 1 << 10; /* Long mode active (read-only) */
pub const EFER_NXE: u64 = 1 << 11; /* PTE No-Execute bit enable */

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

#[derive(Debug, PartialEq)]
pub enum LoaderError {
	IoError(i32),
	ParseError,
	InvalidElfFile,
}

fn load_guest_image(guest_slice: &mut [MaybeUninit<u8>]) -> Result<LoadedKernel, LoaderError> {
	let app = "/image/guest";
	let meta = fs::metadata(app)
		.map_err(|e| LoaderError::IoError(num::ToPrimitive::to_i32(&e).unwrap()))?;
	let len = meta.len();
	let mut file =
		File::open(app).map_err(|e| LoaderError::IoError(num::ToPrimitive::to_i32(&e).unwrap()))?;

	let mut buffer = vec![0; len];
	file.read(&mut buffer)
		.map_err(|e| LoaderError::IoError(num::ToPrimitive::to_i32(&e).unwrap()))?;

	let elf_kernel = KernelObject::parse(&buffer).map_err(|_| LoaderError::ParseError)?;
	let kernel_offset = 128 * BasePageSize::SIZE;
	let loaded_kernel = elf_kernel.load_kernel(
		&mut guest_slice[kernel_offset as usize..kernel_offset as usize + elf_kernel.mem_size()],
		kernel_offset,
	);

	Ok(loaded_kernel)
}

extern "C" fn start_hypervisor(_: usize) {
	// check if we are running on a Intel CPU with VMX support
	intel::check_supported_cpu().unwrap();

	// Create slice for the guest
	let guest_size = 32768 * BasePageSize::SIZE as usize;
	let guest_ptr = sys_alloc(guest_size, BasePageSize::SIZE as usize) as *mut MaybeUninit<u8>;
	let guest_slice: &mut [MaybeUninit<u8>] =
		unsafe { &mut *slice_from_raw_parts_mut(guest_ptr, guest_size) };

	mount_guest_image();

	// load guest
	let loaded_kernel = load_guest_image(guest_slice).unwrap();
	debug!("Kernel entry point 0x{:x}", loaded_kernel.entry_point);
	debug!("LoadInfo {:x?}", loaded_kernel.load_info);

	let load_info = loaded_kernel.load_info;
	let duration = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
	let cpu_freq: u32 = sys_get_processor_frequency().into();
	let boot_info = BootInfo {
		hardware_info: HardwareInfo {
			phys_addr_range: 0..guest_size as u64,
			serial_port_base: SerialPortBase::new(0x800),
			device_tree: None,
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

	unsafe {
		let raw_boot_info_ptr =
			&mut guest_slice[BOOT_INFO_OFFSET as usize] as *mut _ as *mut RawBootInfo;
		*raw_boot_info_ptr = RawBootInfo::from(boot_info);
	}

	interrupts::disable();

	let mut vm = unsafe { intel::Vm::zeroed().assume_init() };
	match vm.init() {
		Ok(_) => debug!("VM initialized"),
		Err(e) => panic!("Failed to initialize VM: {:?}", e),
	}

	// initialize VMX
	vm.setup_vmxon().unwrap();
	// initialize VMCS
	vm.setup_vmcs().unwrap();

	debug!("VMCS Dump: {:#x?}", vm.vmcs_region);

	interrupts::enable();

	loop {
		if let Ok(basic_exit_reason) = vm.run() {
			info!("eixt_reason {:?}", basic_exit_reason);
		} else {
			error!("Failed to run the VM");
			break;
		}
	}
}

#[unsafe(no_mangle)] // don't mangle the name of this function
pub extern "C" fn runtime_entry(_argc: i32, _argv: *const *const u8, _env: *const *const u8) -> ! {
	info!("Initialize rhyve");

	let id = unsafe {
		spawn(
			start_hypervisor,
			0,
			NORMAL_PRIO,
			hermit::DEFAULT_STACK_SIZE,
			-1,
		)
	};
	let _ = join(id);

	shutdown(0);
}
