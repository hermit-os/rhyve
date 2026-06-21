use alloc::boxed::Box;
use alloc::sync::Arc;

use hermit_sync::SpinMutex;
use rhyve_core::*;

use crate::svm::SvmCpu;
use crate::uart::Uart;
use crate::vm::{VmId, VmMemoryLayout};
use crate::vmx::{GuestRegisters, VmxCpu};
use crate::{HypervisorError, HypervisorExtension, check_supported_cpu};

pub type VCpuId = usize;

/// I/O ports (besides the uhyve `Exit` hypercall) whose write ends the run.
/// `0xf4` is the `isa-debug-exit` port; `0x540` is the uhyve v1 exit port.
const EXIT_PORTS: [u16; 2] = [0xf4, 0x540];

/// Initial configuration of a virtual CPU.
///
/// Describes the architectural state a [`Cpu`] starts executing with. The
/// guest-physical address space is described once at the VM level; a vCPU only
/// needs the nested-paging pointer that addresses it.
#[derive(Debug, Clone, Copy)]
pub struct CpuConfig {
	/// Nested-paging pointer of the owning VM, shared by all of its vCPUs (the
	/// EPT pointer on Intel VT-x, the nested CR3 on AMD-V).
	pub nested_paging_pointer: u64,
	/// Host-physical address of the VM's APIC-access page (for hardware APIC
	/// virtualization).
	pub apic_access_hpa: u64,
	/// Initial instruction pointer (the guest entry point).
	pub entry_point: u64,
	/// Initial stack pointer.
	pub stack_pointer: u64,
}

/// A virtual CPU belonging to a virtual machine.
///
/// `Cpu` is the architecture-independent handle that implements [`VCpu`]. The
/// actual virtualization work is delegated to a [`VcpuBackend`] held behind a
/// trait object, so the backend (Intel VT-x today, other extensions later) can
/// be swapped without touching `Cpu` or [`Vm`](crate::vm::Vm). The nested page
/// tables are owned by the VM and shared with every `Cpu` via
/// [`CpuConfig::eptp`].
pub struct Cpu {
	/// Id of the virtual machine this vCPU belongs to.
	vm_id: VmId,
	/// Id of this vCPU within its virtual machine.
	id: VCpuId,
	/// Emulated serial port (16550 UART).
	uart: Arc<SpinMutex<Uart>>,
	/// Description of the VM layout
	mem: VmMemoryLayout,
	/// The virtualization backend driving this vCPU.
	backend: Box<dyn VcpuBackend<GuestRegisters>>,
}

pub trait VCpu: Sized {
	type VCpuConfig;
	type VCpuExitReasons;

	// Create new Intialization instance
	fn new(
		vm_id: VmId,
		vcpu_id: VCpuId,
		uart: Arc<SpinMutex<Uart>>,
		mem: VmMemoryLayout,
		config: Self::VCpuConfig,
	) -> Self;

	/// Executes the VM, running in a loop until a VM-exit occurs.
	///
	/// Launches or resumes the VM based on its current state, handling VM-exits as they occur.
	/// Updates the VM's state based on VM-exit reasons and captures the guest register state post-exit.
	///
	/// # Returns
	///
	/// Returns `Ok(ExitReason)` indicating the reason for the VM-exit, or an `Err(HypervisorError)`
	/// if the VM fails to launch or an unknown exit reason is encountered.
	fn run(&mut self) -> Result<Self::VCpuExitReasons, HypervisorError>;
}

impl VCpu for Cpu {
	type VCpuConfig = CpuConfig;
	type VCpuExitReasons = ();

	/// Creates a virtual CPU and its virtualization backend. The backend enables
	/// virtualization, configures the VM control structures with the VM's
	/// nested-paging pointer and seeds the guest registers from `config` (RIP at
	/// the entry point, RSP at the initial stack).
	fn new(
		vm_id: VmId,
		vcpu_id: VCpuId,
		uart: Arc<SpinMutex<Uart>>,
		mem: VmMemoryLayout,
		config: Self::VCpuConfig,
	) -> Self {
		// Backend-selection point: pick the virtualization extension the host CPU
		// supports. The paging scheme chosen in `Vm::new` must match (both query
		// the same CPU, so they agree).
		let backend: Box<dyn VcpuBackend<GuestRegisters>> =
			if matches!(check_supported_cpu(), Ok(HypervisorExtension::Svm)) {
				Box::new(
					SvmCpu::new(
						config.nested_paging_pointer,
						config.apic_access_hpa,
						vcpu_id as u64,
						config.entry_point,
						config.stack_pointer,
					)
					.expect("Failed to create the AMD-V backend of the vCPU"),
				)
			} else {
				Box::new(
					VmxCpu::new(
						config.nested_paging_pointer,
						config.apic_access_hpa,
						vcpu_id as u64,
						config.entry_point,
						config.stack_pointer,
					)
					.expect("Failed to create the VT-x backend of the vCPU"),
				)
			};

		Self {
			vm_id,
			id: vcpu_id,
			uart,
			mem,
			backend,
		}
	}

	fn run(&mut self) -> Result<(), HypervisorError> {
		loop {
			let reason = self.backend.run()?;

			match reason {
				ExitReason::IoInstruction(q) => {
					use uhyve_interface::v2::HypercallAddress;
					use uhyve_interface::v2::parameters::*;

					let is_in = (q >> 3) & 1 != 0; // Bit 3: 0 = OUT, 1 = IN
					let port = (q >> 16) as u16; // Bits 16–31: port number

					// A write to an exit port is the guest requesting shutdown; end
					// the run cleanly so the caller's output stream closes.
					if !is_in && EXIT_PORTS.contains(&port) {
						return Ok(());
					}

					// uhyve hypercall interface (https://github.com/hermit-os/uhyve):
					// an OUT's value is either a scalar (exit code / byte) or the
					// guest-physical address of a parameter struct in guest RAM.
					if !is_in && let Ok(hypercall) = HypercallAddress::try_from(port as u64) {
						// uhyve passes the hypercall argument (a scalar value or the
						// guest-physical address of a parameter struct) in RDI; the
						// `out dx, eax` only carries a fixed magic in EAX.
						let data = self.registers().rdi;
						match hypercall {
							HypercallAddress::Exit => return Ok(()),
							HypercallAddress::SerialWriteByte => {
								self.uart.lock().write_buffer(alloc::vec![data as u8]);
							}
							HypercallAddress::SerialWriteBuffer => unsafe {
								let p = &*self.guest_ptr::<SerialWriteBufferParams>(data);
								let buf = core::slice::from_raw_parts(
									self.guest_ptr::<u8>(p.buf.as_u64()),
									p.len as usize,
								);
								self.uart.lock().write_buffer(buf.to_vec());
							},
							HypercallAddress::FileWrite => unsafe {
								let p = &mut *self.guest_ptr::<WriteParams>(data);
								if p.fd == 1 || p.fd == 2 {
									// stdout / stderr → console and the output stream.
									let buf = core::slice::from_raw_parts(
										self.guest_ptr::<u8>(p.buf.as_u64()),
										p.len as usize,
									);
									self.uart.lock().write_buffer(buf.to_vec());
									p.ret = p.len as i64;
								} else {
									p.ret = -1; // host file I/O is not supported
								}
							},
							// Host file I/O is not implemented; report an error so the
							// guest fails gracefully instead of hanging.
							HypercallAddress::FileRead => unsafe {
								(*self.guest_ptr::<ReadParams>(data)).ret = -1;
							},
							HypercallAddress::FileOpen => unsafe {
								(*self.guest_ptr::<OpenParams>(data)).ret = -1;
							},
							HypercallAddress::FileClose => unsafe {
								(*self.guest_ptr::<CloseParams>(data)).ret = -1;
							},
							HypercallAddress::FileLseek => unsafe {
								(*self.guest_ptr::<LseekParams>(data)).offset = -1;
							},
							HypercallAddress::FileUnlink => unsafe {
								(*self.guest_ptr::<UnlinkParams>(data)).ret = -1;
							},
							// No console input and no shared memory.
							_ => {}
						}
						continue;
					}

					// Only the guest's serial port (8 consecutive ports from the base)
					// is emulated; other I/O is swallowed.
					let serial = (crate::SERIAL_BASE..crate::SERIAL_BASE + 8).contains(&port);
					let offset = port.wrapping_sub(crate::SERIAL_BASE);

					let rax = self.registers().rax;
					if is_in {
						let value = if serial {
							self.uart.lock().read(offset)
						} else {
							0
						};
						self.registers_mut().rax = (rax & !0xff) | u64::from(value);
					} else if serial {
						let _ = self.uart.lock().write(offset, rax as u8);
					}
				}
				ExitReason::Shutdown => {
					return Ok(());
				}
				_ => {}
			}

			std::thread::yield_now();
		}
	}
}

#[allow(dead_code)]
impl Cpu {
	/// Returns the id of the virtual machine this vCPU belongs to.
	pub fn vm_id(&self) -> VmId {
		self.vm_id
	}

	/// Returns the id of this vCPU within its virtual machine.
	pub fn id(&self) -> VCpuId {
		self.id
	}

	/// Returns the guest register state captured at the last VM-exit.
	pub fn registers(&self) -> &GuestRegisters {
		self.backend.guest_registers()
	}

	/// Returns the guest register state captured at the last VM-exit.
	pub fn registers_mut(&mut self) -> &mut GuestRegisters {
		self.backend.guest_registers_mut()
	}

	/// Translates a guest-physical address into a raw pointer into the host's
	/// mapping of the guest's RAM (guest-physical `0` is at `mem.start_addr`).
	fn guest_ptr<T>(&self, gpa: u64) -> *mut T {
		(self.mem.start_addr.as_u64() + gpa) as *mut T
	}
}
