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
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::time::SystemTime;

use hermit_entry::boot_info::*;
use hermit_entry::elf::{KernelObject, LoadedKernel};
use rhyve_core::error::HypervisorError;
use rhyve_x86::*;
use time::OffsetDateTime;
use tokio::fs::{File, create_dir};
use tokio::io::AsyncWriteExt;
use x86_64::structures::paging::page::{
	PageSize, Size2MiB as LargePageSize, Size4KiB as BasePageSize,
};

use crate::fdt::Fdt;
use crate::vcpu::VCpu;
use crate::vm::{VirtualMachine, Vm, VmConfig};

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

fn load_guest_image(
	image: &str,
	guest_slice: &mut [MaybeUninit<u8>],
) -> Result<LoadedKernel, HypervisorError> {
	let meta = std::fs::metadata(image).map_err(|_| HypervisorError::IoError)?;
	let len = meta.len();
	let mut file = std::fs::File::open(image).map_err(|_| HypervisorError::IoError)?;

	let mut buffer = vec![0; len.try_into().unwrap()];
	std::io::Read::read(&mut file, &mut buffer).map_err(|_| HypervisorError::IoError)?;

	let elf_kernel = KernelObject::parse(&buffer).map_err(|_| HypervisorError::ParseError)?;
	let kernel_offset = 128 * BasePageSize::SIZE;
	let loaded_kernel = elf_kernel.load_kernel(
		&mut guest_slice[kernel_offset as usize..kernel_offset as usize + elf_kernel.mem_size()],
		kernel_offset,
	);

	Ok(loaded_kernel)
}

fn init_hypervisor(
	image: &str,
	output: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
) -> Result<(), HypervisorError> {
	info!("Using image {image:?}");

	// Create slice for the guest
	let guest_size = 512 << 20; // create VM with a memory size of 256 MiB
	let layout =
		unsafe { Layout::from_size_align_unchecked(guest_size, LargePageSize::SIZE as usize) };
	// Keep the allocation around so it can be freed after the run (otherwise every
	// `/run` would leak 512 MiB).
	let allocation = Global.allocate(layout).unwrap();
	let guest_slice = unsafe { allocation.as_uninit_slice_mut() };
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

	// The guest runs with host interrupts left ENABLED: the backend intercepts
	// physical interrupts (SVM `INTERCEPT_INTR` + `clgi`/`stgi`) so every host
	// interrupt becomes a VM-exit the host services itself, instead of leaking
	// into the guest. That keeps the host — timer and network — responsive during
	// the run, so the web service can still answer once the guest is done.
	// Create the VM (building the shared EPT), add the boot vCPU and run it. The
	// VM is dropped at the end of this block (closing the output channel) before
	// the guest memory is freed below.
	let result = {
		let mut vm = Vm::new(
			0,
			VmConfig {
				guest_base: guest_slice.as_mut_ptr() as *mut u8,
				guest_size,
			},
		)?;
		// Stream the guest's serial output to the consumer as it is produced.
		vm.set_output_sink(output);
		let cpu = vm.create_cpu(loaded_kernel.entry_point, BOOT_STACK_TOP)?;
		cpu.run()
	};

	// Release the 256 MiB guest memory now that the run has finished.
	unsafe { Global.deallocate(allocation.cast::<u8>(), layout) };

	result
}

/// Upload page served at `GET /`: pick a file and `PUT` it to
/// `/image/<filename>` (the file's own name) via `fetch`.
const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>rhyve — upload a guest image</title>
  <style>
    body { font-family: sans-serif; max-width: 50rem; margin: 3rem auto; }
    #result { white-space: pre-wrap; margin-top: 1rem; }
    #output {
      margin-top: 1rem;
      height: 24rem;
      overflow: auto;
      white-space: pre;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85rem;
      line-height: 1.3;
      background: #1e1e1e;
      color: #d4d4d4;
      padding: 0.75rem;
      border: 1px solid #444;
      border-radius: 4px;
    }
    #toolbar {
      margin-top: 1rem;
      display: flex;
      gap: 1rem;
      align-items: center;
      font-size: 0.85rem;
    }
    #bytes { color: #888; margin-left: auto; }
  </style>
</head>
<body>
  <h1>rhyve — upload a guest image</h1>
  <p>Select a file; it is stored in <code>/image/&lt;filename&gt;</code>.</p>
  <input type="file" id="file">
  <button id="btn">Upload</button>
  <button id="run" disabled>Run guest</button>
  <div id="result"></div>
  <div id="toolbar" hidden>
    <button id="clear">Clear</button>
    <label><input type="checkbox" id="wrap"> Wrap lines</label>
    <span id="bytes">0 bytes</span>
  </div>
  <pre id="output" hidden></pre>
  <script>
    const result = document.getElementById('result');
    const output = document.getElementById('output');
    const toolbar = document.getElementById('toolbar');
    const bytesEl = document.getElementById('bytes');
    const runBtn = document.getElementById('run');
    let lastName = null;
    let bytes = 0;
    const setBytes = (n) => { bytes = n; bytesEl.textContent = bytes.toLocaleString() + ' bytes'; };
    document.getElementById('clear').onclick = () => { output.textContent = ''; setBytes(0); };
    document.getElementById('wrap').onchange = (e) => {
      output.style.whiteSpace = e.target.checked ? 'pre-wrap' : 'pre';
    };

    document.getElementById('btn').onclick = async () => {
      const input = document.getElementById('file');
      if (!input.files.length) { result.textContent = 'Please select a file.'; return; }
      const file = input.files[0];
      result.textContent = `Uploading ${file.name} (${file.size} bytes)…`;
      try {
        const resp = await fetch('/image/' + encodeURIComponent(file.name), {
          method: 'PUT',
          body: file,
        });
        result.textContent = (resp.ok ? '' : 'Error: ') + await resp.text();
        if (resp.ok) { lastName = file.name; runBtn.disabled = false; }
      } catch (e) {
        result.textContent = 'Error: ' + e;
      }
    };

    runBtn.onclick = async () => {
      if (!lastName) return;
      result.textContent = `Running ${lastName}…`;
      output.hidden = false;
      toolbar.hidden = false;
      output.textContent = '';
      setBytes(0);
      try {
        const resp = await fetch('/run/' + encodeURIComponent(lastName), { method: 'POST' });
        if (!resp.ok) { result.textContent = 'Error: ' + await resp.text(); return; }
        result.textContent = '';
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          // Keep the view pinned to the bottom, unless the user scrolled up.
          const atBottom =
            output.scrollHeight - output.scrollTop - output.clientHeight < 24;
          output.textContent += decoder.decode(value, { stream: true });
          setBytes(bytes + value.length);
          if (atBottom) output.scrollTop = output.scrollHeight;
        }
        output.textContent += decoder.decode();
      } catch (e) {
        result.textContent = 'Error: ' + e;
      }
    };
  </script>
</body>
</html>
"#;

/// Serves the upload page.
async fn index() -> axum::response::Html<&'static str> {
	axum::response::Html(INDEX_HTML)
}

/// Stores an uploaded file under `/image`.
///
/// Handles `PUT /image/{name}` with the file as the raw request body, writing
/// it to `/image/{name}`.
async fn store_image(
	axum::extract::Path(name): axum::extract::Path<String>,
	body: axum::body::Bytes,
) -> Result<String, (axum::http::StatusCode, String)> {
	// The path captures a single segment, but reject anything that could still
	// escape the /image directory.
	if name.is_empty() || name.contains('/') || name.contains("..") {
		return Err((
			axum::http::StatusCode::BAD_REQUEST,
			"invalid file name\n".into(),
		));
	}

	if let Err(e) = create_dir("/image").await
		&& e.kind() != tokio::io::ErrorKind::AlreadyExists
	{
		return Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
	}

	let path = format!("/image/{name}");
	let mut file = File::create(&path)
		.await
		.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
	file.write_all(&body)
		.await
		.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

	let msg = format!("stored {} bytes at {path}\n", body.len());
	Ok(msg)
}

/// Runs the hypervisor on a previously uploaded image, streaming the guest's
/// serial output back as it is produced.
///
/// Handles `POST /run/{name}`, booting `/image/{name}` as the guest. The run is
/// blocking and loops until the guest shuts down, so it runs on a blocking task;
/// its serial output is fed through an unbounded channel and returned as a chunked
/// response body. With a preemptive host scheduler the runtime thread is preempted
/// off the guest periodically and drains the channel, so the client receives the
/// output incrementally; the body ends when the guest finishes (senders dropped).
async fn run_guest(
	axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<axum::response::Response, (axum::http::StatusCode, String)> {
	use tokio_stream::StreamExt;
	use tokio_stream::wrappers::UnboundedReceiverStream;

	if name.is_empty() || name.contains('/') || name.contains("..") {
		return Err((
			axum::http::StatusCode::BAD_REQUEST,
			"invalid file name\n".into(),
		));
	}

	let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
	let stream = UnboundedReceiverStream::new(rx).map(Ok::<Vec<u8>, std::io::Error>);
	let body = axum::body::Body::from_stream(stream);

	let path = format!("/image/{name}");
	tokio::task::spawn_blocking(move || {
		if let Err(e) = init_hypervisor(&path, tx.clone()) {
			let _ = tx.send(format!("\n[run failed: {e:?}]\n").into_bytes());
		}
		// Dropping the last sender closes the channel and ends the response.
	});

	let response = axum::response::Response::builder()
		.header(
			axum::http::header::CONTENT_TYPE,
			"text/plain; charset=utf-8",
		)
		.body(body)
		.expect("failed to build the streaming response");
	Ok(response)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
	println!("Initialize rhyve");

	simple_logger::init_with_level(log::Level::Info).unwrap();

	match check_supported_cpu() {
		Ok(HypervisorExtension::Vmx) => println!("Using the Intel VT-x (VMX) backend"),
		Ok(HypervisorExtension::Svm) => println!("Using the AMD-V (SVM) backend"),
		Err(_) => panic!("CPU doesn't support any virtualization extensions!"),
	}

	// Provide the host's virtual-to-physical translation to the backend.
	rhyve_core::set_host_memory(&HOST_MEMORY);

	// Web service: an upload page at `/`, `PUT /image/{name}` to store an image
	// and `POST /run/{name}` to boot it as the guest.
	let app = axum::Router::new()
		.route("/", axum::routing::get(index))
		.route("/image/{name}", axum::routing::put(store_image))
		.route("/run/{name}", axum::routing::post(run_guest))
		.layer(axum::extract::DefaultBodyLimit::max(256 * 1024 * 1024));

	let addr = "0.0.0.0:9975";
	let listener = tokio::net::TcpListener::bind(addr)
		.await
		.expect("failed to bind the upload service");
	println!("rhyve upload service listening on http://{addr}/image/<name>");
	axum::serve(listener, app)
		.await
		.expect("upload service failed");
}
