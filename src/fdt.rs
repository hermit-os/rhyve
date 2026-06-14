use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use vm_fdt::{FdtWriter, FdtWriterNode, FdtWriterResult};

pub struct Fdt {
	writer: FdtWriter,
	root_node: FdtWriterNode,
	bootargs: Option<String>,
}

impl Fdt {
	pub fn new(platform: &str) -> FdtWriterResult<Self> {
		let mut writer = FdtWriter::new()?;

		let root_node = writer.begin_node("")?;
		writer.property_string("compatible", &format!("hermit,{platform}"))?;
		writer.property_u32("#address-cells", 0x2)?;
		writer.property_u32("#size-cells", 0x2)?;

		let bootargs = None;

		Ok(Self {
			writer,
			root_node,
			bootargs,
		})
	}

	pub fn finish(mut self) -> FdtWriterResult<Vec<u8>> {
		let chosen_node = self.writer.begin_node("chosen")?;
		if let Some(bootargs) = &self.bootargs {
			self.writer.property_string("bootargs", bootargs)?;
		}
		self.writer.end_node(chosen_node)?;

		self.writer.end_node(self.root_node)?;

		self.writer.finish()
	}

	#[cfg_attr(all(target_arch = "x86_64", not(target_os = "uefi")), expect(unused))]
	pub fn rsdp(mut self, rsdp: u64) -> FdtWriterResult<Self> {
		let rsdp_node = self.writer.begin_node(&format!("hermit,rsdp@{rsdp:x}"))?;
		self.writer.property_array_u64("reg", &[rsdp, 1])?;
		self.writer.end_node(rsdp_node)?;

		Ok(self)
	}

	pub fn memory(mut self, memory: Range<u64>) -> FdtWriterResult<Self> {
		let memory_node = self
			.writer
			.begin_node(format!("memory@{:x}", memory.start).as_str())?;
		self.writer.property_string("device_type", "memory")?;
		self.writer
			.property_array_u64("reg", &[memory.start, memory.end - memory.start])?;
		self.writer.end_node(memory_node)?;

		Ok(self)
	}
}
