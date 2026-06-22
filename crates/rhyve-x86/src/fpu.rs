//! Save/restore of the extended FP register file (x87/SSE/AVX/...) across a
//! guest run.
//!
//! Neither `VMRUN` nor `VMLAUNCH`/`VMRESUME` preserves the FP register file, and
//! the guest kernel switches its threads' FP state lazily (a `#NM` fault on the
//! first FP use after a context switch saves the previous thread's state and
//! loads the new one's). The live FP registers therefore belong to whichever
//! guest thread last touched them — and stay live across a VM-exit. If the host
//! code running between exits clobbered them, a later lazy `#NM` save would write
//! that garbage into the preempted thread's save area, corrupting its results
//! (observed as the parallel-Pi assertion in `rusty_demo` failing once the timer
//! started preempting compute threads).
//!
//! The backends guard against this by [saving](FpuState::save) the guest's FP
//! state right after each exit and [restoring](FpuState::restore) it just before
//! the next entry, doing the same for the host's own FP state around that window
//! so neither side corrupts the other.

use core::arch::asm;

/// A 64-byte-aligned `XSAVE` area. 4 KiB comfortably covers every XCR0 component
/// a guest is likely to enable (x87, SSE, AVX, and well beyond).
#[repr(C, align(64))]
pub struct FpuState {
	data: [u8; 4096],
}

impl FpuState {
	/// A scratch area for capturing whatever state is currently live; its
	/// contents are always overwritten by [`save`](Self::save) before any
	/// matching [`restore`](Self::restore), so they need no initialization.
	pub fn scratch() -> Self {
		Self { data: [0; 4096] }
	}

	/// A clean initial guest FP state. An all-zero `XSAVE` header (`XSTATE_BV`,
	/// at offset 512, is 0) makes `XRSTOR` set every state component to its init
	/// value; the legacy `MXCSR` field (offset 24) is seeded with its `0x1F80`
	/// reset value so `XRSTOR` does not `#GP` on its reserved-bit check.
	pub fn init_guest() -> Self {
		let mut state = Self { data: [0; 4096] };
		state.data[24..28].copy_from_slice(&0x1F80u32.to_le_bytes());
		state
	}

	/// Saves the live extended FP state (every XCR0-enabled component) into this
	/// area with `XSAVE`.
	#[inline(always)]
	pub fn save(&mut self) {
		// EDX:EAX is the requested-feature bitmap; all-ones requests every
		// component currently enabled in XCR0.
		unsafe {
			asm!(
				"xsave64 [{ptr}]",
				ptr = in(reg) self.data.as_mut_ptr(),
				in("eax") u32::MAX,
				in("edx") u32::MAX,
				options(nostack),
			);
		}
	}

	/// Restores the live extended FP state from this area with `XRSTOR`.
	#[inline(always)]
	pub fn restore(&self) {
		unsafe {
			asm!(
				"xrstor64 [{ptr}]",
				ptr = in(reg) self.data.as_ptr(),
				in("eax") u32::MAX,
				in("edx") u32::MAX,
				options(nostack, readonly),
			);
		}
	}
}
