/// x2APIC timer registers, accessed by the guest via RDMSR/WRMSR. The MSR
/// permission map intercepts every access in this range, so the local-APIC timer
/// is emulated entirely in the WRMSR handler (mirroring the VT-x backend).
pub const IA32_TSC_DEADLINE: u32 = 0x6e0;
pub const IA32_X2APIC_LVT_TIMER: u32 = 0x832;
pub const IA32_X2APIC_INIT_COUNT: u32 = 0x838;

/// LVT timer-mode field (bits 18:17): one-shot, periodic or TSC-deadline.
pub const LVT_TIMER_MODE_MASK: u64 = 0b11 << 17;
pub const LVT_TIMER_MODE_PERIODIC: u64 = 0b01 << 17;
pub const LVT_TIMER_MODE_TSC_DEADLINE: u64 = 0b10 << 17;
/// LVT mask bit (bit 16): when set, the timer interrupt is suppressed.
pub const LVT_MASKED: u64 = 1 << 16;

/// Emulated state of the guest's local-APIC timer (programmed via x2APIC MSRs).
///
/// VT-x and VMD-V APIC virtualization does not model the local-APIC timer,
/// so the guest's writes to the timer registers are captured here and turned
/// into a host deadline that the preemption timer raises a VM-exit on.
#[derive(Default)]
pub struct ApicTimer {
	/// Interrupt vector to inject, taken from the LVT timer register.
	pub vector: u8,
	/// Whether the timer interrupt is currently masked (LVT bit 16).
	pub masked: bool,
	/// Whether the LVT timer is in TSC-deadline mode.
	pub tsc_deadline_mode: bool,
	/// Whether the LVT timer is in periodic mode (re-arm after each expiry).
	pub periodic: bool,
	/// Absolute host-TSC value at which the timer should next fire, if armed.
	pub deadline: Option<u64>,
	/// Period in TSC ticks, used to re-arm a periodic timer after it expires.
	pub period: Option<u64>,
}
