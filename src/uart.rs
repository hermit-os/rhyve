//! A minimal 16550 UART model for the guest's serial port.
//!
//! It is just complete enough to let the guest's serial driver finish its
//! initialization/polling and transmit characters: the line-status register
//! always reports the transmitter as ready, the modem-status register reports
//! the modem as ready, and the writable scratch/modem-control registers retain
//! the last value written so read-back checks succeed.

/// Register offsets relative to the serial port base.
const REG_DATA: u16 = 0; // RBR (read) / THR (write)
const REG_MCR: u16 = 4; // Modem Control Register
const REG_LSR: u16 = 5; // Line Status Register
const REG_MSR: u16 = 6; // Modem Status Register
const REG_SCR: u16 = 7; // Scratch Register

/// Line Status: THRE (bit 5) and TEMT (bit 6) — transmitter holding register and
/// shift register empty, i.e. always ready to accept a byte. DR (bit 0) is clear,
/// so no input is ever reported as pending.
const LSR_TX_READY: u8 = (1 << 5) | (1 << 6);
/// Modem Status: DCD (bit 7), DSR (bit 5) and CTS (bit 4) set — the modem is
/// "connected and ready", which satisfies drivers that gate transmission on it.
const MSR_READY: u8 = (1 << 7) | (1 << 5) | (1 << 4);

/// Emulated state of a 16550 UART.
pub struct Uart {
	/// Last value written to the Modem Control Register.
	mcr: u8,
	/// Last value written to the Scratch Register.
	scr: u8,
}

impl Uart {
	pub fn new() -> Self {
		Self { mcr: 0, scr: 0 }
	}

	/// Handles an `IN` from the register at `offset` (port − base) and returns
	/// the byte the guest reads.
	pub fn read(&self, offset: u16) -> u8 {
		match offset {
			REG_LSR => LSR_TX_READY,
			REG_MSR => MSR_READY,
			REG_MCR => self.mcr,
			REG_SCR => self.scr,
			_ => 0, // RBR (no input) and the remaining registers
		}
	}

	/// Handles an `OUT` of `value` to the register at `offset` (port − base).
	///
	/// Returns `Some(byte)` when the write targets the transmit register, i.e.
	/// the byte the guest wants to send to the console.
	pub fn write(&mut self, offset: u16, value: u8) -> Option<u8> {
		match offset {
			REG_DATA => return Some(value),
			REG_MCR => self.mcr = value,
			REG_SCR => self.scr = value,
			_ => {} // IER/FCR/LCR: accepted but not modelled
		}
		None
	}
}
