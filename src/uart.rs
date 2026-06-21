//! A minimal 16550 UART model for the guest's serial port.
//!
//! It is just complete enough to let the guest's serial driver finish its
//! initialization/polling and transmit characters: the line-status register
//! always reports the transmitter as ready, the modem-status register reports
//! the modem as ready, and the writable scratch/modem-control registers retain
//! the last value written so read-back checks succeed.

use tokio::sync::mpsc::UnboundedSender;

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

/// State of the ANSI-escape stripper, which spans several transmitted bytes.
#[derive(Clone, Copy)]
enum AnsiState {
	/// Outside any escape sequence.
	Normal,
	/// Saw `ESC`; awaiting the next byte (`[` starts a CSI sequence).
	Esc,
	/// Inside a CSI sequence (`ESC [ … <final byte>`).
	Csi,
}

/// Emulated state of a 16550 UART.
pub struct Uart {
	/// Last value written to the Modem Control Register.
	mcr: u8,
	/// Last value written to the Scratch Register.
	scr: u8,
	/// Channel the guest's transmitted bytes are streamed into, so the run handler
	/// can forward them to the client as they arrive. `None` until a sink is
	/// attached via [`Uart::set_sink`].
	sink: Option<UnboundedSender<Vec<u8>>>,
	/// Running state of the ANSI-escape stripper applied to streamed output.
	ansi: AnsiState,
}

impl Uart {
	pub fn new() -> Self {
		Self {
			mcr: 0,
			scr: 0,
			sink: None,
			ansi: AnsiState::Normal,
		}
	}

	/// Feeds one transmitted byte through the ANSI-escape stripper, returning the
	/// byte to forward to the consumer (or `None` if it is part of an escape
	/// sequence or a non-whitespace control character). This keeps the streamed
	/// output readable in a browser. Newlines, carriage returns and tabs pass
	/// through; other C0 control bytes (e.g. form feed) and `ESC[…<letter>`
	/// sequences are dropped.
	fn ansi_filter(&mut self, byte: u8) -> Option<u8> {
		match self.ansi {
			AnsiState::Normal => match byte {
				0x1b => {
					self.ansi = AnsiState::Esc;
					None
				}
				b'\n' | b'\r' | b'\t' => Some(byte),
				0x00..=0x1f => None, // drop other C0 controls
				_ => Some(byte),
			},
			AnsiState::Esc => {
				self.ansi = if byte == b'[' {
					AnsiState::Csi
				} else {
					AnsiState::Normal
				};
				None
			}
			AnsiState::Csi => {
				// A CSI sequence ends at its final byte (0x40..=0x7E).
				if (0x40..=0x7e).contains(&byte) {
					self.ansi = AnsiState::Normal;
				}
				None
			}
		}
	}

	/// Attaches the channel the guest's transmit bytes are streamed into.
	pub fn set_sink(&mut self, sink: UnboundedSender<Vec<u8>>) {
		self.sink = Some(sink);
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
			REG_DATA => {
				// Strip ANSI escapes/control bytes, then stream the byte to the
				// consumer, if one is attached. The channel is unbounded, so this
				// never blocks the guest.
				let filtered = self.ansi_filter(value);
				if let (Some(out), Some(sink)) = (filtered, &self.sink) {
					let _ = sink.send(alloc::vec![out]);
				}
				return Some(value);
			}
			REG_MCR => self.mcr = value,
			REG_SCR => self.scr = value,
			_ => {} // IER/FCR/LCR: accepted but not modelled
		}
		None
	}

	/// Write a buffer to the uart device
	pub fn write_buffer(&mut self, buf: Vec<u8>) {
		let mut data: Vec<u8> = Vec::new();

		for value in buf {
			// never blocks the guest.
			let filtered = self.ansi_filter(value);
			if let Some(out) = filtered {
				data.push(out);
			}
		}

		if let Some(sink) = &self.sink {
			let _ = sink.send(data);
		}
	}
}
