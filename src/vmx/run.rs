//! Guest register state and the low-level VM-entry trampoline.
//!
//! The assembly routine swaps the host and guest general-purpose registers
//! around a `VMLAUNCH`/`VMRESUME` instruction and regains control at the
//! `VmExit` label when the processor performs a VM-exit (the host RIP in the
//! VMCS points there).
//!
//! Adapted from `Hello-VT-rp` (`run_vmx_vm.S`) to the System V AMD64 C ABI:
//! the `registers` argument is passed in `rdi` rather than `rcx`.

use core::arch::global_asm;

/// Guest general-purpose registers, saved and restored across a VM-entry.
///
/// The field layout (offsets `0x00..=0x70`) is mirrored by the offsets in the
/// `run_vmx_vm` assembly below and must not be reordered. `rip`, `rsp` and
/// `rflags` live in dedicated VMCS fields and are only kept here for the caller.
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct GuestRegisters {
	pub rax: u64,
	pub rbx: u64,
	pub rcx: u64,
	pub rdx: u64,
	pub rdi: u64,
	pub rsi: u64,
	pub rbp: u64,
	pub r8: u64,
	pub r9: u64,
	pub r10: u64,
	pub r11: u64,
	pub r12: u64,
	pub r13: u64,
	pub r14: u64,
	pub r15: u64,
	// Not touched by the trampoline; mirrored into the VMCS by the caller.
	pub rip: u64,
	pub rsp: u64,
	pub rflags: u64,
}

unsafe extern "C" {
	/// Runs the guest until the next VM-exit.
	///
	/// Returns the host `RFLAGS` value right after `VMLAUNCH`/`VMRESUME` so the
	/// caller can detect a failed VM-entry (`VMfailValid`/`VMfailInvalid`).
	pub fn run_vmx_vm(registers: &mut GuestRegisters) -> u64;
}

global_asm!(
	r#"
.set registers_rax, 0x00
.set registers_rbx, 0x08
.set registers_rcx, 0x10
.set registers_rdx, 0x18
.set registers_rdi, 0x20
.set registers_rsi, 0x28
.set registers_rbp, 0x30
.set registers_r8,  0x38
.set registers_r9,  0x40
.set registers_r10, 0x48
.set registers_r11, 0x50
.set registers_r12, 0x58
.set registers_r13, 0x60
.set registers_r14, 0x68
.set registers_r15, 0x70

.global run_vmx_vm
run_vmx_vm:
    // Save the current (host) general-purpose registers onto the stack.
    push    rax
    push    rcx
    push    rdx
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r8
    push    r9
    push    r10
    push    r11
    push    r12
    push    r13
    push    r14
    push    r15

    // r15 <= `registers` (System V first argument in rdi). Keep a copy on the
    // top of the stack so it can be recovered after VM-exit.
    mov     r15, rdi
    push    r15

    // Load the guest general-purpose registers and try VMRESUME.
    mov     rax, [r15 + registers_rax]
    mov     rbx, [r15 + registers_rbx]
    mov     rcx, [r15 + registers_rcx]
    mov     rdx, [r15 + registers_rdx]
    mov     rdi, [r15 + registers_rdi]
    mov     rsi, [r15 + registers_rsi]
    mov     rbp, [r15 + registers_rbp]
    mov     r8,  [r15 + registers_r8]
    mov     r9,  [r15 + registers_r9]
    mov     r10, [r15 + registers_r10]
    mov     r11, [r15 + registers_r11]
    mov     r12, [r15 + registers_r12]
    mov     r13, [r15 + registers_r13]
    mov     r14, [r15 + registers_r14]
    mov     r15, [r15 + registers_r15]
    vmresume
    pushf                       // Preserve flags; the next VMREAD clobbers them.

    // If VMRESUME failed because the VMCS was never launched (error 5), fall
    // through to VMLAUNCH. This happens on the very first entry.
    mov     r15, 0x4400         // VM-instruction error field
    vmread  r15, r15
    cmp     r15, 5
    jz      2f
    popf                        // Restore flags ...
    jmp     4f                  // ... and report the VM-entry failure.

2:  // Launch
    pop     r15                 // Discard the saved flags.
    mov     r15, 0x6C14         // Host RSP
    vmwrite r15, rsp
    lea     r14, [rip + 3f]
    mov     r15, 0x6C16         // Host RIP
    vmwrite r15, r14
    mov     r15, [rsp]          // r15 <= `registers`
    mov     r14, [r15 + registers_r14]
    mov     r15, [r15 + registers_r15]
    vmlaunch
    jmp     4f                  // VMLAUNCH failed if we reach here.

3:  // VmExit: control returns here on VM-exit (host RIP points to this label).
    xchg    r15, [rsp]          // r15 <= `registers`, [rsp] <= guest r15
    mov     [r15 + registers_rax], rax
    mov     [r15 + registers_rbx], rbx
    mov     [r15 + registers_rcx], rcx
    mov     [r15 + registers_rdx], rdx
    mov     [r15 + registers_rsi], rsi
    mov     [r15 + registers_rdi], rdi
    mov     [r15 + registers_rbp], rbp
    mov     [r15 + registers_r8],  r8
    mov     [r15 + registers_r9],  r9
    mov     [r15 + registers_r10], r10
    mov     [r15 + registers_r11], r11
    mov     [r15 + registers_r12], r12
    mov     [r15 + registers_r13], r13
    mov     [r15 + registers_r14], r14
    mov     rax, [rsp]          // rax <= guest r15
    mov     [r15 + registers_r15], rax

4:  // Exit
    pop     rax                 // Drop the saved `registers` pointer.

    // Restore the host general-purpose registers from the stack.
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     r11
    pop     r10
    pop     r9
    pop     r8
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    pop     rdx
    pop     rcx
    pop     rax

    // Return the RFLAGS value produced by VMLAUNCH/VMRESUME.
    pushfq
    pop     rax
    ret
"#
);
