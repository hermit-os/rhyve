//! The low-level VM-entry trampoline for AMD-V.
//!
//! `VMRUN` only swaps a subset of the architectural state around guest
//! execution: it saves and restores the host CR/segment-selector/RIP/RSP/RAX
//! state through the area named by `VM_HSAVE_PA`, and it loads RAX/RSP/RIP/RFLAGS
//! for the guest from the VMCB. Everything else is the hypervisor's job:
//!
//! * The general-purpose registers other than RAX are neither saved nor restored,
//!   so the trampoline loads the guest GPRs before `VMRUN` and writes them back
//!   afterwards (RAX itself lives in the VMCB and is handled by the run loop).
//! * The "hidden" segment state (FS, GS, TR, LDTR together with `KernelGsBase`,
//!   STAR/LSTAR/CSTAR/SFMASK and the SYSENTER MSRs) is touched by neither the
//!   host-save area nor `VMRUN`. The trampoline therefore `VMSAVE`s the host's
//!   copy into a scratch VMCB, `VMLOAD`s the guest's from its VMCB before the run,
//!   `VMSAVE`s the guest's back afterwards and `VMLOAD`s the host's again — the
//!   same dance KVM performs.
//!
//! The [`GuestRegisters`] layout (and thus the field offsets below) is shared
//! with the VT-x backend.

use core::arch::global_asm;

pub use crate::vmx::GuestRegisters;

unsafe extern "C" {
	/// Runs the guest until the next `#VMEXIT`.
	///
	/// `registers` holds the guest GPRs the trampoline manages (everything except
	/// RAX/RSP/RIP/RFLAGS, which are mirrored through the VMCB by the run loop).
	/// `guest_vmcb_pa` and `host_vmcb_pa` are the host-physical addresses of the
	/// guest VMCB and a scratch VMCB used to preserve the host's hidden segment
	/// state across the run.
	pub fn run_svm_vm(registers: &mut GuestRegisters, guest_vmcb_pa: u64, host_vmcb_pa: u64);
}

global_asm!(
	r#"
.set svm_rax, 0x00
.set svm_rbx, 0x08
.set svm_rcx, 0x10
.set svm_rdx, 0x18
.set svm_rdi, 0x20
.set svm_rsi, 0x28
.set svm_rbp, 0x30
.set svm_r8,  0x38
.set svm_r9,  0x40
.set svm_r10, 0x48
.set svm_r11, 0x50
.set svm_r12, 0x58
.set svm_r13, 0x60
.set svm_r14, 0x68
.set svm_r15, 0x70

.global run_svm_vm
run_svm_vm:
    // System V arguments: rdi = registers, rsi = guest_vmcb_pa, rdx = host_vmcb_pa.
    // Preserve the callee-saved registers we are about to load guest state into.
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    // Keep the pointers needed after VMRUN on the stack; the guest will clobber
    // every general-purpose register.
    push    rdi                 // [rsp + 16] registers
    push    rsi                 // [rsp + 8]  guest_vmcb_pa
    push    rdx                 // [rsp + 0]  host_vmcb_pa

    // Save the host's hidden segment state (FS/GS/TR/LDTR/...) into the scratch VMCB.
    mov     rax, rdx
    vmsave  rax

    // Load the guest general-purpose registers (RAX is loaded from the VMCB by VMRUN).
    mov     rbx, [rdi + svm_rbx]
    mov     rcx, [rdi + svm_rcx]
    mov     rdx, [rdi + svm_rdx]
    mov     rsi, [rdi + svm_rsi]
    mov     rbp, [rdi + svm_rbp]
    mov     r8,  [rdi + svm_r8]
    mov     r9,  [rdi + svm_r9]
    mov     r10, [rdi + svm_r10]
    mov     r11, [rdi + svm_r11]
    mov     r12, [rdi + svm_r12]
    mov     r13, [rdi + svm_r13]
    mov     r14, [rdi + svm_r14]
    mov     r15, [rdi + svm_r15]
    mov     rdi, [rdi + svm_rdi] // load rdi last; it is the base pointer above

    // Mask interrupts globally (GIF = 0) for the entry window: while the guest's
    // hidden segment state is loaded the host must not take an interrupt, and any
    // physical interrupt arriving during the guest becomes a #VMEXIT(VMEXIT_INTR)
    // thanks to INTERCEPT_INTR rather than being injected into the guest.
    clgi

    // Load the guest's hidden segment state, run the guest, save it back.
    mov     rax, [rsp + 8]       // guest_vmcb_pa
    vmload  rax
    vmrun   rax
    vmsave  rax

    // Restore the host's hidden segment state, then unmask (GIF = 1): a physical
    // interrupt that exited the guest is now delivered to the host's own ISR,
    // with the host's FS/GS/TR/... back in place.
    mov     rax, [rsp]           // host_vmcb_pa
    vmload  rax
    stgi

    // Recover the registers pointer (its guest RAX value lives in the VMCB, not
    // in a GPR here) and write the guest GPRs back.
    xchg    rax, [rsp + 16]      // rax <= registers
    mov     [rax + svm_rbx], rbx
    mov     [rax + svm_rcx], rcx
    mov     [rax + svm_rdx], rdx
    mov     [rax + svm_rsi], rsi
    mov     [rax + svm_rdi], rdi
    mov     [rax + svm_rbp], rbp
    mov     [rax + svm_r8],  r8
    mov     [rax + svm_r9],  r9
    mov     [rax + svm_r10], r10
    mov     [rax + svm_r11], r11
    mov     [rax + svm_r12], r12
    mov     [rax + svm_r13], r13
    mov     [rax + svm_r14], r14
    mov     [rax + svm_r15], r15

    // Drop the three saved pointers and restore the callee-saved registers.
    add     rsp, 24
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
"#
);
