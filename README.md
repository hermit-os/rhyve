# rhyve

rhyve is a bare-metal hypervisor on top of the [Hermit kernel](https://github.com/hermit-os/kernel).

rhyve itself runs as a Hermit unikernel and boots a guest inside a VM. It
exposes a small web service: upload a guest image, run it, and watch the guest's
serial output stream back to your browser.

![Screenshot of the running web service](images/rhyve.png)

## Requirements

- A Rust **nightly** toolchain — pinned by `rust-toolchain.toml`, with the
  `rust-src` and `llvm-tools` components.
- **QEMU/KVM** with **nested virtualization** enabled (rhyve runs a guest *inside*
  the VM, so the host CPU's VT-x / AMD-V must be exposed to the guest):
  ```sh
  cat /sys/module/kvm_intel/parameters/nested   # Intel — should print Y/1
  cat /sys/module/kvm_amd/parameters/nested      # AMD   — should print Y/1
  ```
- The Hermit **loader** (`hermit-loader-x86_64`) reachable by the QEMU runner in
  `.cargo/config.toml`.
- A Hermit [**guest image**]((https://github.com/hermit-os/hermit-rs)) 
  to run — any Hermit unikernel application binary. A
  ready-made `hello_world` is included under `data/x86_64/`.

rhyve must be built against a Hermit kernel that provides the
`sys_virt_addr_to_phys_addr` syscall (it translates host-virtual to
host-physical addresses for the nested page tables). This function is
provided by the kernel of the branch `rhyve`. Checkout this branch
on your local device

```sh
$ git clone git@github.com:hermit-os/kernel.git
$ cd kernel
$ git checkout rhyve
```

By building `rhyve`, the environment variable `HERMIT_MANIFEST_DIR` should
point to the location of the kernel, which provides to `sys_virt_addr_to_phys_addr`.

## Building and running

rhyve targets `x86_64-unknown-hermit` and is launched under QEMU by the runner
configured in `.cargo/config.toml`. Always use a **release** build (the debug
build does not boot reliably):

```sh
HERMIT_MANIFEST_DIR=/path/to/hermit/kernel cargo run --release
```

Once it is up you will see a line like:

```
rhyve upload service listening on http://0.0.0.0:9975/
```

The QEMU runner forwards host port **9975** to the service.

## Usage

### In the browser

Open <http://localhost:9975/>, choose a guest image, click **Upload**, then
**Run guest**. The guest's serial output streams into a scrollable box as it
runs; a small toolbar provides a *Clear* button, a live byte counter and a
*Wrap lines* toggle.

### From the command line

```sh
# upload a guest image, stored as "hello"
curl --upload-file data/x86_64/hello_world http://localhost:9975/image/hello

# run it and stream the output (-N disables curl's buffering)
curl -N -X POST http://localhost:9975/run/hello
```

### HTTP API

| Method & path        | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| `GET  /`             | The upload page.                                                                                 |
| `PUT  /image/{name}` | Store the request body as `/image/{name}`.                                                       |
| `POST /run/{name}`   | Boot `/image/{name}` as the guest; the response body streams its serial output and ends when the guest shuts down. |

Each run gives the guest 256 MiB of memory and ends when the guest writes its
exit port (e.g. `0xf4`), which closes the response stream.

## Credits

The current stage is just a proof of concept is derived from following tutorials and software distributions:

1. memN0ps ' turtorial [Hypervisor Development in Rust](https://memn0ps.github.io/hypervisor-development-in-rust-part-1/)
2. The Type-1 hypervisor [illusion-rs](https://github.com/memN0ps/illusion-rs)
3. The sample hypervisor [Hello-VT-rp](https://github.com/tandasat/Hello-VT-rp)
4. [memhv](https://github.com/SamuelTulach/memhv) Minimalistic hypervisor with memory introspection capabilities

## Licensing

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
