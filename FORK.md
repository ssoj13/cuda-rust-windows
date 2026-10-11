# CUDA Rust Windows Fork Policy

## Purpose

This fork adds native Windows support for `x86_64-pc-windows-msvc` while
keeping Linux behavior upstream-compatible. Windows changes cover build
tools, library discovery, CI, smoke tests, and platform compatibility fixes.

The shared host runtime lives at the repository root. SIMT examples use
those crates through path dependencies; generated standalone projects pin
the runtime and compiler to one fork revision. The isolated backend uses
the in-tree COFF artifact writer; host crates use the published artifact
format and types.

## Upstream Repository

- Upstream: https://github.com/NVIDIA/cuda-rust
- Upstream branch: `NVIDIA/cuda-rust` `upstream/main`
- Upstream release baseline: CUDA-Oxide 0.2.1
- Primary local branch: `main` Windows release fork branch tracking
  `upstream/main`
- Short-lived branches may be used for experiments, but routine upstream sync
  lands directly on `main`.

## Branch Rules

- `main` remains the Windows release fork branch. It tracks
  `NVIDIA/cuda-rust` `upstream/main`.
- Short-lived branches may be used for Windows enablement work, experiments,
  CI fixes, and compatibility patches.
- Keep Windows patches narrow and reviewable. Prefer build-system, path,
  environment, and documentation helpers over API changes.
- Do not add fork-only public API unless it has been discussed and documented.
- Linux regressions block Windows changes unless the pull request documents why
  the regression is unavoidable and what follow-up restores parity.
- Update the divergence log only for intentional behavior differences, not for
  transient merge conflict resolution or ordinary documentation refreshes.

## Upstream Sync

Configure the upstream remote once:

```bash
git remote add upstream https://github.com/NVIDIA/cuda-rust.git
git remote -v
```

Refresh upstream state:

```bash
git fetch upstream --tags
git checkout main
git status --short
```

Regular sync should use this fetch-plus-merge flow on `main` so the fork keeps
its published history intact while still carrying upstream commits promptly.

The working tree must be clean before merging. If you have unfinished Windows
work, move it to a short-lived branch before syncing.

Merge upstream:

```bash
git merge --no-ff upstream/main
```

Do not rewrite published `main` history for routine syncs.

Conflict policy:

- Preserve upstream behavior first. Prefer the upstream file when behavior is
  unrelated to Windows support.
- Re-apply Windows helpers after the upstream version is understood.
- Keep compatibility patches narrow: paths, environment discovery, CI, docs,
  and smoke support before public API changes.
- Do not use conflict resolution to introduce Linux behavior changes.
- Update this file only when the resolved result intentionally diverges from
  upstream behavior.

Run the canonical no-GPU checks from the repository root:

```powershell
.\scripts\sync-upstream.ps1 -RunChecks
```

Push a completed sync only after the checks pass:

```powershell
.\scripts\sync-upstream.ps1 -RunChecks -Push
```

The helper merges `upstream/main` into `main` and, when `-Push` is passed,
uses a normal `git push origin main`. It does not create releases, move tags,
or change versions.

On a Windows GPU host, also run the full Windows smoke path:

```powershell
cargo oxide doctor
cargo oxide build vecadd
cargo oxide run vecadd
.\cuda-oxide\scripts\smoketest.ps1
```

When a sync changes Windows readiness, update
[CHANGELOG.windows.md](CHANGELOG.windows.md). Keep the changelog focused on
validated targets, requirements, examples, unsupported items, and known
release-readiness gaps.

## Maintenance Cycle

- Daily and weekly upstream monitor: `.github/workflows/upstream-monitor.yml`
  compares this fork with `NVIDIA/cuda-rust/main` and opens or updates one
  issue when upstream has new commits.
- Weekly upstream sync: `.github/workflows/upstream-sync-main.yml` merges
  `NVIDIA/cuda-rust/main` into `main`, runs `.\scripts\sync-upstream.ps1
  -RunChecks -Push`, and opens or updates one issue if the sync fails.
- Weekly hosted Windows canary: `.github/workflows/windows.yml` runs the
  no-GPU MSVC lane on GitHub-hosted `windows-latest`.
- Manual sync: run `.\scripts\sync-upstream.ps1 -RunChecks`; add `-Push` only
  after the local result is clean.
- Release rule: publish `windows-vX.Y.Z` only when upstream has released
  `vX.Y.Z`. Sync-only rebuilds do not invent new project versions.

## Divergence Log Format

Use this format whenever the fork intentionally keeps behavior that differs
from upstream:

```text
## YYYY-MM-DD - short title

- Branch:
- Upstream baseline:
- Files/area:
- Intentional divergence:
- Linux impact:
- Windows validation:
- Follow-up:
```

## Current Divergence Log

## 2026-10-10 - Lowering temporaries in the entry block; local-memory-free sin_cos

- Branch: `perf/entry-block-allocas`, merged into `main` of `ssoj13/cuda-rust-windows`.
- Upstream baseline: fork `main` 158229b5b.
- Files/area: `mir-lower` (`convert/ops/stack_slot.rs`: `entry_alloca`, used by `mir.ref`,
  enum payload reads, runtime-indexed array reads and transmutes), `cuda-device`
  (`math::sin_cos`, `#![feature(core_float_math)]`), example `math_sin_cos`.
- Intentional divergence: every stack slot the lowering needs is a static alloca at the start
  of the function's entry block; only its store and loads stay at the use site. Created at the
  use site (inside a loop), the slot survived SROA (entry-block allocas only), NVPTX hoisted it
  into the local frame and every access became a local load or store. WarpBro, sm_86: fast_*
  frames 456 -> 144-200 bytes, full_* 2248 -> 1632, the per-march-step `Option<Hit>` round trip
  through local memory gone. `cuda_device::math::sin_cos` is libdevice `__nv_sinf/__nv_cosf`'s
  Cody-Waite path alone (|x| < 105615), in plain `core` float math: bit-identical to libdevice
  on 2^24 arguments, on device and host, without the Payne-Hanek table in local memory.
- Linux impact: lowering only moves allocas; same IR semantics. Linux `clippy`, `test`,
  `test-cuda`, `check-guards` pass; the full Linux smoketest did not complete (WSL ran out of
  disk) and is pending. `check-intrinsics` fails on the probe's recorded nightly-vs-stable llc
  version, independent of this change.
- Windows validation: mir-lower (incl. `slot_of_a_later_block_is_allocated_in_the_entry_block`),
  llvm-export, mir-importer, dialect-mir, cuda-device clippy and tests; `math_sin_cos` on an
  RTX 3080 Ti; WarpBro builds and its kernels measured with `bootstrap.py k`.
- Follow-up: rerun the Linux smoketest; offer upstream with the constant-memory fix.

## 2026-10-10 - Ahead-of-time cubins in a fat binary

- Branch: `feat/fatbin-archs`, merged into `main` of `ssoj13/cuda-rust-windows`.
- Upstream baseline: fork `main` 6a2907ae5.
- Files/area: `cuda-artifact-finalizer` (`tool.rs`: pinned `ptxas` / `fatbinary`
  shared by `PtxAssembler` and the new `FatbinBuilder`; errors `ToolNotFound` /
  `InvalidTool` / `ToolFailed` replace the ptxas-only variants), `rustc-codegen-cuda`
  (`materialize.rs`, `lib.rs`), `cargo-oxide` (`cubin-archs` config key, fingerprint),
  `cuda-host` and root `cuda-core` loaders, `ptx-schedule` artifact patching,
  `reserved-oxide-symbols` (`CUBIN_ARCHS_ENV`).
- Intentional divergence: `cubin-archs` / `CUDA_OXIDE_CUBIN_ARCHS` assembles the linked
  PTX with `ptxas` for each listed architecture and embeds one compressed fat binary in
  the bundle's `Cubin` slot, ahead of the unchanged PTX. Loaders fall back to the PTX
  when the driver rejects the binary image. The fat binary reuses the `Cubin` payload
  kind because a new kind would break every oxide-artifacts 0.2.1 reader. Measured on
  WarpBro (22 kernels): four cubins sm_75/86/89/120 compress to 6.1 MB against 8.9 MB
  of PTX; load 0.02 s against a 97 s cold PTX JIT. The LTO route of
  `--materialize-cubin` produced a 23 MB cubin in ~15 min for one architecture.
- Linux impact: none unless the option is set.
- Windows validation: finalizer, backend, cargo-oxide (260), cuda-host, cuda-core and
  ptx-schedule tests; live ptxas/fatbinary tests; `constant_memory` with a fat binary
  for this GPU, with one for another GPU (PTX fallback), and without one.
- Known limits: `checked_targets` compares capability only (an `a`/`f` PTX target that
  ptxas cannot retarget fails loudly in ptxas); a ptx-schedule campaign on a fat-binary
  executable times its baseline from the cubins and its variants through the PTX JIT;
  `examples-compile.yml` reads payload 0 and does not exercise the option.
- Follow-up: offer upstream together with the constant-memory fix.

## 2026-10-09 - ssoj13/cuda-rust-windows: inline intent, Rust 1.99, monorepo sync

- Branch: `main` of `ssoj13/cuda-rust-windows` (renamed from cuda-oxide-windows),
  tracking `ansidium/cuda-rust-windows` main (`upstream` remote).
- Upstream baseline: ansidium `dbaf94819` (NVIDIA/cuda-rust monorepo layout), merged
  in `11c034bcb`.
- Files/area: `cuda-oxide/crates/{reserved-oxide-symbols,dialect-mir,mir-importer,
  mir-transforms,mir-lower,llvm-export,rustc-codegen-cuda}` (inline intent);
  `cuda-oxide/crates/cargo-oxide/src/backend.rs` (pinned backend source).
- Intentional divergence: every Rust `#[inline]` intent reaches LLVM (`#[inline]` ->
  `inlinehint`, `always`/`force` -> `alwaysinline`, `never` -> `noinline`) through
  `InlineIntent` / `MirFuncOp::inline_intent` and `llvm_func_attrs`; upstream only
  emits `alwaysinline` (NVIDIA #188). The `#[inline(never)]` that `#[device]` adds to
  generic device functions for the collector stays host-only
  (`is_generic_device_collector_boundary`); carried as `noinline` it broke const-generic
  folding (`const_generic` example). The cargo-oxide backend pin points at this
  fork. The Rust 1.99 API fixes equal ansidium's and need no divergence.
- Intentional divergence (cargo-oxide, `commands/codegen_env.rs`): release-like
  routes pin opt-level / debug-assertions / overflow-checks / debuginfo through
  `CARGO_PROFILE_{RELEASE,DEV,<--profile>}_*` (`CodegenProfilePolicy::pinned_options`)
  instead of trailing `-C` rustflags; a `-C` pin is appended only when incoming
  rustflags set the same option. Upstream's rustflags changed every crate's cache
  key, so `build` after `test -- --release` rebuilt the whole graph (~680 crates).
- Linux impact: none beyond the inline keywords, which apply on every platform. The
  profile pins apply on every platform; embedded PTX of WarpBro is byte-identical.
- Windows validation: compiler crate tests, backend check, `vecadd` and
  `addressof_sharedarray` on an RTX 3080 Ti (eight `noinline` helpers stay functions);
  cargo-oxide tests (259); WarpBro fresh target: tests, then `build` compiles 1 crate.
- Bug fix carried ahead of upstream (mir-lower `create_device_global`, llvm-export):
  `#[constant]` globals are retained and exported `externally_initialized`, so the
  materialized-cubin route (libNVVM `-gen-lto` + nvJitLink `-lto`) keeps the symbol the
  host writes; before, LTO folded it to zeros and `cuModuleGetGlobal` failed (500).
  Repro: `cargo oxide run constant_memory --materialize-cubin --arch sm_86`.
- Follow-up: offer the inline-intent, profile-pin and constant-memory changes upstream
  (DCO sign-off required; the constant-memory PR branch is `fix/constant-memory-lto`).

## 2026-06-22 - Upstream sync and non-rewriting maintenance

- Branch: `main` Windows release fork branch tracking `upstream/main`.
- Upstream baseline: `upstream/main@d63a0a8d3fef2db450ee342bdcd862a7829c3cbb`,
  CUDA-Oxide 0.2.1.
- Files/area: upstream merge integration, cargo-oxide backend cache handling,
  cuda-core upstream behavior, and sync automation.
- Intentional divergence: keep the Windows support layer on current upstream
  through merge-based `main` syncs.
- Linux impact: intended to be none. Upstream `DeviceBuffer` behavior and
  backend cache source/toolchain invalidation are preserved; Windows-specific
  artifact naming, release-profile backend builds, and loader path handling
  remain scoped to Windows targets.
- Windows validation: run the no-GPU sync sequence and hosted Windows canary
  before publishing release artifacts.
- Follow-up: no new `windows-vX.Y.Z` release is needed until upstream publishes
  a new CUDA-Oxide release baseline.

## 2026-06-18 - Upstream sync and stronger Windows canary

- Branch: `main` Windows release fork branch tracking `upstream/main`.
- Upstream baseline: `upstream/main@56b843f618d973aef6ae4cb613b590008df09a70`,
  CUDA-Oxide 0.2.1.
- Files/area: Windows CI, cargo-oxide backend loader-path handling, and
  cuda-core sync repair.
- Intentional divergence: keep the Windows support layer synced with current
  upstream while strengthening the hosted no-GPU Windows canary to install
  `libffi:x64-windows` through a temporary vcpkg manifest seeded with the
  runner's vcpkg baseline, build the codegen backend, and compile-check
  `vecadd`.
- Linux impact: intended to be none. The sync repair preserves the existing
  `DeviceBuffer` behavior while filling fields required by upstream's
  async-free model.
- Windows validation: `cargo oxide setup` built
  `rustc_codegen_cuda.dll` in release profile, and
  `cargo oxide build vecadd --arch sm_75` completed successfully.
- Follow-up: no new `windows-vX.Y.Z` release is needed until upstream publishes
  a new CUDA-Oxide release baseline.

## 2026-06-14 - Windows support layer

- Branch: `main` Windows release fork branch tracking `upstream/main`.
- Upstream baseline: `upstream/main@cb318ad4e4e37f5e1913ed0a13478af990e857f7`,
  CUDA-Oxide 0.2.1.
- Files/area: CUDA Toolkit discovery, Windows/MSVC path handling, platform
  artifact naming, loader environment handling, import-library checks,
  bindgen/CUDA compatibility, Windows CI, and smoke scripts.
- Intentional divergence: add native Windows support infrastructure for
  `x86_64-pc-windows-msvc` while keeping upstream Linux behavior first. The
  Windows layer covers CUDA Toolkit discovery, Windows and MSVC paths,
  `.exe`/`.dll`/`.obj` platform naming, `PATH` loader behavior, `cuda.lib` and
  `ffi.lib` checks, bindgen/CUDA type compatibility, Windows CI, and
  PowerShell smoke scripts.
- Linux impact: intended to be none. If a Windows helper conflicts with
  upstream Linux behavior, preserve upstream behavior first and re-apply only
  the Windows-specific compatibility needed for MSVC support.
- Windows validation: `cargo build -p cargo-oxide`,
  `cargo test -p oxide-artifacts --features object`,
  `cargo oxide doctor`, `cargo oxide build vecadd`, GPU-host
  `cargo oxide run vecadd`, and `.\scripts\smoketest.ps1` as applicable.
- Follow-up: during each upstream sync, check whether upstream has added native
  equivalents and remove fork-only helpers when upstream behavior covers the
  Windows case.
