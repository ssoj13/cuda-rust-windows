# HANDOFF (fork ssoj13/cuda-rust-windows)

Keep this file current; read `AGENTS.md` (== `CLAUDE.md`) first.

## 2026-10-24 - committed CUDA driver bindings (fbb7df9, main)

- `cuda-bindings` no longer runs bindgen at build time: `cuda-bindings/generated/<triple>/` +
  `generated/meta.json` (per-target CUDA version, header hashes, cfgs) are committed, as cudarc
  does, so consumers build without a CUDA toolkit or libclang; the driver still loads per symbol
  at run time. Why per target: bindgen output differs per platform (C enum repr, layouts).
- Regenerate: `cargo xtask bindings [--toolkit <dir>] [--target <triple>]... [--clang-arg ..]`
  (generator moved from build.rs to `xtask/`; shared table + discovery in `cuda-bindings/shared.rs`).
  Sets: x86_64-pc-windows-msvc (native), x86_64-unknown-linux-gnu (WSL, libclang-22),
  aarch64-unknown-linux-gnu (WSL + arm64 cross libc; generated, NOT compiled). Procedure in
  `cuda-bindings/README.md`.
- `build.rs` only selects the set and replays cfgs; a newer installed toolkit -> warning, never a
  silent switch. cfgs reach `cuda-core` via `links = "cuda_bindings"` (`DEP_CUDA_BINDINGS_*`), so
  cuda-core's own header probe is gone. `cuda_toolkit_dir()` was removed (no caller in our repos;
  lgpt-rs confirmed it does not use it).
- Freshness test: `xtask/tests/regen.rs` (skips without a toolkit; `CUDA_BINDINGS_REQUIRE_TOOLKIT=1`
  makes the skip a failure).
- Verified: `cargo check -p cuda-bindings -p cuda-core` with `CUDA_TOOLKIT_PATH` at a missing dir
  builds (warning only); warpbro-rs d9980e1 builds both crates `--locked`.

## Next / open

- `.github/workflows/unit-tests.yml` still installs the CUDA toolkit; no CI job runs the regen
  test.
- `cuda-oxide/crates/cargo-oxide/src/commands/doctor.rs` still says host crates need `cuda.h`.
- No Windows ARM64 set (no CI target); aarch64 Linux set never compiled.
- Consumers on old revisions: fractal-rs, maya-rs, render-rs, openvdb-rs, ofx-rs (operator: later).
- Context: oiio-rs `plan-cuda-oxide.md` (stage 0b) and oiio-rs `HANDOFF.md` (2026-10-24).
- Untracked `crates/` and `inst.cmd` in the root are not this session's: never touch.
