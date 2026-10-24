// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sources shared by `build.rs` and the `xtask` generator: the supported-target
// table and CUDA toolkit discovery. Pulled in with `include!` (regular `//`
// comments, no inner docs) so there is exactly one copy of both; `build.rs`
// uses them to pick the committed bindings and to warn about a newer toolkit,
// `xtask` to generate them.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

/// A target the committed bindings are generated for. `key` is the Rust target
/// triple and the name of the `generated/<key>/` directory; the other fields
/// are cargo's `CARGO_CFG_TARGET_*` values that select it.
struct Target {
    key: &'static str,
    os: &'static str,
    arch: &'static str,
    env: &'static str,
}

/// Every target the fork supports. Bindgen output differs per platform (C enum
/// width is `c_int` on MSVC and `c_uint` on GCC/Clang, plus layout
/// assertions), hence one directory each. Anything else is a build error.
const TARGETS: &[Target] = &[
    Target { key: "x86_64-pc-windows-msvc", os: "windows", arch: "x86_64", env: "msvc" },
    Target { key: "x86_64-unknown-linux-gnu", os: "linux", arch: "x86_64", env: "gnu" },
    Target { key: "aarch64-unknown-linux-gnu", os: "linux", arch: "aarch64", env: "gnu" },
];

/// The supported target for cargo's `CARGO_CFG_TARGET_OS` / `_ARCH` / `_ENV`.
fn find_target(os: &str, arch: &str, env: &str) -> Option<&'static Target> {
    TARGETS
        .iter()
        .find(|t| t.os == os && t.arch == arch && t.env == env)
}

/// Every cfg the committed bindings can record in `generated/meta.json`.
/// `xtask` detects them from the headers / generated types; `build.rs` replays
/// the recorded ones for the selected target and declares all as known.
const BINDINGS_CFGS: [&str; 4] = [
    // `CUmemLocation_st::id` sits in an anonymous union (CUDA 13.2+).
    "cu_mem_location_anon_union",
    // CUDA 12.8 renamed the event elapsed-time entry point.
    "cuda_has_cuEventElapsedTime_v2",
    // CUDA 13.2 added the sync-mode host function launch.
    "cuda_has_cuLaunchHostFunc_v2",
    // CUDA 12.1 added the `cuMulticast*` driver API; `cuda-core` gates `vmm` on it.
    "cuda_has_multicast",
];

const MIN_CUDA_VERSION: u32 = 13000;

/// Environment variables consulted (in order) to locate the toolkit root.
/// `CUDA_HOME` is the conventional name used by nvcc wrappers and CI images.
const TOOLKIT_ENV_VARS: [&str; 2] = ["CUDA_TOOLKIT_PATH", "CUDA_HOME"];

/// Names one CUDA `targets/<dir>` tree by hand, like nvcc's `-target-dir`.
/// When set it is the only include candidate: the top-level `include/` (on
/// standard installs a symlink to the host architecture's tree) is not
/// consulted, so a cross-build cannot silently bind the host's headers.
const TOOLKIT_TARGET_DIR_ENV: &str = "CUDA_TOOLKIT_TARGET_DIR";

// The `targets/<dir>` selection table, also unit tested by
// `tests/toolkit_target.rs` and mirrored by hand in cargo-oxide's doctor.
include!("toolkit_target.rs");

/// A validated CUDA toolkit: the include directory that actually contains
/// `cuda.h` (`{root}/include` on standard installs, `{root}/targets/<dir>/include`
/// for redistributable and Tegra/sbsa layouts) and its `CUDA_VERSION`.
struct Toolkit {
    include_dir: PathBuf,
    version: u32,
}

/// Finds a toolkit for the target `arch`/`os`. `explicit` (xtask `--toolkit`)
/// wins, then the first set variable of [`TOOLKIT_ENV_VARS`], then the standard
/// install locations. `Ok(None)` only when nothing was named and no default
/// location holds a valid toolkit; a named toolkit that is invalid is an `Err`.
fn discover_toolkit(
    explicit: Option<&Path>,
    arch: &str,
    os: &str,
) -> Result<Option<Toolkit>, String> {
    if let Some(root) = explicit {
        return validate_toolkit(root, arch, os)
            .map(Some)
            .map_err(|e| format!("--toolkit {} is invalid: {e}", root.display()));
    }
    for var in TOOLKIT_ENV_VARS {
        let Some(value) = env::var_os(var) else {
            continue;
        };
        if value.is_empty() {
            return Err(format!(
                "{var} is set to an empty string. Set it to a CUDA 13.0+ toolkit directory."
            ));
        }
        let root = PathBuf::from(value);
        return validate_toolkit(&root, arch, os)
            .map(Some)
            .map_err(|e| format!("{var}={} is invalid: {e}", root.display()));
    }
    Ok(default_toolkit_roots()
        .iter()
        .find_map(|root| validate_toolkit(Path::new(root), arch, os).ok()))
}

#[cfg(windows)]
fn default_toolkit_roots() -> &'static [&'static str] {
    &[
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    ]
}

#[cfg(not(windows))]
fn default_toolkit_roots() -> &'static [&'static str] {
    &[
        "/usr/local/cuda-13.3",
        "/usr/local/cuda-13.2",
        "/usr/local/cuda-13",
        "/usr/local/cuda",
        "/opt/cuda",
    ]
}

/// Probes the include layouts for `cuda.h` (only the directories for the given
/// architecture, never all of `targets/*`) and checks the version floor.
fn validate_toolkit(root: &Path, arch: &str, os: &str) -> Result<Toolkit, String> {
    if !root.is_dir() {
        return Err(format!("{} is not a directory", root.display()));
    }
    let override_dir = env::var(TOOLKIT_TARGET_DIR_ENV).ok();
    let candidates = resolve_toolkit_include_candidates(root, override_dir.as_deref(), arch, os);
    let Some(include_dir) = select_include_dir(&candidates) else {
        let probed: Vec<String> = candidates
            .iter()
            .map(|dir| format!("    {}", dir.join("cuda.h").display()))
            .collect();
        return Err(format!(
            "{} does not contain cuda.h. Probed:\n{}",
            root.display(),
            probed.join("\n")
        ));
    };
    let version = cuda_version_from_header(&include_dir.join("cuda.h"))?;
    if version < MIN_CUDA_VERSION {
        return Err(format!(
            "CUDA toolkit {} is too old (13.0+ required)",
            format_cuda_version(version)
        ));
    }
    Ok(Toolkit {
        include_dir: include_dir.clone(),
        version,
    })
}

/// The `#define CUDA_VERSION` value of a `cuda.h`.
fn cuda_version_from_header(cuda_h: &Path) -> Result<u32, String> {
    let source = fs::read_to_string(cuda_h).map_err(|e| format!("{}: {e}", cuda_h.display()))?;
    source
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
                _ => None,
            }
        })
        .ok_or_else(|| format!("could not find CUDA_VERSION in {}", cuda_h.display()))
}

fn format_cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}
