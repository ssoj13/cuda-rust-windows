// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selects the committed bindings for the build target.
//!
//! No bindgen and no CUDA toolkit are needed here: `generated/<target>/` holds
//! the output of `cargo xtask bindings` and `generated/meta.json` records which
//! cfgs that target's headers imply. This script points `src/` at the right
//! directory and replays the recorded cfgs.
//!
//! Why: the bindings are committed (as `cudarc` does) so consumers build with no
//! CUDA toolkit or libclang, while `libcuda` / `libcurand` still resolve per
//! symbol at run time. One directory per target because bindgen output differs
//! per platform (C enums are `c_int` on MSVC but `c_uint` elsewhere, layout
//! asserts). A newer installed toolkit only produces a warning instead of
//! regenerating: the same sources must build identically on every machine, and
//! a toolkit-dependent build is exactly what committing the bindings removes.
//! The recorded cfgs are also exported through `links` so `cuda-core` follows
//! the bindings rather than probing a toolkit that may be absent or different.

use std::process::exit;

// Target table and toolkit discovery, shared with the xtask generator.
include!("shared.rs");

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        exit(1);
    }
}

fn target_var(name: &str) -> Result<String, String> {
    env::var(name).map_err(|e| format!("{name}: {e}"))
}

fn run() -> Result<(), String> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shared.rs");
    println!("cargo:rerun-if-changed=toolkit_target.rs");
    println!("cargo:rerun-if-changed=generated");
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-env-changed={TOOLKIT_TARGET_DIR_ENV}");
    for cfg in BINDINGS_CFGS {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
    }

    let (os, arch, tenv) = (
        target_var("CARGO_CFG_TARGET_OS")?,
        target_var("CARGO_CFG_TARGET_ARCH")?,
        target_var("CARGO_CFG_TARGET_ENV")?,
    );
    let target = find_target(&os, &arch, &tenv).ok_or_else(|| {
        let supported: Vec<&str> = TARGETS.iter().map(|t| t.key).collect();
        format!(
            "cuda-bindings has no committed bindings for {arch}-{os}-{tenv} (supported: {}). \
             Add the target to cuda-bindings/shared.rs and run `cargo xtask bindings --target <triple>`.",
            supported.join(", ")
        )
    })?;

    let generated = PathBuf::from(target_var("CARGO_MANIFEST_DIR")?).join("generated");
    let dir = generated.join(target.key);
    let meta_path = generated.join("meta.json");
    let meta: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&meta_path).map_err(|e| format!("{}: {e}", meta_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", meta_path.display()))?;
    let entry = &meta["targets"][target.key];
    let recorded_version = entry["cuda_version"].as_u64().ok_or_else(|| {
        format!(
            "{} has no entry for {}; run `cargo xtask bindings --target {}`",
            meta_path.display(),
            target.key,
            target.key
        )
    })?;
    let cfgs = entry["cfgs"]
        .as_array()
        .ok_or_else(|| format!("{}: {} has no `cfgs` array", meta_path.display(), target.key))?;
    for cfg in cfgs {
        let cfg = cfg.as_str().filter(|c| BINDINGS_CFGS.contains(c)).ok_or_else(|| {
            format!("{}: unknown cfg {cfg} for {}", meta_path.display(), target.key)
        })?;
        println!("cargo:rustc-cfg={cfg}");
        // Reaches dependents (`cuda-core`) as `DEP_CUDA_BINDINGS_<CFG>=1`
        // through the `links` key, so they gate on the same headers.
        println!("cargo::metadata={cfg}=1");
    }
    println!("cargo:rustc-env=CUDA_BINDINGS_GENERATED={}", dir.display());

    match discover_toolkit(None, &arch, &os) {
        Ok(Some(toolkit)) => {
            // Re-run when the toolkit's header changes (e.g. after an upgrade) so
            // the freshness warning stays accurate without a `cargo clean`.
            println!(
                "cargo:rerun-if-changed={}",
                toolkit.include_dir.join("cuda.h").display()
            );
            if u64::from(toolkit.version) > recorded_version {
                println!(
                    "cargo:warning=installed CUDA toolkit {} is newer than the committed bindings ({}); \
                     run `cargo xtask bindings` to regenerate",
                    format_cuda_version(toolkit.version),
                    format_cuda_version(recorded_version as u32),
                );
            }
        }
        Ok(None) => {}
        Err(error) => println!("cargo:warning=CUDA toolkit ignored: {error}"),
    }
    Ok(())
}
