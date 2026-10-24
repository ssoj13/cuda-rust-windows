// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `cargo xtask bindings [--toolkit DIR] [--target KEY]... [--clang-arg ARG]... [--out DIR]`

use std::{path::PathBuf, process::ExitCode};

use xtask::{all_keys, generate, generated_dir, host_key, write, Res};

const USAGE: &str = "\
usage: cargo xtask bindings [--toolkit DIR] [--target KEY]... [--clang-arg ARG]... [--out DIR]

Regenerates cuda-bindings/generated/<target>/ and merges generated/meta.json.
  --toolkit DIR    CUDA 13.0+ toolkit root (default: CUDA_TOOLKIT_PATH / CUDA_HOME / standard locations)
  --target KEY     target triple to generate (repeatable; default: the host target)
  --clang-arg ARG  extra clang argument, e.g. --sysroot=DIR for a cross target (repeatable)
  --out DIR        output root (default: cuda-bindings/generated)";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Res<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("bindings") => {}
        _ => return Err(USAGE.into()),
    }
    let (mut toolkit, mut targets, mut clang_args) = (None, Vec::new(), Vec::new());
    let mut out = generated_dir();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--toolkit" => toolkit = Some(PathBuf::from(value()?)),
            "--target" => targets.push(value()?),
            "--clang-arg" => clang_args.push(value()?),
            "--out" => out = PathBuf::from(value()?),
            _ => return Err(format!("unknown argument {flag}\n{USAGE}").into()),
        }
    }
    if targets.is_empty() {
        let host = host_key().ok_or_else(|| {
            format!("host is not a supported target; pass --target ({})", all_keys().join(", "))
        })?;
        targets.push(host.to_string());
    }
    for key in &targets {
        let output = generate(key, toolkit.as_deref(), &clang_args)?;
        write(&output, &out)?;
        let bytes: usize = output.files.iter().map(|(_, text)| text.len()).sum();
        println!("{key}: {} files, {bytes} bytes -> {}", output.files.len(), out.display());
    }
    Ok(())
}
