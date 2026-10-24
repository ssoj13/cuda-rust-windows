/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Gates the multicast driver API (CUDA 12.1+) on what `cuda-bindings` binds.
//!
//! The `cuMulticast*` entry points first appeared in CUDA 12.1 and
//! `cuda-bindings` binds whatever header set its committed bindings were
//! generated from. `cuda-bindings` records that as the `cuda_has_multicast`
//! cfg and exports it through its `links` key, so this crate follows the
//! bindings themselves and needs no CUDA toolkit (and no second header probe
//! that could disagree with them).

fn main() {
    const VAR: &str = "DEP_CUDA_BINDINGS_CUDA_HAS_MULTICAST";
    println!("cargo::rustc-check-cfg=cfg(cuda_has_multicast)");
    println!("cargo:rerun-if-env-changed={VAR}");
    if std::env::var_os(VAR).is_some() {
        println!("cargo:rustc-cfg=cuda_has_multicast");
    }
}
