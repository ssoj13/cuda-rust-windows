// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The committed `cuda-bindings/generated/` tree must equal what the generator
//! produces from the host's CUDA toolkit.
//!
//! Why: `build.rs` never regenerates, so nothing else would notice hand edits,
//! generator changes that forgot to regenerate, or stale output after a toolkit
//! bump. Byte equality (not a semantic diff) keeps the committed files an exact
//! product of `cargo xtask bindings`.
//!
//! Without a toolkit (or on a host that is not a supported target) the test
//! prints why and skips; `CUDA_BINDINGS_REQUIRE_TOOLKIT=1` turns that skip
//! into a failure, for the CI job that is meant to guard the bindings.

use std::{fs, path::Path};

use xtask::{generate, generated_dir, host_key, write};

const REQUIRE_ENV: &str = "CUDA_BINDINGS_REQUIRE_TOOLKIT";

/// Skips (returns) with `reason`, or panics when the toolkit is required.
fn skip(reason: &str) {
    assert!(
        std::env::var_os(REQUIRE_ENV).is_none(),
        "{REQUIRE_ENV} is set but the check cannot run: {reason}"
    );
    eprintln!("SKIPPED regeneration check: {reason} (set {REQUIRE_ENV}=1 to make this fatal)");
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn committed_bindings_match_regenerated() {
    let Some(key) = host_key() else {
        return skip("host is not a supported bindings target");
    };
    let output = match generate(key, None, &[]) {
        Ok(output) => output,
        Err(e) => return skip(&format!("generation unavailable: {e}")),
    };
    let tmp = tempfile::tempdir().expect("temp dir");
    write(&output, tmp.path()).expect("write regenerated bindings");

    let committed = generated_dir();
    for (name, _) in &output.files {
        let rel = Path::new(key).join(name);
        assert!(
            read(&tmp.path().join(&rel)) == read(&committed.join(&rel)),
            "{} differs from the regenerated output; run `cargo xtask bindings`",
            rel.display()
        );
    }
    // Compare this target's meta entry (others come from other machines).
    let entry = |root: &Path| {
        let meta: serde_json::Value = serde_json::from_str(&read(&root.join("meta.json")))
            .expect("meta.json is valid JSON");
        meta["targets"][key].clone()
    };
    assert_eq!(
        entry(tmp.path()),
        entry(&committed),
        "meta.json entry for {key} differs; run `cargo xtask bindings`"
    );
}
