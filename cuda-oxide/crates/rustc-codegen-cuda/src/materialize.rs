/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Strict, opt-in build-time finalization of embedded device artifacts.
//!
//! `cargo oxide --materialize-cubin` discovers and fingerprints the exact
//! libNVVM, nvJitLink, and libdevice inputs before invoking Cargo. Device
//! macros record the complete codegen identity and exact provenance as Cargo
//! environment dependencies. We reopen the tools, bind the parent digest to
//! matching retained-file identities, and revalidate those identities around
//! compilation. Setting the internal opt-in around raw Cargo is unsupported:
//! Cargo can reuse an existing artifact without invoking this backend, and
//! when the backend does run it rejects a missing handshake.
//!
//! The second build-time image is a fat binary (`CUDA_OXIDE_CUBIN_ARCHS`):
//! the PTX that the default route already embeds is assembled with toolkit
//! `ptxas` for each listed GPU generation and packed with `fatbinary`. It is
//! embedded in front of the unchanged PTX payload, so a loader falls back to
//! the PTX whenever the driver cannot use the fat binary.

use cuda_artifact_finalizer::{
    CudaArch, CudaArchParseError, DebugPolicy, FatbinBuilder, FatbinReport, FinalizationOptions,
    Finalizer, FinalizerError, FinalizerOutput, KernelResourceUsage, NamedInput,
};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

pub(crate) const MATERIALIZE_ENV: &str = reserved_oxide_symbols::MATERIALIZE_CUBIN_ENV;
pub(crate) const EXPECTED_PROVENANCE_ENV: &str =
    reserved_oxide_symbols::MATERIALIZER_PROVENANCE_ENV;
pub(crate) const MATERIALIZER_HANDSHAKE_ENV: &str =
    reserved_oxide_symbols::MATERIALIZER_HANDSHAKE_ENV;
pub(crate) const CODEGEN_FINGERPRINT_ENV: &str = reserved_oxide_symbols::CODEGEN_FINGERPRINT_ENV;
pub(crate) const CUBIN_ARCHS_ENV: &str = reserved_oxide_symbols::CUBIN_ARCHS_ENV;
/// Explicit per-assembler optimization-thread budget; zero/all-CPU mode is unsupported.
pub(crate) const PTXAS_SPLIT_COMPILE_ENV: &str = "CUDA_OXIDE_PTXAS_SPLIT_COMPILE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaterializationRequest {
    expected_provenance: [u8; 32],
    tool_identity_handshake: cuda_artifact_finalizer::MaterializerHandshakeV1,
}

/// Cubin bytes plus ptxas resource diagnostics from the final link.
pub(crate) struct MaterializedCubin {
    pub(crate) bytes: Vec<u8>,
    pub(crate) resource_usage: Vec<KernelResourceUsage>,
}

/// Failures in the wrapper/backend materialization contract.
#[derive(Debug, Error)]
pub(crate) enum MaterializeError {
    #[error(
        "{MATERIALIZE_ENV} must be a boolean (accepted true values: 1, true, yes, on; false values: 0, false, no, off), got {value:?}"
    )]
    InvalidBoolean { value: String },

    #[error("{MATERIALIZE_ENV} is not valid Unicode")]
    NonUnicodeBoolean,

    #[error(
        "{MATERIALIZE_ENV}=true requires cargo-oxide's provenance handshake; use `cargo oxide build --materialize-cubin` instead of invoking raw Cargo"
    )]
    MissingExpectedProvenance,

    #[error(
        "{MATERIALIZE_ENV}=true is missing cargo-oxide's tracked codegen fingerprint; use `cargo oxide build --materialize-cubin` instead of invoking raw Cargo"
    )]
    MissingCargoFingerprint,

    #[error(
        "{MATERIALIZE_ENV}=true requires cargo-oxide's named v1 tool-identity handshake in {MATERIALIZER_HANDSHAKE_ENV}"
    )]
    MissingToolIdentityHandshake,

    #[error("{MATERIALIZER_HANDSHAKE_ENV} is not valid Unicode")]
    NonUnicodeToolIdentityHandshake,

    #[error("{MATERIALIZER_HANDSHAKE_ENV} is not a valid named v1 handshake: {reason}")]
    InvalidToolIdentityHandshake { reason: String },

    #[error(
        "cargo-oxide's tracked codegen fingerprint must be exactly 64 lowercase hexadecimal characters, got {value:?}"
    )]
    InvalidCargoFingerprint { value: String },

    #[error(
        "{EXPECTED_PROVENANCE_ENV} must be exactly 64 lowercase hexadecimal characters, got {value:?}"
    )]
    InvalidExpectedProvenance { value: String },

    #[error(
        "the loaded CUDA tools cannot be tied to exact files, so their provenance cannot be verified; refusing build-time cubin materialization"
    )]
    UnverifiableProvenance,

    #[error(
        "CUDA materializer provenance changed after Cargo fingerprinting (expected {expected}, loaded {actual}); rerun `cargo oxide build --materialize-cubin`"
    )]
    ProvenanceMismatch { expected: String, actual: String },

    #[error(
        "build-time device images (--materialize-cubin, cubin-archs) do not yet support generic #[cuda_module] loading because it merges PTX bundles across crates at run time"
    )]
    RequiresPtxBundleMerge,

    #[error(
        "build-time device images (--materialize-cubin, cubin-archs) do not yet support #[device] extern declarations because their ordered external link inputs are not available to the backend"
    )]
    HasDeviceExterns,

    #[error(
        "build-time cubin materialization requires an NVVM IR or LTOIR artifact, but codegen produced PTX; use cargo-oxide so materialization can force NVVM IR emission"
    )]
    PtxInput,

    #[error(
        "build-time cubin materialization expected compiler IR, but codegen already produced a cubin; refusing to bypass the provenance-checked finalization recipe"
    )]
    CubinInput,

    #[error("{CUBIN_ARCHS_ENV} is not valid Unicode")]
    NonUnicodeCubinArchs,

    #[error("{CUBIN_ARCHS_ENV} names an invalid architecture {value:?}: {source}")]
    InvalidCubinArch {
        value: String,
        source: CudaArchParseError,
    },

    #[error(
        "{MATERIALIZE_ENV} and {CUBIN_ARCHS_ENV} both select the embedded device image; use one of them"
    )]
    ConflictingDeviceImages,

    #[error(
        "{CUBIN_ARCHS_ENV} assembles cubins from PTX, but codegen produced {produced}; a fat binary needs the PTX route"
    )]
    FatbinNeedsPtx { produced: &'static str },

    #[error(
        "{PTXAS_SPLIT_COMPILE_ENV} must be a positive u32 thread limit (default 1), got {value:?}"
    )]
    InvalidPtxasSplitCompile { value: String },

    #[error("{PTXAS_SPLIT_COMPILE_ENV} is not valid Unicode")]
    NonUnicodePtxasSplitCompile,

    #[error(transparent)]
    InvalidTarget(#[from] CudaArchParseError),

    #[error(transparent)]
    Finalizer(#[from] FinalizerError),
}

/// Parse the strict opt-in and its wrapper-generated provenance handshake.
/// No CUDA library is loaded here.
pub(crate) fn request_from_env() -> Result<Option<MaterializationRequest>, MaterializeError> {
    let enabled = match std::env::var(MATERIALIZE_ENV) {
        Ok(value) => parse_bool(&value)?,
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(MaterializeError::NonUnicodeBoolean);
        }
    };
    if !enabled {
        return Ok(None);
    }
    let value = std::env::var(EXPECTED_PROVENANCE_ENV)
        .map_err(|_| MaterializeError::MissingExpectedProvenance)?;
    let expected_provenance = parse_digest(&value)?;
    let handshake_json =
        std::env::var(MATERIALIZER_HANDSHAKE_ENV).map_err(|error| match error {
            std::env::VarError::NotPresent => MaterializeError::MissingToolIdentityHandshake,
            std::env::VarError::NotUnicode(_) => MaterializeError::NonUnicodeToolIdentityHandshake,
        })?;
    let handshake: cuda_artifact_finalizer::MaterializerHandshakeV1 =
        serde_json::from_str(&handshake_json).map_err(|error| {
            MaterializeError::InvalidToolIdentityHandshake {
                reason: error.to_string(),
            }
        })?;
    validate_tool_identity_handshake(expected_provenance, &handshake)?;
    validate_codegen_fingerprint()?;
    Ok(Some(MaterializationRequest {
        expected_provenance,
        tool_identity_handshake: handshake,
    }))
}

fn validate_tool_identity_handshake(
    expected_provenance: [u8; 32],
    handshake: &cuda_artifact_finalizer::MaterializerHandshakeV1,
) -> Result<(), MaterializeError> {
    if !handshake.has_consistent_provenance() || handshake.provenance_sha256 != expected_provenance
    {
        return Err(MaterializeError::InvalidToolIdentityHandshake {
            reason: format!(
                "version {} and provenance {} do not match expected v1 provenance {}",
                handshake.version,
                digest_hex(&handshake.provenance_sha256),
                digest_hex(&expected_provenance),
            ),
        });
    }
    Ok(())
}

/// Parse [`CUBIN_ARCHS_ENV`]: the architectures whose cubins are packed into a
/// fat binary ahead of the PTX payload. Unset or blank means none.
pub(crate) fn fatbin_targets_from_env() -> Result<Option<Vec<CudaArch>>, MaterializeError> {
    // Validate the budget before expensive device codegen starts.
    ptxas_split_compile_from_env()?;
    match std::env::var(CUBIN_ARCHS_ENV) {
        Ok(value) => parse_fatbin_targets(&value),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MaterializeError::NonUnicodeCubinArchs),
    }
}

fn parse_fatbin_targets(value: &str) -> Result<Option<Vec<CudaArch>>, MaterializeError> {
    let targets = value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| {
            name.parse::<CudaArch>()
                .map_err(|source| MaterializeError::InvalidCubinArch {
                    value: name.to_string(),
                    source,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((!targets.is_empty()).then_some(targets))
}

/// Reject artifact-loading models that a build-time device image (a
/// materialized cubin or a fat binary) cannot reproduce, before any CUDA
/// compiler library is discovered or loaded.
pub(crate) fn validate_collection(
    builds_device_image: bool,
    has_device_externs: bool,
    requires_ptx_bundle_merge: bool,
) -> Result<(), MaterializeError> {
    if !builds_device_image {
        return Ok(());
    }
    if requires_ptx_bundle_merge {
        return Err(MaterializeError::RequiresPtxBundleMerge);
    }
    if has_device_externs {
        return Err(MaterializeError::HasDeviceExterns);
    }
    Ok(())
}

pub(crate) fn nvvm_ir_to_cubin(
    request: MaterializationRequest,
    nvvm_ir: &[u8],
    module_name: &str,
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<MaterializedCubin, MaterializeError> {
    let options = options(target, allow_fma_contraction, debug_policy)?;
    let finalizer = checked_finalizer(request)?;
    let report = finalizer.materialize_nvvm_ir_with_report(module_name, nvvm_ir, &options)?;
    Ok(MaterializedCubin {
        bytes: report.image,
        resource_usage: report.resource_usage,
    })
}

pub(crate) fn ltoir_to_cubin(
    request: MaterializationRequest,
    ltoir: &[u8],
    module_name: &str,
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<MaterializedCubin, MaterializeError> {
    let options = options(target, allow_fma_contraction, debug_policy)?;
    let finalizer = checked_finalizer(request)?;
    let report = finalizer.link_ltoir_with_report(
        &[NamedInput::new(module_name, ltoir)],
        &options,
        FinalizerOutput::Cubin,
    )?;
    Ok(MaterializedCubin {
        bytes: report.image,
        resource_usage: report.resource_usage,
    })
}

/// Assemble the linked PTX for every requested architecture and pack the
/// cubins into one fat binary. `target` is the architecture the PTX was
/// generated for.
pub(crate) fn ptx_to_fatbin(
    ptx: &[u8],
    module_name: &str,
    target: &str,
    targets: &[CudaArch],
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<FatbinReport, MaterializeError> {
    let options = options(target, allow_fma_contraction, debug_policy)?
        .with_ptxas_split_compile(ptxas_split_compile_from_env()?);
    let builder = FatbinBuilder::discover()?;
    Ok(builder.build(NamedInput::new(module_name, ptx), &options, targets)?)
}

fn ptxas_split_compile_from_env() -> Result<NonZeroU32, MaterializeError> {
    match std::env::var(PTXAS_SPLIT_COMPILE_ENV) {
        Ok(value) => parse_ptxas_split_compile(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_ptxas_split_compile(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(MaterializeError::NonUnicodePtxasSplitCompile)
        }
    }
}

fn parse_ptxas_split_compile(value: Option<&str>) -> Result<NonZeroU32, MaterializeError> {
    match value {
        None => Ok(NonZeroU32::MIN),
        Some(value) => {
            value
                .trim()
                .parse()
                .map_err(|_| MaterializeError::InvalidPtxasSplitCompile {
                    value: value.to_string(),
                })
        }
    }
}

/// Publish the exact assembly receipt without changing the embedded runtime bundle.
///
/// Resource counters are deliberately left to consumers of the raw log: the legacy
/// parsed rows do not distinguish missing diagnostics from reported zeros. Unique
/// create-new paths prevent concurrent compiler processes from replacing a receipt.
pub(crate) fn write_ptxas_reports(
    output_dir: &Path,
    output_name: &str,
    host_target: &str,
    ptx_name: &str,
    entries: &[String],
    report: &FatbinReport,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    static NEXT_REPORT: AtomicU64 = AtomicU64::new(0);
    std::fs::create_dir_all(output_dir)?;
    let stem = format!(
        "{}.{}.{}",
        crate::sanitize_path_component(output_name),
        crate::sanitize_path_component(host_target),
        std::process::id(),
    );
    let directory = allocate_report_directory(output_dir, &stem, &NEXT_REPORT)?;
    let mut paths = Vec::with_capacity(report.ptxas_reports.len());
    for assembly in &report.ptxas_reports {
        let log = assembly.info_log.as_deref().ok_or_else(|| {
            std::io::Error::other(format!(
                "ptxas {} did not retain its raw diagnostics",
                assembly.target.sm()
            ))
        })?;
        let prefix = assembly.target.sm();
        let raw_path = directory.join(format!("{prefix}.ptxas.log"));
        let json_path = directory.join(format!("{prefix}.ptxas.json"));
        let metadata = serde_json::json!({
            "schema": 1,
            "kind": "ptxas-assembly-receipt",
            "module": output_name,
            "host_target": host_target,
            "arch": assembly.target.sm(),
            "ptx_name": ptx_name,
            "ptx_sha256": digest_hex(&report.ptx_sha256),
            "ptxas_path": report.ptxas_path.to_string_lossy(),
            "ptxas_sha256": digest_hex(&report.ptxas_sha256),
            "options": assembly.options,
            "entries": entries,
            "raw_log": raw_path.file_name().unwrap().to_string_lossy(),
            "assembly_succeeded": true,
            "resource_coverage_complete": null,
        });
        use std::io::Write as _;
        {
            let mut raw = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&raw_path)?;
            raw.write_all(log.as_bytes())?;
            raw.sync_all()?;
        }
        // The final JSON is the commit marker. Its sibling raw log is complete
        // and synced before a reader can discover this final path.
        publish_report_json(&json_path, |json| {
            serde_json::to_writer_pretty(&mut *json, &metadata).map_err(std::io::Error::other)?;
            json.write_all(b"\n")
        })?;
        paths.push(json_path);
    }
    Ok(paths)
}

/// Claim an exclusively owned directory, skipping stale names after PID reuse.
fn allocate_report_directory(
    output_dir: &Path,
    stem: &str,
    sequence: &AtomicU64,
) -> std::io::Result<PathBuf> {
    loop {
        let id = sequence.fetch_add(1, Ordering::Relaxed);
        let directory = output_dir.join(format!("{stem}.{id}.ptxas-report"));
        match std::fs::create_dir(&directory) {
            Ok(()) => return Ok(directory),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

/// Stage and sync JSON before an atomic rename inside an exclusively owned directory.
fn publish_report_json(
    path: &Path,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    // Each allocated directory has one writer. A repeated publication is an
    // error, rather than a replacement of an already accepted receipt.
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                path.display().to_string(),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let temporary = path.with_extension("json.tmp");
    let mut created = false;
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        created = true;
        write(&mut file)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() && created {
        // Keep complete raw diagnostics for inspection; remove partial JSON.
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn options(
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<FinalizationOptions, MaterializeError> {
    let target: CudaArch = target.parse()?;
    Ok(FinalizationOptions::new(target)
        .with_fma_contraction(allow_fma_contraction)
        .with_debug_policy(debug_policy))
}

fn checked_finalizer(request: MaterializationRequest) -> Result<Finalizer, MaterializeError> {
    let finalizer = Finalizer::discover_with_handshake(&request.tool_identity_handshake)?;
    let actual = finalizer
        .provenance_digest()
        .ok_or(MaterializeError::UnverifiableProvenance)?;
    if actual != request.expected_provenance {
        return Err(MaterializeError::ProvenanceMismatch {
            expected: digest_hex(&request.expected_provenance),
            actual: digest_hex(&actual),
        });
    }
    Ok(finalizer)
}

fn parse_bool(value: &str) -> Result<bool, MaterializeError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(MaterializeError::InvalidBoolean {
            value: value.to_string(),
        }),
    }
}

fn validate_codegen_fingerprint() -> Result<(), MaterializeError> {
    let value = std::env::var(CODEGEN_FINGERPRINT_ENV).ok();
    validate_codegen_fingerprint_value(value.as_deref())
}

fn validate_codegen_fingerprint_value(value: Option<&str>) -> Result<(), MaterializeError> {
    let value = value.ok_or(MaterializeError::MissingCargoFingerprint)?;
    parse_digest(value)
        .map(|_| ())
        .map_err(|_| MaterializeError::InvalidCargoFingerprint {
            value: value.to_string(),
        })
}

fn parse_digest(value: &str) -> Result<[u8; 32], MaterializeError> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(MaterializeError::InvalidExpectedProvenance {
            value: value.to_string(),
        });
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16).map_err(|_| {
            MaterializeError::InvalidExpectedProvenance {
                value: value.to_string(),
            }
        })?;
    }
    Ok(digest)
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_tool_file_identity() -> cuda_artifact_finalizer::ToolFileIdentity {
        cuda_artifact_finalizer::ToolFileIdentity {
            length: 0,
            modified_seconds: 0,
            modified_nanoseconds: 0,
            device: None,
            inode: None,
            change_time_seconds: None,
            change_time_nanoseconds: None,
        }
    }

    fn test_handshake() -> cuda_artifact_finalizer::MaterializerHandshakeV1 {
        let file = empty_tool_file_identity();
        cuda_artifact_finalizer::MaterializerHandshakeV1::new(
            cuda_artifact_finalizer::PinnedToolProvenance {
                sha256: [1; 32],
                file,
            },
            cuda_artifact_finalizer::PinnedToolProvenance {
                sha256: [2; 32],
                file,
            },
            [3; 32],
        )
    }

    #[test]
    fn strict_boolean_parser_accepts_only_documented_values() {
        for value in ["1", " true ", "YES", "on"] {
            assert!(parse_bool(value).unwrap());
        }
        for value in ["0", " false ", "NO", "off"] {
            assert!(!parse_bool(value).unwrap());
        }
        for value in ["", "enabled", "2", "truthy"] {
            assert!(matches!(
                parse_bool(value),
                Err(MaterializeError::InvalidBoolean { .. })
            ));
        }
    }

    #[test]
    fn provenance_digest_requires_canonical_lower_hex() {
        let value = "0123456789abcdef".repeat(4);
        let digest = parse_digest(&value).unwrap();
        assert_eq!(digest_hex(&digest), value);
        assert!(parse_digest(&"A".repeat(64)).is_err());
        assert!(parse_digest(&"0".repeat(63)).is_err());
        assert!(parse_digest(&format!("{}g", "0".repeat(63))).is_err());
    }

    #[test]
    fn tool_identity_handshake_fails_closed_on_version_or_provenance_mismatch() {
        let mut handshake = test_handshake();
        let expected = handshake.provenance_sha256;
        assert!(validate_tool_identity_handshake(expected, &handshake).is_ok());

        handshake.version += 1;
        assert!(matches!(
            validate_tool_identity_handshake(expected, &handshake),
            Err(MaterializeError::InvalidToolIdentityHandshake { .. })
        ));
        handshake.version = cuda_artifact_finalizer::MaterializerHandshakeV1::VERSION;
        assert!(matches!(
            validate_tool_identity_handshake([8; 32], &handshake),
            Err(MaterializeError::InvalidToolIdentityHandshake { .. })
        ));
    }

    #[test]
    fn unsupported_collection_models_fail_without_tools() {
        assert!(matches!(
            validate_collection(true, false, true),
            Err(MaterializeError::RequiresPtxBundleMerge)
        ));
        assert!(matches!(
            validate_collection(true, true, false),
            Err(MaterializeError::HasDeviceExterns)
        ));
        assert!(validate_collection(false, true, true).is_ok());
    }

    #[test]
    fn cubin_arch_list_is_comma_separated_and_strict() {
        let targets = parse_fatbin_targets(" sm_75, sm_86 ,,sm_120 ")
            .unwrap()
            .unwrap();
        assert_eq!(
            targets.iter().map(CudaArch::sm).collect::<Vec<_>>(),
            ["sm_75", "sm_86", "sm_120"]
        );
        assert!(parse_fatbin_targets(" , ").unwrap().is_none());
        assert!(matches!(
            parse_fatbin_targets("sm_86,gfx90a"),
            Err(MaterializeError::InvalidCubinArch { value, .. }) if value == "gfx90a"
        ));
    }

    #[test]
    fn backend_invocation_rejects_raw_cargo_materialization_without_fingerprint() {
        assert!(matches!(
            validate_codegen_fingerprint_value(None),
            Err(MaterializeError::MissingCargoFingerprint)
        ));
        assert!(validate_codegen_fingerprint_value(Some(&"00".repeat(32))).is_ok());
        assert!(matches!(
            validate_codegen_fingerprint_value(Some("not-a-digest")),
            Err(MaterializeError::InvalidCargoFingerprint { .. })
        ));
    }
    #[test]
    fn explicit_ptxas_thread_budget_is_positive_and_defaults_to_serial() {
        assert_eq!(parse_ptxas_split_compile(None).unwrap().get(), 1);
        assert_eq!(parse_ptxas_split_compile(Some(" 4 ")).unwrap().get(), 4);
        for value in ["", "0", "-1", "all", "4294967296"] {
            assert!(matches!(
                parse_ptxas_split_compile(Some(value)),
                Err(MaterializeError::InvalidPtxasSplitCompile { .. })
            ));
        }
    }

    #[test]
    fn receipt_directory_allocation_survives_a_reused_process_sequence() {
        let root = std::env::temp_dir().join(format!(
            "oxide-report-collision-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir(&root).unwrap();
        let before_restart = AtomicU64::new(0);
        let original = allocate_report_directory(&root, "same-pid", &before_restart).unwrap();
        std::fs::write(original.join("preserved"), b"old receipt").unwrap();
        let after_restart = AtomicU64::new(0);
        let fresh = allocate_report_directory(&root, "same-pid", &after_restart).unwrap();
        assert_ne!(original, fresh);
        assert_eq!(
            std::fs::read(original.join("preserved")).unwrap(),
            b"old receipt"
        );
        assert_eq!(root.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn receipt_json_is_published_only_after_a_complete_successful_write() {
        use std::io::Write as _;
        let root = std::env::temp_dir().join(format!(
            "oxide-report-publication-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("receipt.json");
        let failed = publish_report_json(&path, |file| {
            assert!(
                !path.exists(),
                "partial JSON must not be a visible commit marker"
            );
            file.write_all(b"{partial")?;
            Err(std::io::Error::other("injected writer failure"))
        });
        assert!(failed.is_err());
        assert!(!path.exists());
        assert!(!path.with_extension("json.tmp").exists());
        publish_report_json(&path, |file| {
            assert!(!path.exists());
            file.write_all(b"{\"complete\":true}")
        })
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"{\"complete\":true}");
        assert!(matches!(
            publish_report_json(&path, |_| Ok(())).unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(std::fs::read(&path).unwrap(), b"{\"complete\":true}");
        assert_eq!(root.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "requires CUDA Toolkit ptxas/fatbinary and CUDA_OXIDE_PTXAS_SPLIT_COMPILE=4"]
    fn live_split_compile_receipt_retains_entry_and_helper_diagnostics() {
        const PTX: &[u8] = br#"
.version 8.0
.target sm_86
.address_size 64

.visible .func (.param .b32 result) smoke_helper(.param .b32 input) {
    .reg .b32 %x;
    ld.param.b32 %x, [input];
    add.u32 %x, %x, 1;
    st.param.b32 [result], %x;
    ret;
}
.visible .entry smoke_entry(.param .u64 output) {
    .reg .b64 %rd;
    .reg .b32 %x;
    .param .b32 argument;
    .param .b32 result;
    ld.param.u64 %rd, [output];
    mov.u32 %x, %tid.x;
    st.param.b32 [argument], %x;
    call.uni (result), smoke_helper, (argument);
    ld.param.b32 %x, [result];
    st.global.b32 [%rd], %x;
    ret;
}
"#;
        assert_eq!(ptxas_split_compile_from_env().unwrap().get(), 4);
        let report = ptx_to_fatbin(
            PTX,
            "split-smoke.ptx",
            "sm_86",
            &["sm_86".parse().unwrap()],
            true,
            DebugPolicy::None,
        )
        .unwrap();
        assert_eq!(report.ptxas_reports.len(), 1);
        let assembly = &report.ptxas_reports[0];
        assert_eq!(
            assembly.options,
            [
                "--gpu-name=sm_86",
                "--fmad=true",
                "--split-compile=4",
                "--verbose"
            ]
        );
        let log = assembly.info_log.as_deref().unwrap();
        assert!(log.contains("Function properties for smoke_entry"), "{log}");
        assert!(
            log.contains("Function properties for smoke_helper"),
            "{log}"
        );
        let output = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/ptxas-report-smoke");
        let paths = write_ptxas_reports(
            &output,
            "split-smoke",
            "live-toolchain",
            "split-smoke.ptx",
            &["smoke_entry".into()],
            &report,
        )
        .unwrap();
        assert_eq!(paths.len(), 1);
        // Retain the exact fixture beside its receipt for independent hash checks.
        std::fs::write(paths[0].parent().unwrap().join("split-smoke.ptx"), PTX).unwrap();
        println!("LIVE_PTXAS_RECEIPT {}", paths[0].display());
    }

    #[test]
    fn raw_resource_receipt_preserves_unknown_counters_and_exact_tool_identity() {
        let directory = std::env::temp_dir().join(format!(
            "oxide-ptxas-report-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let log = "ptxas info : Function properties for helper\n    8 bytes spill stores\n";
        let report = FatbinReport {
            image: vec![],
            resource_usage: vec![],
            ptxas_reports: vec![cuda_artifact_finalizer::PtxasReport {
                target: "sm_86".parse().unwrap(),
                options: vec![
                    "--gpu-name=sm_86".into(),
                    "--split-compile=4".into(),
                    "--verbose".into(),
                ],
                info_log: Some(log.into()),
            }],
            ptx_sha256: [0x12; 32],
            ptxas_path: PathBuf::from("toolkit/bin/ptxas"),
            ptxas_sha256: [0x34; 32],
        };
        let entries = vec!["kernel".into()];
        let paths = write_ptxas_reports(
            &directory,
            "fixture",
            "host",
            "fixture.ptx",
            &entries,
            &report,
        )
        .unwrap();
        assert_eq!(paths.len(), 1);
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&paths[0]).unwrap()).unwrap();
        assert_eq!(metadata["ptx_sha256"], "12".repeat(32));
        assert_eq!(metadata["ptxas_sha256"], "34".repeat(32));
        assert_eq!(metadata["entries"], serde_json::json!(["kernel"]));
        assert!(metadata["resource_coverage_complete"].is_null());
        assert!(metadata.get("spill_load_bytes").is_none());
        let raw = paths[0]
            .parent()
            .unwrap()
            .join(metadata["raw_log"].as_str().unwrap());
        assert_eq!(std::fs::read_to_string(raw).unwrap(), log);
        // Publication does not overwrite the previous receipt on a later invocation.
        let again = write_ptxas_reports(
            &directory,
            "fixture",
            "host",
            "fixture.ptx",
            &entries,
            &report,
        )
        .unwrap();
        assert_ne!(paths, again);
        assert_eq!(directory.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
