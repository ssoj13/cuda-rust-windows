/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ahead-of-time cubins for several GPU generations in one fat binary.
//!
//! The linked PTX of a crate is assembled once per requested architecture
//! with toolkit `ptxas` and the cubins are packed by toolkit `fatbinary`.
//! This is the compilation the driver performs when it JIT compiles that PTX
//! at load time, done ahead of time by the toolkit's assembler instead of the
//! driver's, with the crate's FMA and debug policy (`--fmad`, line info or
//! `--device-debug`) where the JIT uses its defaults.
//! The fat binary holds cubins only: the PTX stays a separate artifact payload
//! so a driver that cannot read the fat binary, or a GPU it has no cubin for,
//! still loads the PTX.

use crate::diagnostics::KernelResourceUsage;
use crate::link::logical_ptx;
use crate::provenance::StableDigest;
use crate::tool::{FATBINARY, PTXAS, PinnedTool, TemporaryDirectory};
use crate::{CudaArch, FinalizationOptions, FinalizerError, NamedInput, PtxAssembler};
use std::ffi::OsString;
use std::fs;
use std::sync::Arc;

/// `fatbinary` file magic (`0xBA55ED50`, little-endian) and header length.
const FATBIN_MAGIC: [u8; 4] = [0x50, 0xED, 0x55, 0xBA];
const FATBIN_HEADER_LENGTH: usize = 16;

/// A packed fat binary and the `ptxas` resource report of each cubin in it.
pub struct FatbinReport {
    /// Fat binary bytes, loadable with `cuModuleLoadData`.
    pub image: Vec<u8>,
    /// Per-architecture kernel resource usage, in the order of the cubins.
    pub resource_usage: Vec<(CudaArch, Vec<KernelResourceUsage>)>,
}

/// Pinned `ptxas` and `fatbinary` pair that turns one linked PTX module into
/// a fat binary of target-specific cubins.
#[derive(Clone)]
pub struct FatbinBuilder {
    assembler: PtxAssembler,
    fatbinary: Arc<PinnedTool>,
}

impl FatbinBuilder {
    /// Discover both tools; `fatbinary` follows the `ptxas` search order with
    /// `CUDA_OXIDE_FATBINARY` as its explicit override.
    pub fn discover() -> Result<Self, FinalizerError> {
        Self::discover_with_env(|name| std::env::var_os(name))
    }

    /// [`Self::discover`] over an explicit environment. cargo-oxide uses it to
    /// fingerprint the tools its backend child process will discover.
    pub fn discover_with_env(
        mut get_env: impl FnMut(&str) -> Option<OsString>,
    ) -> Result<Self, FinalizerError> {
        Ok(Self {
            assembler: PtxAssembler::from_tool(PinnedTool::discover_with_env(PTXAS, &mut get_env)?),
            fatbinary: Arc::new(PinnedTool::discover_with_env(FATBINARY, &mut get_env)?),
        })
    }

    /// Digest of both pinned executables, so build caches keyed on it are
    /// invalidated when the toolkit changes. `None` if either file changed
    /// since discovery.
    pub fn tool_digest(&self) -> Option<[u8; 32]> {
        let ptxas = self.assembler.ptxas_digest()?;
        let fatbinary = self.fatbinary.digest()?;
        Some(
            StableDigest::new()
                .field("ptxas-sha256", ptxas)
                .field("fatbinary-sha256", fatbinary)
                .finish(),
        )
    }

    /// Assemble `ptx` for every architecture in `targets` and pack the cubins.
    ///
    /// `options.target()` is the architecture the PTX was generated for; each
    /// requested architecture must be at least that one, because `ptxas`
    /// cannot lower PTX to an older GPU. FMA and debug policy apply to every
    /// cubin. Cubins are assembled concurrently, at most one `ptxas` process
    /// per available core.
    pub fn build(
        &self,
        ptx: NamedInput<'_>,
        options: &FinalizationOptions,
        targets: &[CudaArch],
    ) -> Result<FatbinReport, FinalizerError> {
        let targets = checked_targets(options.target(), targets)?;
        crate::validate_name(ptx.name)?;
        let logical = logical_ptx(ptx)?;
        let input = NamedInput::new(ptx.name, logical);

        // One ptxas process per cubin, at most one per available core: each
        // holds the whole module in memory.
        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        let mut reports = Vec::with_capacity(targets.len());
        for batch in targets.chunks(workers) {
            reports.extend(std::thread::scope(|scope| {
                let jobs = batch
                    .iter()
                    .map(|target| {
                        let options = FinalizationOptions::new(target.clone())
                            .with_fma_contraction(options.allow_fma_contraction())
                            .with_debug_policy(options.debug_policy());
                        scope
                            .spawn(move || self.assembler.assemble_ptx_with_report(input, &options))
                    })
                    .collect::<Vec<_>>();
                jobs.into_iter()
                    .map(|job| {
                        job.join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?);
        }

        let directory = TemporaryDirectory::new("cuda-oxide-fatbin")?;
        let output_path = directory.path().join("module.fatbin");
        let mut arguments = vec![
            OsString::from("--create"),
            output_path.as_os_str().to_owned(),
            OsString::from("-64"),
            OsString::from("--compress-all"),
        ];
        for (target, report) in targets.iter().zip(&reports) {
            let cubin_path = directory.path().join(format!("{}.cubin", target.sm()));
            fs::write(&cubin_path, &report.image).map_err(|source| FinalizerError::Io {
                path: cubin_path.clone(),
                source,
            })?;
            let mut image =
                OsString::from(format!("--image3=kind=elf,sm={},file=", sm_spec(target)));
            image.push(cubin_path.as_os_str());
            arguments.push(image);
        }
        self.fatbinary.run(&arguments)?;

        let image = fs::read(&output_path).map_err(|source| FinalizerError::Io {
            path: output_path,
            source,
        })?;
        if !is_valid_fatbin(&image) {
            return Err(FinalizerError::InvalidFatbin);
        }
        Ok(FatbinReport {
            image,
            resource_usage: targets
                .into_iter()
                .zip(reports)
                .map(|(target, report)| (target, report.resource_usage))
                .collect(),
        })
    }
}

/// Sorted, de-duplicated targets, each at least the PTX target.
fn checked_targets(
    ptx_target: &CudaArch,
    targets: &[CudaArch],
) -> Result<Vec<CudaArch>, FinalizerError> {
    let mut checked = Vec::with_capacity(targets.len());
    for target in targets {
        if target.capability() < ptx_target.capability() {
            return Err(FinalizerError::FatbinTargetBelowPtx {
                requested: target.sm(),
                ptx_target: ptx_target.sm(),
            });
        }
        if !checked.contains(target) {
            checked.push(target.clone());
        }
    }
    if checked.is_empty() {
        return Err(FinalizerError::NoFatbinTargets);
    }
    checked.sort_by_key(|target| (target.capability(), target.suffix()));
    Ok(checked)
}

/// `fatbinary --image3` spells an architecture without the `sm_` prefix.
fn sm_spec(target: &CudaArch) -> String {
    target.sm().trim_start_matches("sm_").to_string()
}

/// Whether `bytes` start with a complete `fatbinary` header whose declared
/// payload fits in the buffer.
pub fn is_valid_fatbin(bytes: &[u8]) -> bool {
    let Some(header) = bytes.get(..FATBIN_HEADER_LENGTH) else {
        return false;
    };
    let mut size = [0_u8; 8];
    size.copy_from_slice(&header[8..16]);
    header[..4] == FATBIN_MAGIC
        && usize::try_from(u64::from_le_bytes(size))
            .ok()
            .and_then(|size| size.checked_add(FATBIN_HEADER_LENGTH))
            .is_some_and(|end| end <= bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch(name: &str) -> CudaArch {
        name.parse().unwrap()
    }

    #[test]
    fn targets_are_sorted_deduplicated_and_never_below_the_ptx_target() {
        let checked = checked_targets(
            &arch("sm_75"),
            &[arch("sm_120"), arch("sm_75"), arch("sm_86"), arch("sm_86")],
        )
        .unwrap();
        assert_eq!(checked, [arch("sm_75"), arch("sm_86"), arch("sm_120")]);
        assert!(matches!(
            checked_targets(&arch("sm_86"), &[arch("sm_75")]),
            Err(FinalizerError::FatbinTargetBelowPtx { requested, ptx_target })
                if requested == "sm_75" && ptx_target == "sm_86"
        ));
        assert!(matches!(
            checked_targets(&arch("sm_75"), &[]),
            Err(FinalizerError::NoFatbinTargets)
        ));
    }

    #[test]
    fn image_spec_drops_the_sm_prefix_and_keeps_the_suffix() {
        assert_eq!(sm_spec(&arch("sm_86")), "86");
        assert_eq!(sm_spec(&arch("sm_90a")), "90a");
    }

    #[test]
    fn fatbin_validation_checks_magic_and_declared_size() {
        let mut bytes = vec![0_u8; FATBIN_HEADER_LENGTH + 4];
        bytes[..4].copy_from_slice(&FATBIN_MAGIC);
        bytes[4..6].copy_from_slice(&1_u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&(FATBIN_HEADER_LENGTH as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&4_u64.to_le_bytes());
        assert!(is_valid_fatbin(&bytes));
        assert!(!is_valid_fatbin(&bytes[..bytes.len() - 1]));
        assert!(!is_valid_fatbin(&bytes[..8]));
        bytes[0] = 0;
        assert!(!is_valid_fatbin(&bytes));
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    const PTX: &[u8] = br#"
.version 8.0
.target sm_75
.address_size 64

.visible .entry kernel() {
    ret;
}
"#;

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit ptxas and fatbinary"]
    fn live_fatbin_packs_one_cubin_per_target() {
        let builder = FatbinBuilder::discover().unwrap();
        let options = FinalizationOptions::new("sm_75".parse().unwrap());
        let targets = ["sm_75", "sm_86"].map(|name| name.parse().unwrap());
        let report = builder
            .build(NamedInput::new("kernel.ptx", PTX), &options, &targets)
            .unwrap();
        assert!(is_valid_fatbin(&report.image));
        assert_eq!(report.resource_usage.len(), 2);
    }
}
