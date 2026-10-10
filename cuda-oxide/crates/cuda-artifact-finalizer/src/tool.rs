/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pinned CUDA Toolkit executables (`ptxas`, `fatbinary`).
//!
//! Discovery opens and hashes the selected executable once. Every later
//! invocation revalidates that file identity, so a toolkit replaced on disk
//! mid-build cannot produce output attributed to the digest Cargo saw. On
//! Linux a native ELF executable is run through the retained descriptor.

use crate::FinalizerError;
use crate::nvvm::report_changed_tool;
use crate::provenance::{ToolFileIdentity, digest_file_handle, with_revalidated_tool_identity};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;

/// How one toolkit executable is named, overridden, and recognized.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ToolSpec {
    /// Executable stem, also used in diagnostics.
    pub(crate) name: &'static str,
    /// Environment variable naming an explicit executable.
    pub(crate) env: &'static str,
    /// Lowercase text the `--version` banner must contain.
    pub(crate) banner: &'static str,
}

pub(crate) const PTXAS: ToolSpec = ToolSpec {
    name: "ptxas",
    env: "CUDA_OXIDE_PTXAS",
    banner: "ptx optimizing assembler",
};

pub(crate) const FATBINARY: ToolSpec = ToolSpec {
    name: "fatbinary",
    env: "CUDA_OXIDE_FATBINARY",
    banner: "cuda fat binary constructor",
};

impl ToolSpec {
    fn executable(&self) -> String {
        if cfg!(windows) {
            format!("{}.exe", self.name)
        } else {
            self.name.to_string()
        }
    }
}

static TEMP_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Private scratch directory, removed on drop, so concurrent tool runs never
/// share input or output files.
pub(crate) struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    pub(crate) fn new(prefix: &str) -> Result<Self, FinalizerError> {
        let root = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for _ in 0..128 {
            let sequence = TEMP_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let candidate = root.join(format!(
                "{prefix}-{}-{timestamp:x}-{sequence:x}",
                std::process::id()
            ));
            let builder = fs::DirBuilder::new();
            #[cfg(unix)]
            let mut builder = builder;
            #[cfg(unix)]
            builder.mode(0o700);
            match builder.create(&candidate) {
                Ok(()) => return Ok(Self { path: candidate }),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(source) => {
                    return Err(FinalizerError::Io {
                        path: candidate,
                        source,
                    });
                }
            }
        }
        Err(FinalizerError::Io {
            path: root,
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate a unique tool scratch directory",
            ),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// One discovered, hashed, and version-checked toolkit executable.
pub(crate) struct PinnedTool {
    spec: ToolSpec,
    path: PathBuf,
    file: File,
    identity: ToolFileIdentity,
    digest: [u8; 32],
    #[cfg(target_os = "linux")]
    execute_from_fd: bool,
}

impl PinnedTool {
    /// Search order is the spec's explicit variable, toolkit roots selected
    /// by `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, or `CUDA_PATH`, conventional
    /// toolkit roots, then `PATH`. An explicit executable that fails
    /// validation is an error rather than a reason to keep searching.
    pub(crate) fn discover(spec: ToolSpec) -> Result<Self, FinalizerError> {
        Self::discover_with_env(spec, |name| std::env::var_os(name))
    }

    /// [`Self::discover`] over an explicit environment, so a parent process
    /// can find the executable its child will find.
    pub(crate) fn discover_with_env(
        spec: ToolSpec,
        get_env: impl FnMut(&str) -> Option<OsString>,
    ) -> Result<Self, FinalizerError> {
        let (candidates, explicit) = tool_candidates(spec, get_env);
        let mut tried = Vec::new();
        let mut first_error = None;
        for (index, path) in candidates.into_iter().enumerate() {
            tried.push(path.display().to_string());
            if !path.is_file() {
                continue;
            }
            match Self::open(spec, path) {
                Ok(tool) => return Ok(tool),
                Err(error) if explicit && index == 0 => return Err(error),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Err(FinalizerError::ToolNotFound {
            tool: spec.name,
            env: spec.env,
            tried: tried.join("\n  "),
        })
    }

    pub(crate) fn open(spec: ToolSpec, path: PathBuf) -> Result<Self, FinalizerError> {
        let invalid = |path: &Path, details: &str| FinalizerError::InvalidTool {
            tool: spec.name,
            path: path.to_path_buf(),
            details: details.to_string(),
        };
        let file = File::open(&path).map_err(|source| FinalizerError::Io {
            path: path.clone(),
            source,
        })?;
        if !file
            .metadata()
            .map_err(|source| FinalizerError::Io {
                path: path.clone(),
                source,
            })?
            .is_file()
        {
            return Err(invalid(&path, "candidate is not a regular file"));
        }
        let identity = ToolFileIdentity::capture(&file)
            .ok_or_else(|| invalid(&path, "could not capture a stable file identity"))?;
        let digest = digest_file_handle(&file).map_err(|source| FinalizerError::Io {
            path: path.clone(),
            source,
        })?;

        #[cfg(target_os = "linux")]
        let execute_from_fd = {
            let mut magic = [0_u8; 4];
            file.read_exact_at(&mut magic, 0).is_ok() && magic == *b"\x7fELF"
        };

        let tool = Self {
            spec,
            path,
            file,
            identity,
            digest,
            #[cfg(target_os = "linux")]
            execute_from_fd,
        };
        tool.validate_version()?;
        Ok(tool)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Digest of the exact executable, if its identity still holds.
    pub(crate) fn digest(&self) -> Option<[u8; 32]> {
        let digest = self.current_digest();
        if digest.is_none() {
            report_changed_tool(self.spec.name);
        }
        digest
    }

    fn validate_version(&self) -> Result<(), FinalizerError> {
        let output = self.invoke([OsStr::new("--version")])?;
        let details = combined_diagnostics(&output);
        let recognized =
            output.status.success() && details.to_ascii_lowercase().contains(self.spec.banner);
        if recognized {
            Ok(())
        } else {
            Err(FinalizerError::InvalidTool {
                tool: self.spec.name,
                path: self.path.clone(),
                details: if details.is_empty() {
                    format!("version probe exited with {}", output.status)
                } else {
                    details
                },
            })
        }
    }

    /// Run the pinned executable, refusing output if its file changed.
    pub(crate) fn invoke<I, S>(&self, args: I) -> Result<Output, FinalizerError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let args = args
            .into_iter()
            .map(|arg| arg.as_ref().to_owned())
            .collect::<Vec<_>>();
        with_revalidated_tool_identity(
            self.spec.name,
            Some(self.digest),
            || self.current_digest(),
            || {
                let mut command = self.command();
                command.args(&args);
                run_tolerating_busy_text_file(&mut command).map_err(|source| FinalizerError::Io {
                    path: self.path.clone(),
                    source,
                })
            },
        )
    }

    /// Run with `args` and turn a non-zero exit into [`FinalizerError::ToolFailed`].
    /// Returns the combined diagnostics of a successful run.
    pub(crate) fn run(&self, args: &[OsString]) -> Result<String, FinalizerError> {
        let output = self.invoke(args)?;
        let diagnostics = combined_diagnostics(&output);
        if !output.status.success() {
            return Err(FinalizerError::ToolFailed {
                tool: self.spec.name,
                status: output.status.to_string(),
                diagnostics,
            });
        }
        Ok(diagnostics)
    }

    fn current_digest(&self) -> Option<[u8; 32]> {
        if !self.identity.matches_file(&self.file) {
            return None;
        }

        #[cfg(target_os = "linux")]
        if self.execute_from_fd {
            return Some(self.digest);
        }

        let current = File::open(&self.path).ok()?;
        self.identity.matches_file(&current).then_some(self.digest)
    }

    fn command(&self) -> Command {
        #[cfg(target_os = "linux")]
        if self.execute_from_fd {
            return Command::new(format!("/proc/self/fd/{}", self.file.as_raw_fd()));
        }

        Command::new(&self.path)
    }
}

/// Runs `command`, retrying briefly when the kernel reports ETXTBSY.
///
/// Executing a just-written tool by path can collide with an unrelated
/// `Command` spawn on another thread: the concurrent fork inherits the
/// writer's still-open descriptor for the moment before its own exec, and
/// exec of the tool during that moment fails with "Text file busy". The
/// descriptor vanishes as soon as that child execs, so a short bounded
/// retry rides out the collision while a persistent error still surfaces.
fn run_tolerating_busy_text_file(command: &mut Command) -> io::Result<Output> {
    let mut delay = Duration::from_millis(2);
    for _ in 0..8 {
        match command.output() {
            Err(error) if error.kind() == io::ErrorKind::ExecutableFileBusy => {
                thread::sleep(delay);
                delay = delay.saturating_mul(2);
            }
            result => return result,
        }
    }
    command.output()
}

pub(crate) fn tool_candidates(
    spec: ToolSpec,
    mut get_env: impl FnMut(&str) -> Option<OsString>,
) -> (Vec<PathBuf>, bool) {
    let mut candidates = Vec::new();
    let explicit = get_env(spec.env);
    if let Some(path) = explicit.as_ref() {
        push_unique(&mut candidates, PathBuf::from(path));
    }

    let executable = spec.executable();
    for variable in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = get_env(variable) {
            push_unique(
                &mut candidates,
                PathBuf::from(root).join("bin").join(&executable),
            );
        }
    }
    #[cfg(unix)]
    for root in ["/usr/local/cuda", "/opt/cuda"] {
        push_unique(
            &mut candidates,
            PathBuf::from(root).join("bin").join(&executable),
        );
    }
    if let Some(path) = get_env("PATH") {
        for directory in std::env::split_paths(&path) {
            push_unique(&mut candidates, directory.join(&executable));
        }
    }
    (candidates, explicit.is_some())
}

fn push_unique(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.contains(&candidate) {
        paths.push(candidate);
    }
}

pub(crate) fn combined_diagnostics(output: &Output) -> String {
    let mut diagnostics = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        if !diagnostics.is_empty() && !diagnostics.ends_with('\n') {
            diagnostics.push('\n');
        }
        diagnostics.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn discovery_order_is_explicit_tool_then_roots_then_path() {
        for spec in [PTXAS, FATBINARY] {
            let executable = spec.executable();
            let explicit_path = PathBuf::from("explicit").join(&executable);
            let search_path = std::env::join_paths(["first", "second"]).unwrap();
            let environment = HashMap::from([
                (spec.env, explicit_path.clone().into_os_string()),
                ("CUDA_TOOLKIT_PATH", OsString::from("toolkit")),
                ("CUDA_HOME", OsString::from("home")),
                ("CUDA_PATH", OsString::from("cuda-path")),
                ("PATH", search_path),
            ]);
            let (candidates, explicit) =
                tool_candidates(spec, |name| environment.get(name).cloned());
            assert!(explicit);
            assert_eq!(candidates[0], explicit_path);
            assert_eq!(
                candidates[1],
                PathBuf::from("toolkit").join("bin").join(&executable)
            );
            assert_eq!(
                candidates[2],
                PathBuf::from("home").join("bin").join(&executable)
            );
            assert_eq!(
                candidates[3],
                PathBuf::from("cuda-path").join("bin").join(&executable)
            );
            assert!(candidates.ends_with(&[
                PathBuf::from("first").join(&executable),
                PathBuf::from("second").join(&executable)
            ]));
        }
    }
}
