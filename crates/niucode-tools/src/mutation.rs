//! Optimistic file versions and atomic replacement. External editors do not
//! participate in our lock: a final-check/rename race remains explicit.
use crate::{input_string, ChangePreview, FilePreview, SandboxMode, Toolbox};
use anyhow::{bail, Context, Result};
use niucode_core::FileChange;
use sha2::{Digest, Sha256};
use similar::TextDiff;
use std::collections::HashMap;
use std::fs::{self, Metadata};
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct Conflict(pub String);
impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "file conflict: {}; read again before retrying", self.0)
    }
}
impl std::error::Error for Conflict {}

#[derive(Clone, PartialEq, Eq)]
struct Version {
    digest: String,
    identity: (u64, u64),
    mode: u32,
}
impl Version {
    fn new(text: &str, metadata: &Metadata) -> Self {
        #[cfg(unix)]
        let (identity, mode) = {
            use std::os::unix::fs::MetadataExt;
            ((metadata.dev(), metadata.ino()), metadata.mode())
        };
        #[cfg(not(unix))]
        let (identity, mode) = (
            (0, metadata.len()),
            u32::from(metadata.permissions().readonly()),
        );
        Self {
            digest: digest(text.as_bytes()),
            identity,
            mode,
        }
    }
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn match_line_numbers(text: &str, matches: &[(usize, &str)]) -> String {
    let mut nums: Vec<String> = matches
        .iter()
        .take(5)
        .map(|(offset, _)| {
            let line = text[..*offset].bytes().filter(|&b| b == b'\n').count() + 1;
            line.to_string()
        })
        .collect();
    if matches.len() > 5 {
        nums.push("...".into());
    }
    nums.join(", ")
}

fn diagnose_missing_old(old: &str, from: &str) -> Option<String> {
    let normalized_old = old.replace("\r\n", "\n");
    let normalized_from = from.replace("\r\n", "\n");
    if normalized_old != old || normalized_from != from {
        if normalized_old.contains(&normalized_from) {
            return Some("hint: line endings differ between CRLF and LF".into());
        }
    }

    let old_lines: Vec<&str> = old.lines().collect();
    let from_lines: Vec<&str> = from.lines().collect();
    if !from_lines.is_empty() && from_lines.len() <= old_lines.len() {
        for start in 0..=(old_lines.len() - from_lines.len()) {
            let trimmed_match = old_lines[start..start + from_lines.len()]
                .iter()
                .zip(from_lines.iter())
                .all(|(a, b)| a.trim() == b.trim());
            if trimmed_match && from_lines.iter().any(|l| !l.trim().is_empty()) {
                let first_line = start + 1;
                let last_line = start + from_lines.len();
                return Some(format!(
                    "hint: lines {first_line}-{last_line} match when leading/trailing whitespace is ignored; check exact indentation with read offset={first_line}"
                ));
            }
        }
    }

    if let Some(anchor) = from_lines.iter().map(|l| l.trim()).find(|l| !l.is_empty()) {
        if let Some(idx) = old_lines.iter().position(|line| line.trim() == anchor) {
            let line_no = idx + 1;
            return Some(format!(
                "hint: first line matches near line {line_no}; use read offset={line_no} to inspect exact surrounding lines"
            ));
        }
    }

    None
}

struct Observed {
    version: Version,
    ranges: Vec<Range<usize>>,
}
#[derive(Default)]
pub(crate) struct Observations {
    files: HashMap<PathBuf, Observed>,
    aliases: HashMap<PathBuf, PathBuf>,
}
impl Observations {
    pub(crate) fn record(
        &mut self,
        requested: PathBuf,
        path: PathBuf,
        text: &str,
        metadata: &Metadata,
        range: Range<usize>,
    ) {
        let version = Version::new(text, metadata);
        let observed = self.files.entry(path.clone()).or_insert_with(|| Observed {
            version: version.clone(),
            ranges: Vec::new(),
        });
        if observed.version != version {
            observed.version = version;
            observed.ranges.clear();
        }
        observed.ranges.push(range);
        observed.ranges.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<usize>> = Vec::new();
        for range in observed.ranges.drain(..) {
            if let Some(last) = merged.last_mut().filter(|last| last.end >= range.start) {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        observed.ranges = merged;
        self.aliases.insert(requested, path);
    }
    fn validate(
        &self,
        requested: &Path,
        path: &Path,
        version: Option<&Version>,
        range: Range<usize>,
    ) -> Result<()> {
        if self.aliases.get(requested).is_some_and(|old| old != path) {
            return Err(Conflict(format!(
                "{} now resolves to a different file",
                requested.display()
            ))
            .into());
        }
        match (self.files.get(path), version) {
            (Some(old), Some(now)) if &old.version == now => {
                if !old
                    .ranges
                    .iter()
                    .any(|read| read.start <= range.start && read.end >= range.end)
                {
                    bail!("read the affected contents before writing {}; use read offset/limit to observe the missing lines", path.display());
                }
                Ok(())
            }
            (Some(_), _) => Err(Conflict(format!("{} changed since read", path.display())).into()),
            (None, Some(_)) => bail!("read {} before writing it", path.display()),
            (None, None) => Ok(()),
        }
    }
}

/// Immutable contents/version presented for approval and revalidated at apply.
pub struct PreparedMutation {
    raw: String,
    requested: PathBuf,
    path: PathBuf,
    before: Option<(String, Metadata, Version)>,
    updated: String,
    affected: Range<usize>,
    name: String,
    preview: ChangePreview,
}
impl PreparedMutation {
    pub fn preview(&self) -> &ChangePreview {
        &self.preview
    }
}

impl Toolbox {
    pub fn prepare_mutation(
        &self,
        name: &str,
        input: &serde_json::Value,
    ) -> Result<PreparedMutation> {
        let raw = input_string(input, "path")?;
        let requested = self.candidate(raw);
        let path = self.resolve_for_write(raw)?;
        let before = match fs::metadata(&path) {
            Ok(metadata) => {
                if !metadata.is_file() {
                    bail!("{} is not a regular file", path.display());
                }
                let text = fs::read_to_string(&path)
                    .context("existing file is not valid UTF-8 or cannot be read")?;
                let version = Version::new(&text, &metadata);
                Some((text, metadata, version))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let old = before.as_ref().map_or("", |(text, _, _)| text.as_str());
        let (updated, affected) = match name {
            "write" => (
                input_string(input, "content")?.to_string(),
                0..old.lines().count(),
            ),
            "edit" => {
                if before.is_none() {
                    bail!("cannot edit missing file {raw}");
                }
                let from = input_string(input, "old")?;
                let to = input_string(input, "new")?;
                if from.is_empty() {
                    bail!("old string must not be empty");
                }
                let matches: Vec<_> = old.match_indices(from).collect();
                if matches.is_empty() {
                    if let Some(hint) = diagnose_missing_old(old, from) {
                        bail!("old string not found in {raw} ({hint})");
                    }
                    bail!("old string not found in {raw}");
                }
                if matches.len() != 1 {
                    let lines = match_line_numbers(old, &matches);
                    bail!(
                        "old string is not unique in {raw}; found {} matches (at lines {lines}; include more surrounding context)",
                        matches.len()
                    );
                }
                let start = old[..matches[0].0].bytes().filter(|&b| b == b'\n').count();
                let end = old[..matches[0].0 + from.len()]
                    .bytes()
                    .filter(|&b| b == b'\n')
                    .count()
                    + usize::from(!from.ends_with('\n'));
                (old.replacen(from, to, 1), start..end.max(start + 1))
            }
            _ => bail!("unsupported mutation {name}"),
        };
        let warning = if self.policy.mode == SandboxMode::ReadOnly {
            Some("sandbox is read-only; this change will be rejected".into())
        } else {
            self.observed
                .lock()
                .map_err(|_| anyhow::anyhow!("observation lock poisoned"))?
                .validate(
                    &requested,
                    &path,
                    before.as_ref().map(|(_, _, version)| version),
                    affected.clone(),
                )
                .err()
                .map(|e| format!("{e}; this call will be rejected"))
        };
        let mut preview = ChangePreview::diff(vec![FilePreview::from_text(
            self.display_path(&path),
            old,
            &updated,
            before.is_none(),
        )]);
        preview.warning = warning;
        Ok(PreparedMutation {
            raw: raw.into(),
            requested,
            path,
            before,
            updated,
            affected,
            name: name.into(),
            preview,
        })
    }

    pub(crate) fn mutate(
        &self,
        name: &str,
        input: &serde_json::Value,
    ) -> Result<(String, FileChange)> {
        self.apply_mutation(self.prepare_mutation(name, input)?)
    }

    pub(crate) fn apply_mutation(
        &self,
        prepared: PreparedMutation,
    ) -> Result<(String, FileChange)> {
        let PreparedMutation {
            raw,
            requested,
            path,
            before,
            updated,
            affected,
            name,
            ..
        } = prepared;
        if self.policy.mode == SandboxMode::ReadOnly {
            bail!("sandbox is read-only; cannot {name} files");
        }
        if self.candidate(&raw) != requested || self.resolve_for_write(&raw)? != path {
            return Err(Conflict(format!("{raw} changed target or workspace")).into());
        }
        self.observed
            .lock()
            .map_err(|_| anyhow::anyhow!("observation lock poisoned"))?
            .validate(
                &requested,
                &path,
                before.as_ref().map(|(_, _, version)| version),
                affected.clone(),
            )?;
        let old = before.as_ref().map_or("", |(text, _, _)| text.as_str());
        let parent = path.parent().context("path has no parent")?;
        let mut temporary =
            tempfile::NamedTempFile::new_in(parent).context("prepare atomic write")?;
        temporary
            .write_all(updated.as_bytes())
            .context("write temporary content")?;
        if let Some((_, metadata, _)) = &before {
            temporary
                .as_file()
                .set_permissions(metadata.permissions())?;
        }
        temporary
            .as_file()
            .sync_all()
            .context("flush temporary content")?;
        // Re-resolve after preparation/approval; refuse moved aliases and stale
        // versions. Rename is atomic replacement, not external-editor CAS.
        if self.resolve_for_write(&raw)? != path {
            return Err(Conflict(format!("{raw} changed target")).into());
        }
        let current = match fs::metadata(&path) {
            Ok(metadata) => Some(Version::new(&fs::read_to_string(&path)?, &metadata)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        if current.as_ref() != before.as_ref().map(|(_, _, version)| version) {
            return Err(Conflict(format!("{raw} changed while preparing the write")).into());
        }
        if before.is_some() {
            temporary
                .persist(&path)
                .map_err(|e| e.error)
                .context("replace file")?;
        } else {
            temporary
                .persist_noclobber(&path)
                .map_err(|e| e.error)
                .context("create file without overwriting")?;
        }
        let metadata = fs::metadata(&path)?;
        // A partial edit does not authorize overwriting previously unseen lines.
        let known = if name == "write" {
            0..updated.lines().count()
        } else {
            affected.start
                ..affected.end.saturating_add_signed(
                    updated.lines().count() as isize - old.lines().count() as isize,
                )
        };
        self.observed
            .lock()
            .map_err(|_| anyhow::anyhow!("observation lock poisoned"))?
            .record(requested, path.clone(), &updated, &metadata, known);
        let evidence = FileChange {
            path: self.display_path(&path),
            before_digest: before.as_ref().map(|(_, _, v)| v.digest.clone()),
            after_digest: digest(updated.as_bytes()),
        };
        let diff = TextDiff::from_lines(old, &updated)
            .unified_diff()
            .context_radius(3)
            .header("before", "after")
            .to_string();
        Ok((
            format!(
                "{} {}\n{}",
                if name == "write" { "wrote" } else { "edited" },
                self.display_path(&path),
                diff.chars().take(4000).collect::<String>()
            ),
            evidence,
        ))
    }
}
