//! Workspace tools exposed to the model.
//!
//! The tool surface is deliberately small: `read`, `write`, `edit`, `grep`,
//! `glob`, and `bash`. All file paths are resolved inside the working
//! directory; `write`/`edit` require the file to have been observed by `read`
//! first.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use globset::GlobBuilder;
use ignore::WalkBuilder;
use niucode_provider::ToolSpec;
use regex::Regex;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::time::timeout;

mod mutation;
mod preview;
pub use mutation::{digest, PreparedMutation};
use niucode_core::{ExecutionEvidence, ProcessOutcome};

pub use preview::{
    ChangePreview, FilePreview, LineKind, PreviewBody, PreviewHunk, PreviewLine,
    PREVIEW_CONTEXT_LINES,
};

const READ_MAX_LINES: usize = 2_000;
const READ_MAX_BYTES: usize = 50_000;
const BASH_MAX_BYTES: usize = 20_000;
const BASH_DEFAULT_TIMEOUT: u64 = 120;
const BASH_MAX_TIMEOUT: u64 = 600;
const BASH_CAPTURE_BYTES: usize = 8 * 1024 * 1024;
const SEARCH_MAX_RESULTS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxMode {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

impl SandboxMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::FullAccess => "full-access",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkMode {
    On,
    Off,
}

impl NetworkMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxPolicy {
    pub mode: SandboxMode,
    pub network: NetworkMode,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            mode: SandboxMode::WorkspaceWrite,
            network: NetworkMode::Off,
        }
    }
}

impl SandboxPolicy {
    pub fn new(mode: SandboxMode, network: NetworkMode) -> Self {
        Self { mode, network }
    }

    pub fn backend(self) -> &'static str {
        if self.mode == SandboxMode::FullAccess && self.network == NetworkMode::On {
            "none"
        } else if sandbox_available() {
            "bwrap"
        } else {
            "unavailable"
        }
    }
}

pub fn sandbox_available() -> bool {
    StdCommand::new("bwrap")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

pub struct RecordedToolResult {
    pub result: ToolResult,
    pub evidence: ExecutionEvidence,
}

impl RecordedToolResult {
    pub fn failed(error: anyhow::Error) -> Self {
        Self::from_result(Err(error), ExecutionEvidence::default())
    }
    fn from_result(result: Result<String>, mut evidence: ExecutionEvidence) -> Self {
        let result = match result {
            Ok(output) => ToolResult {
                ok: !matches!(evidence.process, Some(ProcessOutcome::TimedOut)),
                output,
            },
            Err(error) => {
                if let Some(conflict) = error.downcast_ref::<mutation::Conflict>() {
                    evidence.conflict = Some(conflict.to_string());
                }
                ToolResult {
                    ok: false,
                    output: format!("error: {error:#}"),
                }
            }
        };
        Self { result, evidence }
    }
}

pub struct Toolbox {
    cwd: PathBuf,
    observed: Mutex<mutation::Observations>,
    policy: SandboxPolicy,
    allow_unsandboxed: bool,
    read_roots: Vec<PathBuf>,
}

impl Toolbox {
    pub fn new(cwd: impl AsRef<Path>) -> Result<Self> {
        Self::with_policy(cwd, SandboxPolicy::default())
    }

    pub fn with_policy(cwd: impl AsRef<Path>, policy: SandboxPolicy) -> Result<Self> {
        Self::with_policy_and_fallback(cwd, policy, false)
    }

    pub fn with_policy_and_fallback(
        cwd: impl AsRef<Path>,
        policy: SandboxPolicy,
        allow_unsandboxed: bool,
    ) -> Result<Self> {
        Self::with_policy_and_roots(cwd, policy, allow_unsandboxed, Vec::new())
    }

    pub fn with_policy_and_roots(
        cwd: impl AsRef<Path>,
        policy: SandboxPolicy,
        allow_unsandboxed: bool,
        read_roots: Vec<PathBuf>,
    ) -> Result<Self> {
        let cwd = cwd.as_ref().canonicalize().with_context(|| {
            format!("canonicalize working directory {}", cwd.as_ref().display())
        })?;
        if !cwd.is_dir() {
            bail!("working directory is not a directory: {}", cwd.display());
        }
        let read_roots = read_roots
            .into_iter()
            .map(|root| {
                root.canonicalize()
                    .with_context(|| format!("canonicalize read-only root {}", root.display()))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            cwd,
            observed: Mutex::new(mutation::Observations::default()),
            policy,
            allow_unsandboxed,
            read_roots,
        })
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn policy(&self) -> SandboxPolicy {
        self.policy
    }

    pub fn allows_unsandboxed(&self) -> bool {
        self.allow_unsandboxed
    }

    pub fn specs() -> Vec<ToolSpec> {
        vec![
            ToolSpec::function(
                "read",
                "Read a UTF-8 text file inside the workspace. Returns line numbers. Read a file before editing or overwriting it.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative path"},
                        "offset": {"type": "integer", "description": "1-based first line to return", "minimum": 1},
                        "limit": {"type": "integer", "description": "Maximum lines to return", "minimum": 1, "maximum": READ_MAX_LINES}
                    },
                    "required": ["path"]
                }),
            ),
            ToolSpec::function(
                "write",
                "Create or fully overwrite a text file inside the workspace. Reading an existing file first is required.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }),
            ),
            ToolSpec::function(
                "edit",
                "Replace one exact, unique string in a file after reading it. Fails if the old string is missing or appears more than once.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string", "description": "Exact text to replace; must be unique"},
                        "new": {"type": "string", "description": "Replacement text"}
                    },
                    "required": ["path", "old", "new"]
                }),
            ),
            ToolSpec::function(
                "bash",
                "Run a shell command in the workspace. Returns the last 20KB of combined output and the exit code. Prefer this for tests and builds.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_secs": {"type": "integer", "minimum": 1, "maximum": BASH_MAX_TIMEOUT}
                    },
                    "required": ["command"]
                }),
            ),
            ToolSpec::function(
                "grep",
                "Search text files in the workspace with a regular expression. Respects .gitignore, skips binaries and target/.git, and caps results.",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Rust regex"},
                        "path": {"type": "string", "description": "File or directory, defaults to the workspace root"},
                        "include": {"type": "string", "description": "Optional glob such as *.rs"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": SEARCH_MAX_RESULTS}
                    },
                    "required": ["pattern"]
                }),
            ),
            ToolSpec::function(
                "glob",
                "List files in the workspace matching a glob pattern, for example **/*.rs. Respects .gitignore and skips target/.git.",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": SEARCH_MAX_RESULTS}
                    },
                    "required": ["pattern"]
                }),
            ),
            ToolSpec::function(
                "subagent",
                "Run one or more subagents in parallel. explore is read-only (read/grep/glob + read-only external tools); worker may also write/edit/run bash under a shared write lock and optional write_scope; inherit requires explicit --allow-subagent-inheritance and copies the parent tool surface (minus nested subagents). Each task gets its own session log and budget.",
                json!({
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "prompt": {"type": "string", "description": "Task for the subagent"},
                                    "profile": {"type": "string", "enum": ["explore", "worker", "inherit"], "default": "explore", "description": "explore = read-only; worker = workspace writes; inherit = parent surface (explicit opt-in; no nested subagents)"},
                                    "worktree": {"type": "boolean", "default": false, "description": "Run a Worker in an isolated git worktree; returns a patch path. Falls back to the shared workspace outside git."},
                                    "write_scope": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "maxItems": 8,
                                        "description": "Worker only: workspace-relative path prefixes; bash is not path-scoped"
                                    },
                                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 12},
                                    "timeout_secs": {"type": "integer", "minimum": 1, "maximum": 600}
                                },
                                "required": ["prompt"]
                            }
                        }
                    },
                    "required": ["tasks"]
                }),
            ),
        ]
    }

    pub async fn execute(&self, name: &str, input: &Value) -> ToolResult {
        self.execute_recorded(name, input).await.result
    }

    pub async fn execute_recorded(&self, name: &str, input: &Value) -> RecordedToolResult {
        let mut evidence = ExecutionEvidence::default();
        let result = match name {
            "read" => self.read(input),
            "write" | "edit" => self.mutate(name, input).map(|(output, change)| {
                evidence.change = Some(change);
                output
            }),
            "grep" => self.grep(input),
            "glob" => self.glob(input),
            "bash" => self.bash(input).await.map(|(output, process)| {
                evidence.process = Some(process);
                output
            }),
            "subagent" => Err(anyhow::anyhow!(
                "subagent is orchestrated by the agent loop"
            )),
            other => Err(anyhow::anyhow!("unknown tool {other:?}")),
        };
        RecordedToolResult::from_result(result, evidence)
    }

    pub fn execute_prepared(&self, prepared: PreparedMutation) -> RecordedToolResult {
        let mut evidence = ExecutionEvidence::default();
        let result = self.apply_mutation(prepared).map(|(output, change)| {
            evidence.change = Some(change);
            output
        });
        RecordedToolResult::from_result(result, evidence)
    }

    fn read(&self, input: &Value) -> Result<String> {
        let path = self.resolve_existing_read(input_string(input, "path")?)?;
        let mut file =
            fs::File::open(&path).with_context(|| format!("read {}", self.display_path(&path)))?;
        let metadata = file.metadata()?;
        let mut text = String::new();
        std::io::Read::read_to_string(&mut file, &mut text)?;

        let offset = input
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(READ_MAX_LINES as u64)
            .clamp(1, READ_MAX_LINES as u64) as usize;

        let lines = text.lines().collect::<Vec<_>>();
        let total = lines.len();
        let start = offset.saturating_sub(1).min(total);
        let mut output = String::new();
        let mut shown = 0_usize;

        for (index, line) in lines.iter().enumerate().skip(start).take(limit) {
            let rendered = format!("{:>5}  {line}\n", index + 1);
            if output.len() + rendered.len() > READ_MAX_BYTES {
                break;
            }
            output.push_str(&rendered);
            shown += 1;
        }

        if shown == 0 {
            if total == 0 {
                output.push_str("(no lines; file has 0 lines)\n");
            } else {
                output.push_str(&format!(
                    "(no lines at offset {offset}; file has {total} lines)\n"
                ));
            }
        } else if start + shown < total {
            let end_line = start + shown;
            let next_offset = end_line + 1;
            output.push_str(&format!(
                "... ({} more lines, shown {}-{end_line} of {total}; use offset={next_offset} to continue)\n",
                total - end_line,
                start + 1,
            ));
        }

        self.observed
            .lock()
            .map_err(|_| anyhow::anyhow!("observation lock poisoned"))?
            .record(
                self.candidate(input_string(input, "path")?),
                path.clone(),
                &text,
                &metadata,
                start..start + shown,
            );
        Ok(output)
    }

    async fn bash(&self, input: &Value) -> Result<(String, ProcessOutcome)> {
        let command = input_string(input, "command")?;
        if command.trim().is_empty() {
            bail!("command must not be empty");
        }

        let timeout_secs = input
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(BASH_DEFAULT_TIMEOUT)
            .clamp(1, BASH_MAX_TIMEOUT);

        let mut child_command = self.bash_command(command)?;
        child_command
            .current_dir(&self.cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        child_command.process_group(0);

        let mut child = child_command.spawn().context("spawn bash")?;
        let pid = child.id().context("bash process id")?;
        let _guard = ProcessGroupGuard::new(pid);

        let stdout = child.stdout.take().context("take stdout")?;
        let stderr = child.stderr.take().context("take stderr")?;
        let capture = timeout(Duration::from_secs(timeout_secs), async {
            tokio::try_join!(
                async {
                    let status = child.wait().await?;
                    // Descendants must not keep output pipes or survive this call.
                    kill_group(pid);
                    Ok::<_, std::io::Error>(status)
                },
                read_capped(stdout, BASH_CAPTURE_BYTES),
                read_capped(stderr, BASH_CAPTURE_BYTES),
            )
        })
        .await;
        let (status, (stdout, stdout_truncated), (stderr, stderr_truncated)) = match capture {
            Ok(result) => result.context("capture bash output")?,
            Err(_) => {
                kill_group(pid);
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Ok((
                    format!("command timed out after {timeout_secs}s"),
                    ProcessOutcome::TimedOut,
                ));
            }
        };
        let process = match status.code() {
            Some(code) => ProcessOutcome::Exited { code },
            None => {
                #[cfg(unix)]
                let signal = {
                    use std::os::unix::process::ExitStatusExt;
                    status.signal()
                };
                #[cfg(not(unix))]
                let signal = None;
                ProcessOutcome::Signaled { signal }
            }
        };

        let mut combined = String::new();
        if !stdout.is_empty() {
            combined.push_str("stdout:\n");
            combined.push_str(&String::from_utf8_lossy(&stdout));
        }
        if !stderr.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str("stderr:\n");
            combined.push_str(&String::from_utf8_lossy(&stderr));
        }
        if combined.is_empty() {
            combined.push_str("(no output)\n");
        }

        let status = status
            .code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "signal".into());

        if stdout_truncated || stderr_truncated || combined.chars().count() > BASH_MAX_BYTES {
            let path = temp_output_path(pid);
            fs::write(&path, &combined)
                .with_context(|| format!("write full bash output {}", path.display()))?;
            let total_lines = combined.lines().count();
            let (body, _) = tail_chars(&combined, BASH_MAX_BYTES);
            let shown_lines = body.lines().count();
            let omitted_lines = total_lines.saturating_sub(shown_lines);
            let qualifier = if stdout_truncated || stderr_truncated {
                "captured output (capped at 8MB)"
            } else {
                "full output"
            };
            return Ok((
                format!(
                    "exit: {status}\n{body}\n... ({omitted_lines} earlier lines omitted; {qualifier}: {})\n",
                    path.display()
                ),
                process,
            ));
        }

        Ok((format!("exit: {status}\n{combined}"), process))
    }

    fn bash_command(&self, command: &str) -> Result<Command> {
        if self.policy.mode == SandboxMode::FullAccess && self.policy.network == NetworkMode::On {
            let mut child = Command::new("bash");
            child.arg("-c").arg(command);
            return Ok(child);
        }

        if !sandbox_available() {
            if self.allow_unsandboxed {
                let mut child = Command::new("bash");
                child.arg("-c").arg(command);
                return Ok(child);
            }
            bail!(
                "sandbox backend bwrap is not installed; use --approval ask to allow an unsandboxed fallback, or install bubblewrap"
            );
        }

        let mut child = Command::new("bwrap");
        child.arg("--die-with-parent").arg("--unshare-all");
        if self.policy.network == NetworkMode::On {
            child.arg("--share-net");
        }

        if self.policy.mode == SandboxMode::FullAccess {
            child.arg("--bind").arg("/").arg("/");
        } else {
            child.arg("--ro-bind").arg("/").arg("/");
        }
        child.arg("--dev").arg("/dev").arg("--proc").arg("/proc");

        if self.policy.mode == SandboxMode::WorkspaceWrite {
            child.arg("--bind").arg(&self.cwd).arg(&self.cwd);
        }

        child
            .arg("--chdir")
            .arg(&self.cwd)
            .arg("--")
            .arg("bash")
            .arg("-c")
            .arg(command);
        Ok(child)
    }

    fn grep(&self, input: &Value) -> Result<String> {
        let pattern = input_string(input, "pattern")?;
        if pattern.is_empty() {
            bail!("pattern must not be empty");
        }
        let regex = Regex::new(pattern).with_context(|| format!("invalid regex {pattern:?}"))?;
        let root = match input.get("path").and_then(Value::as_str) {
            Some(path) if !path.trim().is_empty() => self.resolve_existing(path)?,
            _ => self.cwd.clone(),
        };
        let include = match input.get("include").and_then(Value::as_str) {
            Some(glob) if !glob.trim().is_empty() => Some(
                GlobBuilder::new(glob)
                    .literal_separator(false)
                    .build()
                    .with_context(|| format!("invalid include glob {glob:?}"))?
                    .compile_matcher(),
            ),
            _ => None,
        };
        let max = input
            .get("max_results")
            .and_then(Value::as_u64)
            .unwrap_or(100)
            .clamp(1, SEARCH_MAX_RESULTS as u64) as usize;

        let walker = WalkBuilder::new(&root)
            .standard_filters(true)
            .require_git(false)
            .filter_entry(|entry| {
                let name = entry.file_name().to_string_lossy();
                name != ".git" && name != "target" && name != "node_modules"
            })
            .build();

        let mut matches = Vec::new();
        let mut truncated = false;

        'outer: for entry in walker {
            let entry = entry.context("walk workspace")?;
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            let path = entry.path();
            let relative = self.display_path(path);
            if let Some(matcher) = &include {
                if !matcher.is_match(&relative) {
                    continue;
                }
            }
            if entry.metadata().map(|meta| meta.len()).unwrap_or(0) > 2_000_000 {
                continue;
            }
            let Ok(text) = fs::read_to_string(path) else {
                continue;
            };
            if text.contains('\0') {
                continue;
            }

            for (index, line) in text.lines().enumerate() {
                if !regex.is_match(line) {
                    continue;
                }
                let preview: String = line.chars().take(500).collect();
                matches.push(format!("{relative}:{}: {preview}", index + 1));
                if matches.len() >= max {
                    truncated = true;
                    break 'outer;
                }
            }
        }

        if matches.is_empty() {
            return Ok(format!("no matches for {pattern:?}"));
        }

        let mut output = matches.join("\n");
        if truncated {
            output.push_str(&format!(
                "\n... (stopped at {max} matches; narrow `path`, `include`, or `pattern` to see more)"
            ));
        }
        Ok(output)
    }

    fn glob(&self, input: &Value) -> Result<String> {
        let pattern = input_string(input, "pattern")?;
        if pattern.trim().is_empty() {
            bail!("pattern must not be empty");
        }
        let matcher = GlobBuilder::new(pattern)
            .literal_separator(false)
            .build()
            .with_context(|| format!("invalid glob {pattern:?}"))?
            .compile_matcher();
        let max = input
            .get("max_results")
            .and_then(Value::as_u64)
            .unwrap_or(SEARCH_MAX_RESULTS as u64)
            .clamp(1, SEARCH_MAX_RESULTS as u64) as usize;

        let walker = WalkBuilder::new(&self.cwd)
            .standard_filters(true)
            .require_git(false)
            .filter_entry(|entry| {
                let name = entry.file_name().to_string_lossy();
                name != ".git" && name != "target" && name != "node_modules"
            })
            .build();

        let mut paths = Vec::new();
        let mut truncated = false;
        for entry in walker {
            let entry = entry.context("walk workspace")?;
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            let relative = self.display_path(entry.path());
            if matcher.is_match(&relative) {
                if paths.len() >= max {
                    truncated = true;
                    break;
                }
                paths.push(relative);
            }
        }

        if paths.is_empty() {
            return Ok(format!("no files match {pattern:?}"));
        }
        paths.sort();
        let mut output = paths.join("\n");
        if truncated {
            output.push_str(&format!(
                "\n... (stopped at {max} files; narrow `pattern` to see more)"
            ));
        }
        Ok(output)
    }

    fn resolve_existing_read(&self, raw: &str) -> Result<PathBuf> {
        let candidate = self.candidate(raw);
        let canonical = candidate
            .canonicalize()
            .with_context(|| format!("resolve {raw}"))?;
        if canonical.starts_with(&self.cwd)
            || self
                .read_roots
                .iter()
                .any(|root| canonical.starts_with(root))
        {
            Ok(canonical)
        } else {
            bail!(
                "{} is outside the workspace and skill roots",
                self.display_path(&canonical)
            )
        }
    }

    fn resolve_existing(&self, raw: &str) -> Result<PathBuf> {
        let candidate = self.candidate(raw);
        let canonical = candidate
            .canonicalize()
            .with_context(|| format!("resolve {}", raw))?;
        self.ensure_inside(&canonical)?;
        Ok(canonical)
    }

    fn resolve_for_write(&self, raw: &str) -> Result<PathBuf> {
        let candidate = self.candidate(raw);
        if candidate.exists() {
            let canonical = candidate
                .canonicalize()
                .with_context(|| format!("resolve {}", raw))?;
            self.ensure_inside(&canonical)?;
            return Ok(canonical);
        }

        let parent = candidate.parent().context("path has no parent")?;
        let parent = parent
            .canonicalize()
            .with_context(|| format!("resolve parent of {raw}"))?;
        self.ensure_inside(&parent)?;
        let name = candidate.file_name().context("path has no file name")?;
        Ok(parent.join(name))
    }

    fn candidate(&self, raw: &str) -> PathBuf {
        let path = Path::new(raw);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    fn ensure_inside(&self, path: &Path) -> Result<()> {
        if !path.starts_with(&self.cwd) {
            bail!("path escapes the workspace: {}", path.display());
        }
        Ok(())
    }

    fn display_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.cwd)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

/// Kills a child process group on completion or cancellation.
/// The child must be spawned with `process_group(0)` on Unix.
pub struct ProcessGroupGuard {
    pid: u32,
}

impl ProcessGroupGuard {
    pub fn new(pid: u32) -> Self {
        Self { pid }
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        kill_group(self.pid);
    }
}

fn kill_group(pid: u32) {
    #[cfg(unix)]
    {
        if pid > 0 {
            // Negative pid targets the process group created by process_group(0).
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
}

async fn read_capped<R: AsyncRead + Unpin>(
    mut reader: R,
    cap: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut data = Vec::new();
    let mut buffer = [0_u8; 8192];
    let mut truncated = false;

    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        if data.len() < cap {
            let take = (cap - data.len()).min(read);
            data.extend_from_slice(&buffer[..take]);
            if take < read {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }

    Ok((data, truncated))
}

fn temp_output_path(pid: u32) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("niucode-bash-{pid}-{stamp}.log"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub ok: bool,
    pub output: String,
}

pub(crate) fn input_string<'a>(input: &'a Value, key: &str) -> Result<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string argument {key:?}"))
}

fn tail_chars(value: &str, max_chars: usize) -> (String, bool) {
    let count = value.chars().count();
    if count <= max_chars {
        return (value.to_string(), false);
    }
    (value.chars().skip(count - max_chars).collect(), true)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_workspace(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("niucode-tools-{tag}-{stamp}-{sequence}"));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).expect("create workspace");
        directory
    }

    fn test_toolbox(directory: &Path) -> Toolbox {
        Toolbox::with_policy(
            directory,
            SandboxPolicy::new(SandboxMode::FullAccess, NetworkMode::On),
        )
        .expect("toolbox")
    }

    #[tokio::test]
    async fn read_write_edit_round_trip() {
        let directory = temp_workspace("roundtrip");
        fs::write(directory.join("a.txt"), "hello\nworld\n").expect("fixture");
        let tools = test_toolbox(&directory);

        let read = tools.execute("read", &json!({"path": "a.txt"})).await;
        assert!(read.ok, "{}", read.output);
        assert!(read.output.contains("hello"));
        assert!(read.output.contains("world"));

        let write = tools
            .execute("write", &json!({"path": "b.txt", "content": "new file"}))
            .await;
        assert!(write.ok, "{}", write.output);
        assert_eq!(
            fs::read_to_string(directory.join("b.txt")).expect("read"),
            "new file"
        );

        let edit = tools
            .execute(
                "edit",
                &json!({"path": "a.txt", "old": "world", "new": "niucode"}),
            )
            .await;
        assert!(edit.ok, "{}", edit.output);
        assert_eq!(
            fs::read_to_string(directory.join("a.txt")).expect("read"),
            "hello\nniucode\n"
        );

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn existing_file_must_be_read_before_write() {
        let directory = temp_workspace("read-first");
        fs::write(directory.join("a.txt"), "old").expect("fixture");
        let tools = test_toolbox(&directory);

        let rejected = tools
            .execute("write", &json!({"path": "a.txt", "content": "new"}))
            .await;
        assert!(!rejected.ok);
        assert!(rejected.output.contains("read ") && rejected.output.contains("a.txt"));

        let read = tools.execute("read", &json!({"path": "a.txt"})).await;
        assert!(read.ok);
        let written = tools
            .execute("write", &json!({"path": "a.txt", "content": "new"}))
            .await;
        assert!(written.ok, "{}", written.output);

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn edit_rejects_ambiguous_match() {
        let directory = temp_workspace("ambiguous");
        fs::write(directory.join("a.txt"), "same same").expect("fixture");
        let tools = test_toolbox(&directory);
        tools.execute("read", &json!({"path": "a.txt"})).await;

        let result = tools
            .execute("edit", &json!({"path": "a.txt", "old": "same", "new": "x"}))
            .await;
        assert!(!result.ok);
        assert!(result.output.contains("not unique"));

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn rejects_paths_outside_workspace() {
        let directory = temp_workspace("escape");
        let outside = directory
            .parent()
            .expect("parent")
            .join("outside-secret.txt");
        fs::write(&outside, "secret").expect("fixture");
        let tools = test_toolbox(&directory);

        let result = tools
            .execute("read", &json!({"path": "../outside-secret.txt"}))
            .await;
        assert!(!result.ok);
        assert!(result.output.contains("outside") || result.output.contains("escapes"));

        let _ = fs::remove_file(outside);
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn grep_and_glob_respect_include_and_gitignore() {
        let directory = temp_workspace("search");
        fs::create_dir_all(directory.join("src")).expect("src dir");
        fs::write(directory.join("src/a.rs"), "fn needle() {}\n").expect("fixture");
        fs::write(directory.join("src/b.txt"), "needle\n").expect("fixture");
        fs::write(directory.join(".gitignore"), "ignored.rs\n").expect("gitignore");
        fs::write(directory.join("ignored.rs"), "needle\n").expect("fixture");
        let tools = test_toolbox(&directory);

        let grep = tools
            .execute("grep", &json!({"pattern": "needle", "include": "*.rs"}))
            .await;
        assert!(grep.ok, "{}", grep.output);
        assert!(grep.output.contains("src/a.rs:1:"));
        assert!(!grep.output.contains("b.txt"));
        assert!(!grep.output.contains("ignored.rs"));

        let glob = tools.execute("glob", &json!({"pattern": "*.rs"})).await;
        assert!(glob.ok, "{}", glob.output);
        assert!(glob.output.contains("src/a.rs"));
        assert!(!glob.output.contains("ignored.rs"));

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn read_only_policy_denies_mutation() {
        let directory = temp_workspace("read-only");
        fs::write(directory.join("a.txt"), "old").expect("fixture");
        let tools = Toolbox::with_policy(
            &directory,
            SandboxPolicy::new(SandboxMode::ReadOnly, NetworkMode::Off),
        )
        .expect("toolbox");
        tools.execute("read", &json!({"path": "a.txt"})).await;

        let write = tools
            .execute("write", &json!({"path": "a.txt", "content": "new"}))
            .await;
        assert!(!write.ok);
        assert!(write.output.contains("read-only"));

        let edit = tools
            .execute(
                "edit",
                &json!({"path": "a.txt", "old": "old", "new": "new"}),
            )
            .await;
        assert!(!edit.ok);
        assert!(edit.output.contains("read-only"));

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn workspace_write_denies_escape_through_bash() {
        if !sandbox_available() {
            return;
        }

        let directory = temp_workspace("bwrap");
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let outside = std::env::temp_dir().join(format!("niucode-escape-{stamp}"));
        let tools = Toolbox::with_policy(
            &directory,
            SandboxPolicy::new(SandboxMode::WorkspaceWrite, NetworkMode::Off),
        )
        .expect("toolbox");

        let result = tools
            .execute(
                "bash",
                &json!({"command": format!("echo bad > {}", outside.display())}),
            )
            .await;

        assert!(result.ok);
        assert!(result.output.contains("exit: 1"));
        assert!(!outside.exists());

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn bash_timeout_kills_the_process_group() {
        let directory = temp_workspace("timeout");
        let tools = test_toolbox(&directory);
        let started = Instant::now();

        let result = tools
            .execute("bash", &json!({"command": "sleep 5", "timeout_secs": 1}))
            .await;

        assert!(!result.ok);
        assert!(result.output.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(4));

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn bash_returns_output_and_exit_code() {
        let directory = temp_workspace("bash");
        let tools = test_toolbox(&directory);

        let result = tools
            .execute("bash", &json!({"command": "printf 'hello'; exit 3"}))
            .await;
        assert!(result.ok);
        assert!(result.output.contains("hello"));
        assert!(result.output.contains("exit: 3"));

        fs::remove_dir_all(directory).expect("cleanup");
    }
    #[tokio::test]
    async fn read_allows_extra_read_only_roots() {
        let workspace =
            std::env::temp_dir().join(format!("niucode-tool-roots-{}", std::process::id()));
        let root = std::env::temp_dir().join(format!("niucode-skill-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&workspace);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&workspace).expect("workspace");
        std::fs::create_dir_all(&root).expect("root");
        std::fs::write(root.join("SKILL.md"), "# Skill\n").expect("skill");
        let toolbox = Toolbox::with_policy_and_roots(
            &workspace,
            SandboxPolicy::default(),
            false,
            vec![root.clone()],
        )
        .expect("toolbox");
        let result = toolbox
            .execute("read", &serde_json::json!({"path": root.join("SKILL.md")}))
            .await;
        assert!(result.ok, "{}", result.output);
        assert!(result.output.contains("# Skill"));
        let _ = std::fs::remove_dir_all(workspace);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn read_grep_glob_bash_emit_structured_truncation_hints() {
        let directory = temp_workspace("truncation-hints");
        fs::write(directory.join("lines.txt"), "one\ntwo\nthree\nfour\nfive\n").expect("fixture");
        fs::write(directory.join("other.txt"), "one\ntwo\n").expect("fixture");
        let tools = test_toolbox(&directory);

        let read = tools
            .execute(
                "read",
                &json!({"path": "lines.txt", "offset": 2, "limit": 2}),
            )
            .await;
        assert!(read.ok, "{}", read.output);
        assert!(
            read.output
                .contains("... (2 more lines, shown 2-3 of 5; use offset=4 to continue)"),
            "unexpected read output: {}",
            read.output
        );

        let beyond = tools
            .execute("read", &json!({"path": "lines.txt", "offset": 10}))
            .await;
        assert!(beyond.ok, "{}", beyond.output);
        assert!(
            beyond
                .output
                .contains("(no lines at offset 10; file has 5 lines)"),
            "unexpected beyond output: {}",
            beyond.output
        );

        let grep = tools
            .execute("grep", &json!({"pattern": "o", "max_results": 1}))
            .await;
        assert!(grep.ok, "{}", grep.output);
        assert!(
            grep.output.contains(
                "... (stopped at 1 matches; narrow `path`, `include`, or `pattern` to see more)"
            ),
            "unexpected grep output: {}",
            grep.output
        );

        let glob = tools
            .execute("glob", &json!({"pattern": "*.txt", "max_results": 1}))
            .await;
        assert!(glob.ok, "{}", glob.output);
        assert!(
            glob.output
                .contains("... (stopped at 1 files; narrow `pattern` to see more)"),
            "unexpected glob output: {}",
            glob.output
        );

        let bash = tools
            .execute(
                "bash",
                &json!({"command": "for i in $(seq 1 2500); do echo \"line-$i-0123456789\"; done"}),
            )
            .await;
        assert!(bash.ok, "{}", bash.output);
        assert!(
            bash.output.contains("earlier lines omitted; full output:"),
            "unexpected bash output: {}",
            bash.output
        );

        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[tokio::test]
    async fn edit_reports_whitespace_and_duplicate_line_diagnostics() {
        let directory = temp_workspace("edit-diagnostics");
        fs::write(
            directory.join("code.rs"),
            "fn alpha() {\n    let value = 1;\n    println!(\"{value}\");\n}\nfn beta() {\n    let value = 1;\n}\n",
        )
        .expect("fixture");
        let tools = test_toolbox(&directory);
        tools.execute("read", &json!({"path": "code.rs"})).await;

        let indent_mismatch = tools
            .execute(
                "edit",
                &json!({
                    "path": "code.rs",
                    "old": "  let value = 1;\n  println!(\"{value}\");",
                    "new": "    let value = 2;"
                }),
            )
            .await;
        assert!(!indent_mismatch.ok);
        assert!(
            indent_mismatch
                .output
                .contains("hint: lines 2-3 match when leading/trailing whitespace is ignored"),
            "unexpected indent diagnostic: {}",
            indent_mismatch.output
        );

        let anchor_hint = tools
            .execute(
                "edit",
                &json!({
                    "path": "code.rs",
                    "old": "fn beta() {\n    let wrong = 99;",
                    "new": "fn beta() {}"
                }),
            )
            .await;
        assert!(!anchor_hint.ok);
        assert!(
            anchor_hint
                .output
                .contains("hint: first line matches near line 5"),
            "unexpected anchor diagnostic: {}",
            anchor_hint.output
        );

        let non_unique = tools
            .execute(
                "edit",
                &json!({
                    "path": "code.rs",
                    "old": "    let value = 1;",
                    "new": "    let value = 2;"
                }),
            )
            .await;
        assert!(!non_unique.ok);
        assert!(
            non_unique.output.contains("at lines 2, 6"),
            "unexpected non-unique diagnostic: {}",
            non_unique.output
        );

        fs::remove_dir_all(directory).expect("cleanup");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRisk {
    Low,
    Medium,
    High,
    Unknown,
}

impl ToolRisk {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEffect {
    Read,
    Write,
    Execute,
    Network,
    Delegate,
}

impl ToolEffect {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Network => "network",
            Self::Delegate => "delegate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolMetadata {
    pub name: &'static str,
    pub risk: ToolRisk,
    pub effect: ToolEffect,
    pub read_only: bool,
}

/// Static metadata for built-in tools. Unknown/external tools are treated as
/// high-risk, non-read-only by callers unless the extension declares otherwise.
pub fn metadata(name: &str) -> Option<ToolMetadata> {
    let (risk, effect, read_only) = match name {
        "read" | "grep" | "glob" => (ToolRisk::Low, ToolEffect::Read, true),
        "write" | "edit" => (ToolRisk::Medium, ToolEffect::Write, false),
        "bash" => (ToolRisk::High, ToolEffect::Execute, false),
        "subagent" => (ToolRisk::High, ToolEffect::Delegate, false),
        _ => return None,
    };
    Some(ToolMetadata {
        name: match name {
            "read" => "read",
            "grep" => "grep",
            "glob" => "glob",
            "write" => "write",
            "edit" => "edit",
            "bash" => "bash",
            "subagent" => "subagent",
            _ => "unknown",
        },
        risk,
        effect,
        read_only,
    })
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    #[test]
    fn builtin_metadata_classifies_effects() {
        assert_eq!(metadata("read").unwrap().risk, ToolRisk::Low);
        assert!(metadata("grep").unwrap().read_only);
        assert_eq!(metadata("write").unwrap().effect, ToolEffect::Write);
        assert_eq!(metadata("bash").unwrap().risk, ToolRisk::High);
        assert_eq!(metadata("subagent").unwrap().effect, ToolEffect::Delegate);
        assert!(metadata("external").is_none());
    }
}
