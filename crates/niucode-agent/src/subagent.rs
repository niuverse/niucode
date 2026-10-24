//! Read-only parallel subagents with independent session logs.

use std::path::{Path, PathBuf};
#[cfg(test)]
use std::process::Command;
use std::time::Duration;
use tokio::io::AsyncReadExt;

use anyhow::{bail, Context, Result};
use futures_util::future::join_all;
use niucode_core::{now_ms, Event, SessionLog, ToolCall, Usage};
use niucode_provider::Provider;
use niucode_tools::{NetworkMode, SandboxMode, SandboxPolicy, ToolResult, Toolbox};
use serde_json::Value;

use crate::{run, AgentConfig, ApprovalMode};

const MAX_TASKS: usize = 4;
const MAX_STEPS: usize = 12;
const MAX_TIMEOUT_SECS: u64 = 600;
const MAX_PROMPT_CHARS: usize = 8_000;
const MAX_SUMMARY_CHARS: usize = 2_000;
const MAX_EVENT_TASK_CHARS: usize = 4_000;
const SUBAGENT_CONTEXT_BUDGET: usize = 16_000;
const SUBAGENT_TOOLS: [&str; 3] = ["read", "grep", "glob"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubagentProfile {
    Explore,
    Worker,
    Inherit,
}

impl SubagentProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Worker => "worker",
            Self::Inherit => "inherit",
        }
    }
}

struct TaskSpec {
    prompt: String,
    profile: SubagentProfile,
    write_scope: Option<Vec<String>>,
    worktree: bool,
    max_steps: usize,
    timeout_secs: u64,
}

struct SubagentOutcome {
    ok: bool,
    profile: SubagentProfile,
    write_scope: Option<Vec<String>>,
    turns: u64,
    usage: Usage,
    summary: String,
    session: std::path::PathBuf,
}

/// Execute the `subagent` tool. Errors become an ordinary failed tool result.
pub async fn run_tool(
    provider: &dyn Provider,
    parent_session: &SessionLog,
    parent_config: &AgentConfig,
    call: &ToolCall,
) -> ToolResult {
    match run_tasks(provider, parent_session, parent_config, call).await {
        Ok(outcomes) => {
            let ok = outcomes.iter().all(|outcome| outcome.ok);
            ToolResult {
                ok,
                output: render_outcomes(&outcomes),
            }
        }
        Err(error) => ToolResult {
            ok: false,
            output: format!("error: {error:#}"),
        },
    }
}

async fn run_tasks(
    provider: &dyn Provider,
    parent_session: &SessionLog,
    parent_config: &AgentConfig,
    call: &ToolCall,
) -> Result<Vec<SubagentOutcome>> {
    let tasks = parse_tasks(&call.input)?;
    if tasks
        .iter()
        .any(|task| task.profile == SubagentProfile::Inherit)
        && !parent_config.allow_subagent_inheritance
    {
        bail!(
            "subagent profile \"inherit\" requires --allow-subagent-inheritance (explicit opt-in)"
        );
    }
    let directory = parent_session
        .path()
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{}.subagents", parent_session.id()));
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create subagent session directory {}", directory.display()))?;

    let mut jobs = Vec::with_capacity(tasks.len());
    for (index, task) in tasks.into_iter().enumerate() {
        let id = format!("{}.{}", call.id, index + 1);
        let mut config = subagent_config(parent_config, &task);
        let worktree = if task.worktree {
            match Worktree::create(parent_config, &id).await {
                Some(worktree) => {
                    config.working_dir = worktree.path.clone();
                    config.write_lock = None;
                    Some(worktree)
                }
                None => None,
            }
        } else {
            None
        };
        let session = SessionLog::create(&directory)?;
        session.append(&Event::SessionStart {
            id: id.clone(),
            cwd: parent_config.working_dir.display().to_string(),
            model: parent_config.model.clone(),
            provider: parent_config.provider.clone(),
            ts: now_ms(),
        })?;
        session.append(&Event::Policy {
            sandbox: config.sandbox.mode.as_str().into(),
            approval: config.approval.as_str().into(),
            network: config.sandbox.network.as_str().into(),
            backend: config.sandbox.backend().into(),
            ts: now_ms(),
        })?;
        let session_path = session.path().to_path_buf();
        parent_session.append(&Event::SubagentSpawn {
            id: id.clone(),
            call_id: call.id.clone(),
            task: task.prompt.chars().take(MAX_EVENT_TASK_CHARS).collect(),
            profile: task.profile.as_str().to_string(),
            session: session_path.display().to_string(),
            max_steps: task.max_steps as u64,
            timeout_secs: task.timeout_secs,
            context_budget: config.context.budget_tokens as u64,
            ts: now_ms(),
        })?;
        jobs.push((id, task, session, session_path, config, worktree));
    }

    let futures = jobs.into_iter().map(
        |(id, task, mut session, session_path, config, worktree)| async move {
            let fallback = task.worktree && worktree.is_none();
            let timeout_secs = task.timeout_secs;
            let result = tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                run(
                    provider,
                    &mut session,
                    &config,
                    &task.prompt,
                    &mut |_| false,
                    &mut |_| {},
                ),
            )
            .await;

            let (ok, summary, turns, usage) = match result {
                Ok(Ok(outcome)) => (
                    true,
                    outcome.final_text,
                    outcome.turns as u64,
                    outcome.usage,
                ),
                Ok(Err(error)) => {
                    let _ = session.append(&Event::SessionEnd { ts: now_ms() });
                    (false, format!("error: {error:#}"), 0, Usage::default())
                }
                Err(_) => {
                    let _ = session.append(&Event::SessionEnd { ts: now_ms() });
                    (
                        false,
                        format!("timeout after {timeout_secs}s"),
                        0,
                        Usage::default(),
                    )
                }
            };

            let retained = worktree.as_ref().map(|tree| tree.path.clone());
            let patch = match worktree {
                Some(tree) => tree.finish().await,
                None => None,
            };
            let summary = match patch {
                Some(patch) => format!(
                    "worktree patch: {}\n\n{}",
                    patch.display(),
                    summarize(&summary)
                ),
                None => match retained.filter(|path| path.exists()) {
                    Some(path) => format!(
                        "worktree retained for inspection: {}\n{}",
                        path.display(),
                        summarize(&summary)
                    ),
                    None => summarize(&summary),
                },
            };
            let summary = if fallback {
                format!("worktree unavailable; shared workspace used\n{summary}")
            } else {
                summary
            };
            (
                id,
                task.profile,
                task.write_scope,
                session_path,
                ok,
                summary,
                turns,
                usage,
            )
        },
    );

    let results = join_all(futures).await;
    let mut outcomes = Vec::with_capacity(results.len());
    for (id, profile, write_scope, session, ok, summary, turns, usage) in results {
        parent_session.append(&Event::SubagentResult {
            id: id.clone(),
            call_id: call.id.clone(),
            ok,
            summary: summary.clone(),
            session: session.display().to_string(),
            turns,
            usage: usage.clone(),
            ts: now_ms(),
        })?;
        outcomes.push(SubagentOutcome {
            ok,
            profile,
            write_scope,
            turns,
            usage,
            summary,
            session,
        });
    }
    Ok(outcomes)
}

struct Worktree {
    path: PathBuf,
    repo: PathBuf,
    patch: PathBuf,
}

impl Worktree {
    async fn create(parent: &AgentConfig, id: &str) -> Option<Self> {
        let root = parent.worktree_root.as_ref()?;
        tokio::fs::create_dir_all(root).await.ok()?;
        let repo = git_output(
            &parent.working_dir,
            &["rev-parse".as_ref(), "--show-toplevel".as_ref()],
        )
        .await?;
        let repo = PathBuf::from(String::from_utf8(repo).ok()?.trim());
        // Never remove a previous interrupted worker to reuse its name.
        let name = format!(
            "{}-{}-{}",
            id.replace('/', "_"),
            std::process::id(),
            now_ms()
        );
        let path = root.join(&name);
        if path.exists() {
            return None;
        }
        let patch = root.join(format!("{name}.patch"));
        git_output(
            &repo,
            &[
                "worktree".as_ref(),
                "add".as_ref(),
                "--detach".as_ref(),
                path.as_os_str(),
                "HEAD".as_ref(),
            ],
        )
        .await?;
        Some(Self { path, repo, patch })
    }

    async fn finish(self) -> Option<PathBuf> {
        git_output(&self.path, &["add".as_ref(), "-N".as_ref(), ".".as_ref()]).await?;
        let diff = git_output(
            &self.path,
            &["diff".as_ref(), "--binary".as_ref(), "--no-color".as_ref()],
        )
        .await?;
        // Persist the patch before removing the worktree. Failure/cancellation
        // leaves the directory available for inspection, never silently erased.
        if !diff.is_empty() {
            tokio::fs::write(&self.patch, &diff).await.ok()?;
        }
        git_output(
            &self.repo,
            &[
                "worktree".as_ref(),
                "remove".as_ref(),
                "--force".as_ref(),
                self.path.as_os_str(),
            ],
        )
        .await?;
        (!diff.is_empty()).then_some(self.patch)
    }
}

async fn git_output(directory: &Path, args: &[&std::ffi::OsStr]) -> Option<Vec<u8>> {
    let operation = async {
        let mut command = tokio::process::Command::new("git");
        command
            .arg("-C")
            .arg(directory)
            .args(args)
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command.spawn().ok()?;
        let _group = niucode_tools::ProcessGroupGuard::new(child.id()?);
        let mut bytes = Vec::new();
        child
            .stdout
            .take()?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .await
            .ok()?;
        if bytes.len() > 8 * 1024 * 1024 || !child.wait().await.ok()?.success() {
            return None;
        }
        Some(bytes)
    };
    tokio::time::timeout(Duration::from_secs(30), operation)
        .await
        .ok()
        .flatten()
}

fn parse_tasks(input: &Value) -> Result<Vec<TaskSpec>> {
    let tasks = input
        .get("tasks")
        .and_then(Value::as_array)
        .context("missing tasks array")?;
    if tasks.is_empty() {
        bail!("tasks must not be empty");
    }
    if tasks.len() > MAX_TASKS {
        bail!("at most {MAX_TASKS} subagent tasks per call");
    }

    tasks
        .iter()
        .enumerate()
        .map(|(index, task)| {
            let prompt = task
                .get("prompt")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|prompt| !prompt.is_empty())
                .with_context(|| format!("task {} is missing a prompt", index + 1))?;
            if prompt.chars().count() > MAX_PROMPT_CHARS {
                bail!("task {} prompt is too long", index + 1);
            }
            let profile = match task
                .get("profile")
                .and_then(Value::as_str)
                .unwrap_or("explore")
            {
                "explore" => SubagentProfile::Explore,
                "worker" => SubagentProfile::Worker,
                "inherit" => SubagentProfile::Inherit,
                other => bail!("task {} has unknown profile {other:?}", index + 1),
            };
            let write_scope = task
                .get("write_scope")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                })
                .filter(|values| !values.is_empty());
            if profile == SubagentProfile::Explore && write_scope.is_some() {
                bail!(
                    "task {} is explore-only and cannot declare write_scope",
                    index + 1
                );
            }
            let worktree = task
                .get("worktree")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if worktree && profile != SubagentProfile::Worker {
                bail!(
                    "task {}: worktree is only supported for worker profile",
                    index + 1
                );
            }
            let max_steps = task
                .get("max_steps")
                .and_then(Value::as_u64)
                .map_or(6, |steps| steps.clamp(1, MAX_STEPS as u64) as usize);
            let timeout_secs = task
                .get("timeout_secs")
                .and_then(Value::as_u64)
                .map_or(120, |secs| secs.clamp(1, MAX_TIMEOUT_SECS));
            Ok(TaskSpec {
                prompt: prompt.to_string(),
                profile,
                write_scope,
                worktree,
                max_steps,
                timeout_secs,
            })
        })
        .collect()
}

fn subagent_config(parent: &AgentConfig, task: &TaskSpec) -> AgentConfig {
    let mut config = AgentConfig::new(parent.model.clone(), parent.working_dir.clone());
    config.provider = parent.provider.clone();
    config.max_steps = task.max_steps.clamp(1, MAX_STEPS);
    config.context.budget_tokens = parent
        .context
        .budget_tokens
        .clamp(1_000, SUBAGENT_CONTEXT_BUDGET);
    config.context.compact_at_percent = 75;
    config.context.keep_recent_turns = 1;
    config.extension_host = parent.extension_host.clone();
    // ADR-0034: mutating MCP is never inherited. A worker needs its own
    // approval design before it can see these tools.
    config.mcp_host = None;
    config.write_lock = parent.write_lock.clone();
    config.read_roots = parent.read_roots.clone();

    let mut allowed = match task.profile {
        SubagentProfile::Explore => SUBAGENT_TOOLS
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>(),
        SubagentProfile::Worker => ["read", "write", "edit", "grep", "glob", "bash"]
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>(),
        // Replaced by the parent tool surface in the Inherit branch below.
        SubagentProfile::Inherit => Vec::new(),
    };
    if let Some(host) = &parent.extension_host {
        allowed.extend(host.read_only_tools());
    }
    allowed.sort();
    allowed.dedup();
    config.allowed_tools = Some(allowed);

    match task.profile {
        SubagentProfile::Explore => {
            config.system_prompt = format!(
                "You are a read-only exploration subagent working inside {}.\n\
                 You may use read, grep, glob, and read-only external tools only. \n\
                 Do not modify files and do not run shell commands.\n\
                 Answer with a concise summary that includes file paths and line numbers. \n\
                 The parent agent sees only this summary; the full transcript stays in your own session log.\n\
                 Do not ask questions.",
                parent.working_dir.display()
            );
            config.sandbox = SandboxPolicy::new(SandboxMode::ReadOnly, NetworkMode::Off);
            config.approval = ApprovalMode::Never;
            config.allow_unsandboxed = false;
        }
        SubagentProfile::Worker => {
            config.system_prompt = format!(
                "You are a Worker subagent working inside {}.\n\
                 You may use read, write, edit, grep, glob, bash, and read-only external tools.\n\
                 Writes are serialized by a shared workspace write lock; respect the declared write scope.\n\
                 Do not delegate to another subagent. Keep changes scoped and verify them before returning.\n\
                 The parent agent sees only your final summary; the full transcript stays in your own session log.\n\
                 Do not ask questions.",
                parent.working_dir.display()
            );
            config.sandbox = parent.sandbox;
            config.approval = parent.approval;
            config.allow_unsandboxed = false;
            config.write_scope = task.write_scope.clone();
        }
        SubagentProfile::Inherit => {
            config.system_prompt = format!(
                "You are an explicitly inherited subagent working inside {}.\n\
                 You inherit the parent's tool surface and policy, except that nested subagents are disabled.\n\
                 Respect the parent's sandbox, approval, write lock, and declared write scope.\n\
                 The parent agent sees only your final summary; the full transcript stays in your own session log.\n\
                 Do not ask questions.",
                parent.working_dir.display()
            );
            let mut tools = match &parent.allowed_tools {
                Some(tools) => tools.clone(),
                None => {
                    let mut names = Toolbox::specs()
                        .into_iter()
                        .map(|tool| tool.function.name)
                        .collect::<Vec<_>>();
                    if let Some(host) = &parent.extension_host {
                        names.extend(host.specs().into_iter().map(|tool| tool.function.name));
                    }
                    names
                }
            };
            tools.retain(|name| name != "subagent");
            tools.sort();
            tools.dedup();
            config.allowed_tools = Some(tools);
            config.sandbox = parent.sandbox;
            config.approval = parent.approval;
            config.allow_unsandboxed = parent.allow_unsandboxed;
            config.plan_mode = parent.plan_mode;
            config.write_scope = task.write_scope.clone();
        }
    }
    config
}

fn summarize(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return "(no final summary)".to_string();
    }
    let mut summary = text.chars().take(MAX_SUMMARY_CHARS).collect::<String>();
    if text.chars().count() > MAX_SUMMARY_CHARS {
        summary.push('…');
    }
    summary
}

fn render_outcomes(outcomes: &[SubagentOutcome]) -> String {
    let mut rendered = format!("subagent: {} task(s)\n", outcomes.len());
    for (index, outcome) in outcomes.iter().enumerate() {
        let cache_pct = if outcome.usage.input_tokens > 0 {
            (outcome.usage.cached_tokens.saturating_mul(100)) / outcome.usage.input_tokens
        } else {
            0
        };
        let scope = match &outcome.write_scope {
            Some(paths) if !paths.is_empty() => format!(" · scope={}", paths.join(",")),
            _ => String::new(),
        };
        rendered.push_str(&format!(
            "[{}] {} ({} · {} turn(s) · {} in / {} out · {}% cached{scope}) · {}\n    log: {}\n",
            index + 1,
            if outcome.ok { "ok" } else { "error" },
            outcome.profile.as_str(),
            outcome.turns,
            outcome.usage.input_tokens,
            outcome.usage.output_tokens,
            cache_pct,
            outcome.summary,
            outcome.session.display()
        ));
    }
    rendered
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_tasks_and_clamps_budgets() {
        let tasks = parse_tasks(&json!({
            "tasks": [
                {"prompt": "inspect a", "max_steps": 99, "timeout_secs": 9999},
                {"prompt": "inspect b"}
            ]
        }))
        .expect("tasks");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].max_steps, MAX_STEPS);
        assert_eq!(tasks[0].timeout_secs, MAX_TIMEOUT_SECS);
        assert_eq!(tasks[1].max_steps, 6);
        assert_eq!(tasks[1].timeout_secs, 120);
    }

    #[test]
    fn rejects_empty_and_oversized_task_lists() {
        assert!(parse_tasks(&json!({"tasks": []})).is_err());
        let tasks = (0..5)
            .map(|index| json!({"prompt": format!("task {index}")}))
            .collect::<Vec<_>>();
        assert!(parse_tasks(&json!({"tasks": tasks})).is_err());
        assert!(parse_tasks(&json!({"tasks": [{}]})).is_err());
    }

    #[test]
    fn summaries_are_bounded() {
        let long = "x".repeat(MAX_SUMMARY_CHARS + 10);
        let summary = summarize(&long);
        assert_eq!(summary.chars().count(), MAX_SUMMARY_CHARS + 1);
        assert!(summary.ends_with('…'));
        assert_eq!(summarize("  "), "(no final summary)");
    }

    #[test]
    fn render_outcomes_includes_cache_percentage_and_write_scope() {
        let outcomes = vec![SubagentOutcome {
            ok: true,
            profile: SubagentProfile::Worker,
            write_scope: Some(vec!["src/lib.rs".into(), "tests/".into()]),
            turns: 2,
            usage: Usage {
                input_tokens: 1_000,
                output_tokens: 120,
                cached_tokens: 750,
                reasoning_tokens: 0,
            },
            summary: "updated src/lib.rs".into(),
            session: PathBuf::from("/tmp/sub.jsonl"),
        }];
        let rendered = render_outcomes(&outcomes);
        assert!(rendered.contains("worker · 2 turn(s) · 1000 in / 120 out · 75% cached · scope=src/lib.rs,tests/"));
        assert!(rendered.contains("updated src/lib.rs"));
    }
}

#[cfg(test)]
mod worktree_tests {
    use super::*;
    use std::sync::Arc;

    fn temp_dir(tag: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("niucode-worktree-{tag}-{stamp}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp");
        path
    }

    fn git(directory: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(args)
            .output()
            .expect("git");
        assert!(status.status.success(), "git {args:?}");
    }

    fn repo(tag: &str) -> (PathBuf, AgentConfig) {
        let directory = temp_dir(tag);
        git(&directory, &["init", "-q"]);
        git(
            &directory,
            &["config", "user.email", "test@example.invalid"],
        );
        git(&directory, &["config", "user.name", "test"]);
        std::fs::write(directory.join("file.txt"), "base\n").expect("base");
        git(&directory, &["add", "."]);
        git(&directory, &["commit", "-qm", "base"]);
        let root = directory.join("worktrees");
        let mut config = AgentConfig::new("test", directory.clone());
        config.worktree_root = Some(root);
        (directory, config)
    }

    #[tokio::test]
    async fn two_workers_get_independent_worktrees_and_patches() {
        let (directory, config) = repo("two");
        let first = Worktree::create(&config, "worker-a")
            .await
            .expect("first worktree");
        let second = Worktree::create(&config, "worker-b")
            .await
            .expect("second worktree");
        assert_ne!(first.path, second.path);
        std::fs::write(first.path.join("file.txt"), "first\n").expect("first write");
        std::fs::write(second.path.join("file.txt"), "second\n").expect("second write");
        let first_patch = first.finish().await.expect("first patch");
        let second_patch = second.finish().await.expect("second patch");
        assert!(std::fs::read_to_string(first_patch)
            .expect("first patch text")
            .contains("first"));
        assert!(std::fs::read_to_string(second_patch)
            .expect("second patch text")
            .contains("second"));
        assert_eq!(
            std::fs::read_to_string(directory.join("file.txt")).expect("parent file"),
            "base\n",
            "worker worktree leaked into parent"
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn subagent_config_never_inherits_mcp_host() {
        let host = Arc::new(
            niucode_mcp::McpHost::mount(&[], std::time::Duration::from_secs(1))
                .await
                .expect("empty host"),
        );
        let mut parent = AgentConfig::new("test", ".");
        parent.mcp_host = Some(host);
        let task = TaskSpec {
            prompt: "parent".into(),
            profile: SubagentProfile::Explore,
            write_scope: None,
            worktree: false,
            max_steps: 1,
            timeout_secs: 30,
        };
        assert!(subagent_config(&parent, &task).mcp_host.is_none());
    }

    #[tokio::test]
    async fn non_git_workspace_falls_back_to_shared_workdir() {
        let directory = temp_dir("non-git");
        let mut config = AgentConfig::new("test", directory.clone());
        config.worktree_root = Some(directory.join("worktrees"));
        assert!(Worktree::create(&config, "fallback").await.is_none());
        std::fs::remove_dir_all(directory).expect("cleanup");
    }
}
