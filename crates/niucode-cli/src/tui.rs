//! Linear in-process TUI.
//!
//! It only consumes `AgentEvent`s and calls the same `run_with_history` entry
//! point as headless exec. No loop internals are imported here.

use std::fs::File;
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use crossterm::cursor::{MoveTo, Show};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, Event as TerminalEvent, KeyCode, KeyEvent,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen};
use niucode_agent::{
    build_handoff, estimate_context, project_history, run_with_attachments, run_with_history,
    AgentConfig, AgentEvent, ApprovalMode, ApprovalRequest, RunOutcome,
};
use niucode_attachments::{check_budget, AttachmentStore};
use niucode_core::{load, now_ms, AttachmentRef, Event, EventId, SessionLog, Usage};
use niucode_provider::{ChatMessage, ModelSelection, Provider};
use niucode_tools::{NetworkMode, SandboxMode, SandboxPolicy, Toolbox};
use similar::TextDiff;

use crate::diff::render_plain;
use crate::editor::{mouse_events_seen, Input, LineEditor};
use crate::goal::{self, GoalStatus, MAX_GOAL_TURNS};
use crate::provider_factory::{available_model_choices, provider_choices, ProviderFactory};
use crate::review::{choice_for_key, review_diff, ApprovalChoice};
use crate::session_ops;
use crate::skills;
use crate::slash::{
    self, Command as SlashCommand, CommandRegistry, ComputerCommand, GoalAction,
    Parsed as SlashParsed, PlanAction,
};
use crate::text_layout::clean_text;
use crate::theme::{self, Theme, Tone};
use crate::workspace::Workspace;
use crate::{ApprovalKeys, SharedHooks};

const TOOL_OUTPUT_FOLD_LINES: usize = 12;
const TOOL_OUTPUT_FOLD_BYTES: usize = 2_000;
struct ToolEcho {
    name: String,
    summary: String,
    output: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermissionProfile {
    Ask,
    Review,
    FullAccess,
    Ci,
}

impl PermissionProfile {
    const ALL: [Self; 4] = [Self::Ask, Self::Review, Self::FullAccess, Self::Ci];

    fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Review => "review",
            Self::FullAccess => "full-access",
            Self::Ci => "ci",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value
            .trim()
            .trim_start_matches(':')
            .to_ascii_lowercase()
            .as_str()
        {
            "ask" => Some(Self::Ask),
            "review" => Some(Self::Review),
            "full-access" | "full" => Some(Self::FullAccess),
            "ci" => Some(Self::Ci),
            _ => None,
        }
    }

    fn apply(self, config: &mut AgentConfig) {
        match self {
            Self::Ask => {
                config.sandbox = SandboxPolicy::new(SandboxMode::WorkspaceWrite, NetworkMode::Off);
                config.approval = ApprovalMode::Ask;
                config.allow_unsandboxed = false;
            }
            Self::Review => {
                config.sandbox = SandboxPolicy::new(SandboxMode::WorkspaceWrite, NetworkMode::Off);
                config.approval = ApprovalMode::OnFailure;
                config.allow_unsandboxed = false;
            }
            Self::FullAccess => {
                config.sandbox = SandboxPolicy::new(SandboxMode::FullAccess, NetworkMode::On);
                config.approval = ApprovalMode::Never;
                config.allow_unsandboxed = false;
            }
            Self::Ci => {
                config.sandbox = SandboxPolicy::new(SandboxMode::ReadOnly, NetworkMode::Off);
                config.approval = ApprovalMode::Never;
                config.allow_unsandboxed = false;
            }
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Ask => "workspace-write · network off · ask before risk",
            Self::Review => "workspace-write · network off · review failures",
            Self::FullAccess => "full-access · network on · no approval",
            Self::Ci => "read-only · network off · no approval",
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SessionUiState {
    plan_mode: bool,
    goal: Option<String>,
    goal_run: bool,
    goal_turns: usize,
    pending_prompt: Option<String>,
    /// Last `/init` proposal; applying is explicit.
    init_proposal: Option<String>,
    /// Ephemeral request from `/reset` to drop the line editor history.
    clear_history: bool,
    /// Runtime provider/model override; None means the startup config values.
    provider: Option<String>,
    model: Option<String>,
    /// Runtime permission profile; None means the startup policy.
    permissions: Option<PermissionProfile>,
}

impl SessionUiState {
    fn from_session(session: &SessionLog) -> Result<Self> {
        let loaded = load(session.path())?;
        Ok(Self::from_events(&loaded.events))
    }

    fn from_events(events: &[Event]) -> Self {
        let mut state = Self::default();
        for event in events {
            match event {
                Event::SessionStart { provider, .. } if !provider.is_empty() => {
                    state.provider = Some(provider.clone());
                }
                Event::PlanMode { active, .. } => state.plan_mode = *active,
                Event::GoalSet { text, .. } => state.goal = Some(text.clone()),
                Event::GoalCleared { .. } => state.goal = None,
                Event::ModelChange {
                    provider, model, ..
                } => {
                    if !provider.is_empty() {
                        state.provider = Some(provider.clone());
                    }
                    state.model = Some(model.clone());
                }
                Event::PermissionChange { profile, .. } => {
                    state.permissions = PermissionProfile::parse(profile);
                }
                Event::InitProposal { content, .. } => {
                    state.init_proposal = Some(content.clone());
                }
                _ => {}
            }
        }
        state
    }
}

fn effective_model<'a>(state: &'a SessionUiState, config: &'a AgentConfig) -> &'a str {
    state.model.as_deref().unwrap_or(&config.model)
}

fn effective_provider<'a>(state: &'a SessionUiState, config: &'a AgentConfig) -> &'a str {
    state.provider.as_deref().unwrap_or(&config.provider)
}

fn current_selection(state: &SessionUiState, config: &AgentConfig) -> String {
    format!(
        "{}/{}",
        effective_provider(state, config),
        effective_model(state, config)
    )
}

fn apply_runtime_config(config: &AgentConfig, state: &SessionUiState) -> AgentConfig {
    let mut runtime = config.clone();
    runtime.provider = effective_provider(state, config).to_string();
    runtime.model = effective_model(state, config).to_string();
    if runtime.context_budget_auto {
        runtime.context.budget_tokens =
            niucode_provider::derived_context_budget(&runtime.provider, &runtime.model);
    }
    if let Some(profile) = state.permissions {
        profile.apply(&mut runtime);
    }
    runtime.goal = state.goal.clone();
    if state.plan_mode {
        runtime.allowed_tools = Some(vec![
            "read".to_string(),
            "grep".to_string(),
            "glob".to_string(),
        ]);
    }
    runtime
}

fn ghostty_terminal() -> bool {
    std::env::var("TERM_PROGRAM")
        .map(|value| value.eq_ignore_ascii_case("ghostty"))
        .unwrap_or(false)
        || std::env::var("TERM")
            .map(|value| value.to_ascii_lowercase().contains("ghostty"))
            .unwrap_or(false)
        || std::env::var_os("GHOSTTY_RESOURCES_DIR").is_some()
}

/// Inline ratatui is still being hardened against terminal-specific cursor
/// behavior. Ghostty currently falls back to the stable linear editor unless
/// the user explicitly forces inline mode.
fn inline_tui_allowed() -> bool {
    if std::env::var_os("NIUCODE_TUI_LINEAR").is_some() {
        return false;
    }
    if std::env::var_os("NIUCODE_TUI_INLINE").is_some() {
        return true;
    }
    !ghostty_terminal()
}

/// Expand user-typed `@path` tokens only when the path exists. Missing tokens
/// stay literal so email handles and nonexistent paths are not swallowed.
fn split_at_paths(input: &str) -> (String, Vec<PathBuf>) {
    let mut kept = Vec::new();
    let mut paths = Vec::new();
    for token in input.split_whitespace() {
        if let Some(path) = token.strip_prefix('@') {
            if !path.is_empty() && Path::new(path).exists() {
                paths.push(PathBuf::from(path));
                continue;
            }
        }
        kept.push(token);
    }
    (kept.join(" "), paths)
}

fn print_attachment_chips(attachments: &[AttachmentRef], out: Theme) {
    if attachments.is_empty() {
        eprintln!("{}", out.dim("no pending attachments"));
        return;
    }
    for (index, attachment) in attachments.iter().enumerate() {
        eprintln!(
            "{} [{}] {:?} · {} · {} B",
            out.dim("⬒"),
            index + 1,
            attachment.kind,
            attachment
                .name
                .as_deref()
                .unwrap_or(&attachment.sha256[..8]),
            attachment.bytes
        );
    }
}

pub async fn run(
    mut provider: Arc<dyn Provider>,
    provider_factory: Option<&ProviderFactory>,
    session: &mut SessionLog,
    config: &AgentConfig,
    history: Vec<ChatMessage>,
    hooks: Option<SharedHooks>,
    approval_keys: ApprovalKeys,
) -> Result<()> {
    reset_terminal_modes();
    let out = Theme::detect_stdout();
    let err = Theme::detect_stderr();
    let mut commands = CommandRegistry::builtin();
    if let Some(hooks) = &hooks {
        commands.extend(hooks.borrow().commands());
    }
    let mut editor = if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        let inline = inline_tui_allowed();
        let mut editor = LineEditor::new(commands.clone(), inline);
        editor.set_models(niucode_model_choices(config, provider_factory));
        editor.set_current_model(format!("{}/{}", config.provider, config.model));
        editor.set_cwd(config.working_dir.clone());
        editor.set_permissions(
            PermissionProfile::ALL
                .iter()
                .map(|profile| {
                    (
                        profile.as_str().to_string(),
                        profile.description().to_string(),
                    )
                })
                .collect(),
        );
        editor.set_skills(trusted_skill_names(config));
        Some(editor)
    } else {
        None
    };

    let mut state = SessionUiState::from_session(session)?;
    let mut history = history;
    let mut allow_session = false;
    let mut last_tool: Option<ToolEcho> = None;
    let mut pending_attachments: Vec<AttachmentRef> = Vec::new();
    let mut attachment_store: Option<AttachmentStore> = None;

    loop {
        let input = if let Some(prompt) = state.pending_prompt.take() {
            eprintln!(
                "{} {}",
                out.accent("↻ goal run"),
                out.dim(format!(
                    "turn {}/{} · {}",
                    state.goal_turns + 1,
                    MAX_GOAL_TURNS,
                    state.goal.as_deref().unwrap_or("")
                ))
            );
            prompt
        } else if let Some(editor) = &mut editor {
            editor.set_current_model(current_selection(&state, config));
            println!();
            let mode = if state.plan_mode {
                Some("plan")
            } else if state.goal_run {
                Some("goal")
            } else {
                None
            };
            let raw_prompt = match mode {
                Some(label) => format!("niucode {} {label} ", theme::PROMPT),
                None => format!("niucode {} ", theme::PROMPT),
            };
            let footer = editor_footer(&state, config);
            match editor.read_line(&raw_prompt, &footer, out)? {
                Input::Line(line) | Input::Queue(line) => line,
                Input::Details => {
                    crate::live::show_details(session)?;
                    continue;
                }
                Input::Interrupted => continue,
                Input::Eof => break,
            }
        } else {
            print!("\n{} ", out.accent(format!("niucode {}", theme::PROMPT)));
            std::io::stdout().flush()?;
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input)? == 0 {
                break;
            }
            input
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (prompt_owned, at_paths) = if trimmed.contains('@') {
            split_at_paths(trimmed)
        } else {
            (trimmed.to_string(), Vec::new())
        };
        if !at_paths.is_empty() {
            if attachment_store.is_none() {
                let root = session
                    .path()
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join("attachments");
                match AttachmentStore::open(root) {
                    Ok(store) => attachment_store = Some(store),
                    Err(error) => {
                        eprintln!("{} attach: {error:#}", err.paint(Tone::Danger, theme::ERR));
                        continue;
                    }
                }
            }
            let store = attachment_store.as_ref().expect("store checked above");
            match crate::ingest_attachments(&at_paths, None, store) {
                Ok(mut refs) => {
                    pending_attachments.append(&mut refs);
                    if let Err(error) =
                        check_budget(&pending_attachments, config.context.budget_tokens)
                    {
                        eprintln!("{} {error}", err.paint(Tone::Warning, theme::WARN));
                    } else {
                        print_attachment_chips(&pending_attachments, out);
                    }
                }
                Err(error) => {
                    eprintln!("{} attach: {error:#}", err.paint(Tone::Danger, theme::ERR));
                    continue;
                }
            }
        }
        if let SlashParsed::Command(SlashCommand::Computer(command)) = slash::parse(&prompt_owned) {
            match config.computer.as_ref() {
                Some(runtime) => {
                    let mut runtime = runtime.lock().unwrap();
                    match command {
                        ComputerCommand::Status => eprintln!(
                            "computer: policy={} armed={}",
                            runtime.policy().as_str(),
                            runtime.is_armed()
                        ),
                        ComputerCommand::Arm => match runtime.arm() {
                            Ok(()) => eprintln!("computer: armed"),
                            Err(error) => eprintln!("computer: cannot arm: {error}"),
                        },
                        ComputerCommand::Stop => match runtime.emergency_stop() {
                            Ok(()) => eprintln!("computer: stopped, disarmed, input released"),
                            Err(error) => eprintln!("computer: stopped (release failed: {error})"),
                        },
                    }
                }
                None => {
                    eprintln!("computer: off (start with --computer observe|control|autonomous)")
                }
            }
            continue;
        }
        if prompt_owned.is_empty() {
            continue;
        }
        let prompt = prompt_owned.as_str();
        match slash::parse(prompt) {
            SlashParsed::Command(SlashCommand::Attach(path)) => {
                if attachment_store.is_none() {
                    let root = session
                        .path()
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new("."))
                        .join("attachments");
                    match AttachmentStore::open(root) {
                        Ok(store) => attachment_store = Some(store),
                        Err(error) => {
                            eprintln!("{} attach: {error:#}", err.paint(Tone::Danger, theme::ERR));
                            continue;
                        }
                    }
                }
                let store = attachment_store.as_ref().expect("store checked above");
                match crate::ingest_attachments(&[path], None, store) {
                    Ok(mut refs) => {
                        pending_attachments.append(&mut refs);
                        if let Err(error) =
                            check_budget(&pending_attachments, config.context.budget_tokens)
                        {
                            eprintln!("{} {error}", err.paint(Tone::Warning, theme::WARN));
                        } else {
                            print_attachment_chips(&pending_attachments, out);
                        }
                    }
                    Err(error) => {
                        eprintln!("{} attach: {error:#}", err.paint(Tone::Danger, theme::ERR))
                    }
                }
                continue;
            }
            SlashParsed::Command(SlashCommand::Detach(index)) => {
                match index {
                    Some(index) if index <= pending_attachments.len() => {
                        pending_attachments.remove(index - 1);
                    }
                    Some(index) => eprintln!(
                        "{} no attachment {index}",
                        err.paint(Tone::Warning, theme::WARN)
                    ),
                    None => pending_attachments.clear(),
                }
                print_attachment_chips(&pending_attachments, out);
                continue;
            }
            _ => {}
        }
        match slash::parse(prompt) {
            SlashParsed::Command(command) => match handle_command(
                command,
                session,
                config,
                &mut history,
                &mut allow_session,
                &mut last_tool,
                &mut state,
                &commands,
                &mut provider,
                provider_factory,
                hooks.as_ref(),
                out,
                err,
            )
            .await
            {
                Ok(CommandOutcome::Exit) => break,
                Ok(CommandOutcome::Continue) => {
                    if let Some(editor) = &mut editor {
                        if state.clear_history {
                            editor.clear_history();
                        }
                        editor.set_current_model(current_selection(&state, config));
                        editor.set_models(niucode_model_choices(config, provider_factory));
                        editor.set_skills(trusted_skill_names(config));
                    }
                    state.clear_history = false;
                    continue;
                }
                Err(error) => {
                    eprintln!(
                        "{} {}",
                        err.paint(Tone::Danger, theme::ERR),
                        err.paint(Tone::Danger, format!("{error:#}"))
                    );
                    continue;
                }
            },
            SlashParsed::Unknown(name) => {
                let handled = hooks
                    .as_ref()
                    .and_then(|hooks| hooks.borrow_mut().command(prompt));
                if let Some(output) = handled {
                    if !output.is_empty() {
                        println!("{output}");
                    }
                } else {
                    eprintln!(
                        "{} unknown command /{name}; try {}",
                        err.paint(Tone::Warning, theme::WARN),
                        err.paint(Tone::Bright, "/help")
                    );
                }
                continue;
            }
            SlashParsed::Usage(usage) => {
                eprintln!(
                    "{} usage: {}",
                    err.paint(Tone::Warning, theme::WARN),
                    err.paint(Tone::Bright, usage)
                );
                continue;
            }
            SlashParsed::None => {}
        }

        let mut line_open = false;
        let mut thinking = false;
        let mut text_started = false;
        let mut turn_no = 0_usize;
        let mut pending_usage: Option<Usage> = None;
        let mut tools_pending = 0_usize;

        if let Some(factory) = provider_factory {
            let selected = effective_provider(&state, config);
            if provider.name() != selected {
                provider = factory
                    .build(selected, effective_model(&state, config))
                    .with_context(|| format!("switch provider to {selected}"))?;
                eprintln!(
                    "{} provider {} · auth resolved",
                    out.accent("switch"),
                    out.paint(Tone::Bright, selected)
                );
            }
        }
        let mut run_config = apply_runtime_config(config, &state);
        run_config.plan_mode = state.plan_mode;
        if let Ok(workspace) = Workspace::discover(&config.working_dir) {
            if let Ok(registry) = skills::Registry::discover(&workspace) {
                let (block, roots) = registry.activate(prompt);
                if !block.is_empty() {
                    run_config.system_prompt.push_str(&block);
                    run_config.read_roots.extend(roots);
                }
            }
        }

        let result = if let Some(editor) = &mut editor {
            Some(
                crate::live::run(
                    provider.as_ref(),
                    session,
                    &run_config,
                    prompt,
                    history.clone(),
                    pending_attachments.clone(),
                    attachment_store.as_ref(),
                    editor,
                    &mut allow_session,
                    hooks.clone(),
                    approval_keys,
                )
                .await,
            )
        } else {
            let mut approver = |request: &ApprovalRequest| -> bool {
                if allow_session {
                    eprintln!(
                        "{} {}",
                        err.paint(Tone::Success, theme::OK),
                        err.dim(format!("auto-approved · {}", clean_text(&request.summary)))
                    );
                    return true;
                }
                match prompt_approval(request, out, err, approval_keys) {
                    ApprovalChoice::Allow => true,
                    ApprovalChoice::Deny => false,
                    ApprovalChoice::Session => {
                        allow_session = true;
                        true
                    }
                }
            };

            let hook_cell = hooks.clone();
            let mut observer = |event: AgentEvent| {
                if let Some(hooks) = &hook_cell {
                    hooks.borrow_mut().on_event(&event);
                }
                match event {
                    AgentEvent::Phase(_) => {}
                    AgentEvent::TextDelta(delta) => {
                        if delta.is_empty() {
                            return;
                        }
                        if !text_started {
                            if line_open {
                                println!();
                            }
                            if thinking {
                                eprintln!();
                            }
                            flush_usage(err, &mut pending_usage, turn_no);
                            print!("{} ", out.accent(theme::ASSISTANT));
                            text_started = true;
                            thinking = false;
                        }
                        print!("{delta}");
                        let _ = std::io::stdout().flush();
                        line_open = !delta.ends_with('\n');
                    }
                    AgentEvent::ReasoningDelta(delta) => {
                        if delta.is_empty() {
                            return;
                        }
                        if !thinking {
                            if line_open {
                                println!();
                                line_open = false;
                            }
                            flush_usage(err, &mut pending_usage, turn_no);
                            eprintln!("\r{}", err.dim(format!("{} thinking", theme::THINKING)));
                            thinking = true;
                        }
                    }
                    AgentEvent::ToolCall { name, input } => {
                        if thinking {
                            eprintln!();
                        }
                        if line_open {
                            println!();
                            line_open = false;
                        }
                        text_started = false;
                        thinking = false;
                        tools_pending += 1;
                        let detail = compact_json(&input);
                        let mut line = format!(
                            "{} {}",
                            err.paint(Tone::Magenta, theme::TOOL),
                            err.paint(Tone::Bright, name)
                        );
                        if !detail.is_empty() {
                            line.push(' ');
                            line.push_str(&err.dim(detail));
                        }
                        eprintln!("{line}");
                    }
                    AgentEvent::ToolResult {
                        name,
                        ok,
                        summary,
                        output,
                    } => {
                        if thinking {
                            eprintln!();
                        }
                        if line_open {
                            println!();
                            line_open = false;
                        }
                        text_started = false;
                        thinking = false;
                        render_tool_result(err, &name, ok, &summary, &output, &mut last_tool);
                        tools_pending = tools_pending.saturating_sub(1);
                        if tools_pending == 0 {
                            flush_usage(err, &mut pending_usage, turn_no);
                        }
                    }
                    AgentEvent::TurnEnd { usage } => {
                        if thinking {
                            eprintln!();
                        }
                        if line_open {
                            println!();
                            line_open = false;
                        }
                        text_started = false;
                        thinking = false;
                        turn_no += 1;
                        pending_usage = Some(usage);
                    }
                    AgentEvent::Compaction {
                        replaced_events,
                        before_tokens,
                        after_tokens,
                    } => {
                        if thinking {
                            eprintln!();
                            thinking = false;
                        }
                        if line_open {
                            println!();
                            line_open = false;
                        }
                        text_started = false;
                        flush_usage(err, &mut pending_usage, turn_no);
                        eprintln!(
                    "{} {}",
                    err.paint(Tone::Magenta, theme::COMPACT),
                    err.dim(format!(
                        "compacted · {replaced_events} events · ~{before_tokens} → ~{after_tokens} tokens"
                    ))
                );
                    }
                    AgentEvent::Warning(message) => {
                        if thinking {
                            eprintln!();
                            thinking = false;
                        }
                        if line_open {
                            println!();
                            line_open = false;
                        }
                        text_started = false;
                        flush_usage(err, &mut pending_usage, turn_no);
                        eprintln!(
                            "{} {}",
                            err.paint(Tone::Warning, theme::WARN),
                            err.paint(Tone::Warning, clean_text(&message))
                        );
                    }
                }
            };

            let result = {
                let run = async {
                    if let Some(store) = attachment_store.as_ref() {
                        run_with_attachments(
                            provider.as_ref(),
                            session,
                            &run_config,
                            prompt,
                            pending_attachments.clone(),
                            store,
                            &mut approver,
                            &mut observer,
                        )
                        .await
                    } else {
                        run_with_history(
                            provider.as_ref(),
                            session,
                            &run_config,
                            prompt,
                            history.clone(),
                            &mut approver,
                            &mut observer,
                        )
                        .await
                    }
                };
                let run = Box::pin(run);
                tokio::pin!(run);
                tokio::select! {
                    result = &mut run => Some(result),
                    _ = tokio::signal::ctrl_c() => {
                        crate::stop_computer(&run_config);
                        None
                    }
                }
            };
            if let Some(Ok(outcome)) = &result {
                if let Some(hooks) = &hook_cell {
                    hooks.borrow_mut().on_turn_end(outcome);
                }
            }
            result
        };
        pending_attachments.clear();

        if line_open {
            println!();
        }
        flush_usage(err, &mut pending_usage, turn_no);

        match result {
            None => {
                state.goal_run = false;
                state.pending_prompt = None;
                let _ = session.append(&Event::SessionEnd { ts: now_ms() });
                history = project_history(&load(session.path())?.events)?;
                eprintln!(
                    "{}",
                    err.paint(
                        Tone::Warning,
                        format!("{} cancelled · Ctrl-C again within 2s to exit", theme::WARN)
                    )
                );
            }
            Some(Ok(outcome)) => {
                if editor.is_none() {
                    eprintln!(
                        "{} {}",
                        err.paint(Tone::Success, theme::OK),
                        err.dim(format!(
                            "done · {} turns · in {} · out {} · cache {}%",
                            outcome.turns,
                            outcome.usage.input_tokens,
                            outcome.usage.output_tokens,
                            outcome.usage.cache_hit_percent()
                        ))
                    );
                }
                run_turn_prompts(&hooks, &outcome);
                history = project_history(&load(session.path())?.events)?;
                if state.goal_run {
                    if let Err(error) = goal_after_turn(
                        provider.as_ref(),
                        session,
                        &mut state,
                        &outcome,
                        &run_config.model,
                        out,
                        err,
                    )
                    .await
                    {
                        state.goal_run = false;
                        state.pending_prompt = None;
                        eprintln!(
                            "{} {}",
                            err.paint(Tone::Warning, theme::WARN),
                            err.paint(Tone::Warning, format!("goal evaluator failed: {error:#}"))
                        );
                    }
                }
            }
            Some(Err(error)) => {
                state.goal_run = false;
                state.pending_prompt = None;
                let message = format!("{error:#}");
                let cancelled = error.is::<niucode_agent::Cancelled>();
                let tone = if cancelled {
                    Tone::Warning
                } else {
                    Tone::Danger
                };
                // Live terminal guards have unwound here, including early I/O failures.
                // Print each error cause once, without allowing an unbounded terminal dump.
                if editor.is_some() {
                    let width = terminal::size()
                        .map_or(80, |s| s.0 as usize)
                        .saturating_sub(1)
                        .max(1);
                    eprintln!(
                        "{}",
                        err.paint(
                            tone,
                            crate::text_layout::clip(
                                &clean_text(&format!(
                                    "{} {message}",
                                    if cancelled { theme::WARN } else { theme::ERR }
                                )),
                                width
                            )
                        )
                    );
                } else {
                    eprintln!(
                        "{} {}",
                        err.paint(tone, if cancelled { theme::WARN } else { theme::ERR }),
                        err.paint(tone, &message)
                    );
                    if message.contains("max steps reached") || message.contains("safety turn cap")
                    {
                        let current = if config.max_steps == 0 {
                            "unbounded (safety 256)".to_string()
                        } else {
                            config.max_steps.to_string()
                        };
                        eprintln!(
                        "{}",
                        err.paint(
                            Tone::Warning,
                            format!(
                                "hint: --max-steps is currently {current}; project writes are preserved"
                            )
                        )
                    );
                    }
                }
                history = project_history(&load(session.path())?.events)?;
            }
        }
    }

    Ok(())
}

enum CommandOutcome {
    Continue,
    Exit,
}

#[allow(clippy::too_many_arguments)]
async fn handle_command(
    command: SlashCommand,
    session: &mut SessionLog,
    config: &AgentConfig,
    history: &mut Vec<ChatMessage>,
    allow_session: &mut bool,
    last_tool: &mut Option<ToolEcho>,
    state: &mut SessionUiState,
    commands: &CommandRegistry,
    provider: &mut Arc<dyn Provider>,
    provider_factory: Option<&ProviderFactory>,
    hooks: Option<&SharedHooks>,
    out: Theme,
    err: Theme,
) -> Result<CommandOutcome> {
    match command {
        SlashCommand::Attach(_) | SlashCommand::Detach(_) | SlashCommand::Computer(_) => {}
        SlashCommand::Help => print_help(out, commands),
        SlashCommand::Last => {
            if std::io::stdout().is_terminal() {
                crate::live::show_details(session)?;
            } else {
                print_last(out, last_tool.as_ref());
            }
        }
        SlashCommand::Init { apply } => {
            if apply {
                run_init_apply(session, config, state, out, err).await?;
            } else {
                run_init_generate(provider.as_ref(), session, config, state, out, err).await?;
            }
        }
        SlashCommand::Skills(arg) => handle_skills(config, state, arg, out)?,
        SlashCommand::Tools => handle_tools(config, out),
        SlashCommand::Plan(action) => handle_plan(session, state, action, out)?,
        SlashCommand::Goal(action) => handle_goal(session, state, action, out)?,
        SlashCommand::Clear => {
            let previous = session.path().to_path_buf();
            let plan = state.plan_mode;
            let goal = state.goal.clone();
            let model = state.model.clone();
            let provider = state.provider.clone();
            let permissions = state.permissions;
            let custom = match hooks {
                Some(hooks) => hooks.borrow_mut().new_session_path()?,
                None => None,
            };
            let fresh = new_session(session, config, custom)?;
            notify_session(hooks, fresh.path());
            if plan {
                fresh.append(&Event::PlanMode {
                    active: true,
                    ts: now_ms(),
                })?;
            }
            if let Some(goal) = &goal {
                fresh.append(&Event::GoalSet {
                    text: goal.clone(),
                    ts: now_ms(),
                })?;
            }
            if let Some(model) = &model {
                fresh.append(&Event::ModelChange {
                    provider: provider.clone().unwrap_or_else(|| config.provider.clone()),
                    model: model.clone(),
                    ts: now_ms(),
                })?;
            }
            if let Some(profile) = permissions {
                let mut applied = config.clone();
                profile.apply(&mut applied);
                fresh.append(&Event::PermissionChange {
                    profile: profile.as_str().to_string(),
                    sandbox: applied.sandbox.mode.as_str().to_string(),
                    approval: applied.approval.as_str().to_string(),
                    network: applied.sandbox.network.as_str().to_string(),
                    ts: now_ms(),
                })?;
            }
            *session = fresh;
            history.clear();
            *allow_session = false;
            *last_tool = None;
            state.plan_mode = plan;
            state.goal = goal;
            state.model = model;
            state.provider = provider;
            state.permissions = permissions;
            state.goal_run = false;
            state.goal_turns = 0;
            state.pending_prompt = None;
            reset_screen(err);
            println!(
                "{} {}",
                out.paint(Tone::Success, theme::OK),
                out.dim(format!(
                    "new conversation · {} · kept plan={} goal={} model={} permissions={}",
                    session.path().display(),
                    state.plan_mode,
                    state.goal.is_some(),
                    effective_model(state, config),
                    state
                        .permissions
                        .map(PermissionProfile::as_str)
                        .unwrap_or("startup")
                ))
            );
            println!("{}", out.dim(format!("previous · {}", previous.display())));
        }
        SlashCommand::Reset => {
            let previous = session.path().to_path_buf();
            let custom = match hooks {
                Some(hooks) => hooks.borrow_mut().new_session_path()?,
                None => None,
            };
            *session = new_session(session, config, custom)?;
            notify_session(hooks, session.path());
            history.clear();
            *allow_session = false;
            *last_tool = None;
            *state = SessionUiState::default();
            state.clear_history = true;
            reset_screen(err);
            println!(
                "{} {}",
                out.paint(Tone::Success, theme::OK),
                out.dim(format!(
                    "runtime reset · new conversation · plan/goal/history cleared · {}",
                    session.path().display()
                ))
            );
            println!("{}", out.dim(format!("previous · {}", previous.display())));
            reload_instructions(config, out, err)?;
        }
        SlashCommand::Reload => reload_instructions(config, out, err)?,
        SlashCommand::Model(arg) => {
            handle_model(session, config, state, arg, provider, provider_factory, out)?
        }
        SlashCommand::Permissions(arg) => {
            handle_permissions(session, config, state, arg, out, err)?
        }
        SlashCommand::Compact => compact_now(session, config, out, err)?,
        SlashCommand::Resume(arg) => {
            resume_session(
                session,
                config,
                history,
                allow_session,
                last_tool,
                state,
                arg.as_deref(),
                out,
                err,
            )?;
            notify_session(hooks, session.path());
        }
        SlashCommand::Fork(at) => {
            let report = fork_session(session, at.as_deref())?;
            for warning in &report.warnings {
                eprintln!(
                    "{} {}",
                    err.paint(Tone::Warning, theme::WARN),
                    err.dim(warning)
                );
            }
            *session = SessionLog::open(&report.destination)?;
            let loaded = load(session.path())?;
            *history = project_history(&loaded.events)?;
            *state = SessionUiState::from_events(&loaded.events);
            *allow_session = false;
            *last_tool = None;
            notify_session(hooks, session.path());
            reset_screen(err);
            println!(
                "{} {}",
                out.paint(Tone::Success, theme::OK),
                out.dim(format!(
                    "fork · {} · copied {} {}",
                    report.destination.display(),
                    report.copied,
                    if report.copied == 1 {
                        "event"
                    } else {
                        "events"
                    }
                ))
            );
            if report.attachments > 0 {
                println!(
                    "{}",
                    out.dim(format!("attachment(s) copied: {}", report.attachments))
                );
            }
            println!(
                "{}",
                out.dim(format!(
                    "at {} · parent {}",
                    report.at,
                    report.parent.display()
                ))
            );
        }
        SlashCommand::Status => print_status(session, config, out)?,
        SlashCommand::Session => print_session(session, out)?,
        SlashCommand::Exit => return Ok(CommandOutcome::Exit),
    }
    Ok(CommandOutcome::Continue)
}

fn compact_now(
    session: &mut SessionLog,
    config: &AgentConfig,
    out: Theme,
    err: Theme,
) -> Result<()> {
    let loaded = load(session.path())?;
    let before_history = project_history(&loaded.events)?;
    let before = estimate_context(&before_history, &[]);
    let handoff = build_handoff(&loaded.events, config.context.keep_recent_turns)?;
    let mut candidate = loaded.events.clone();
    candidate.push(Event::Compaction {
        summary: handoff.summary.clone(),
        upto: handoff.upto as u64,
        before_tokens: before as u64,
        after_tokens: 0,
        ts: now_ms(),
    });
    let after_history = project_history(&candidate)?;
    let after = estimate_context(&after_history, &[]);
    if after >= before {
        bail!("compaction would not reduce context (~{before} → ~{after} tokens)");
    }
    session.append(&Event::Compaction {
        summary: handoff.summary,
        upto: handoff.upto as u64,
        before_tokens: before as u64,
        after_tokens: after as u64,
        ts: now_ms(),
    })?;
    println!(
        "{} {}",
        out.paint(Tone::Success, theme::OK),
        out.paint(
            Tone::Bright,
            format!(
                "compacted {} events · ~{before} → ~{after} tokens",
                handoff.upto
            )
        )
    );
    println!(
        "{}",
        err.dim("  next turn uses the new projection; raw log is unchanged")
    );
    Ok(())
}

fn niucode_model_choices(_config: &AgentConfig, factory: Option<&ProviderFactory>) -> Vec<String> {
    let from_env = std::env::var("NIUCODE_MODEL_CHOICES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !from_env.is_empty() {
        return from_env;
    }
    factory
        .map(available_model_choices)
        .filter(|choices| !choices.is_empty())
        .unwrap_or_else(provider_choices)
}

fn handle_model(
    session: &mut SessionLog,
    config: &AgentConfig,
    state: &mut SessionUiState,
    arg: Option<String>,
    provider: &mut Arc<dyn Provider>,
    factory: Option<&ProviderFactory>,
    out: Theme,
) -> Result<()> {
    let current_provider = effective_provider(state, config).to_string();
    let current_model = effective_model(state, config).to_string();
    let current = format!("{current_provider}/{current_model}");
    let Some(name) = arg else {
        println!(
            "{} {}",
            out.accent("model"),
            out.paint(Tone::Bright, &current)
        );
        println!(
            "{}",
            out.dim("  switch with /model <provider/model> · next turn")
        );
        let choices = provider_choices();
        if !choices.is_empty() {
            println!("{}", out.dim("  suggestions:"));
            for choice in choices {
                let marker = if choice == current { "*" } else { " " };
                println!("{}", out.dim(format!("  {marker} {choice}")));
            }
        }
        return Ok(());
    };
    let selection =
        ModelSelection::parse(name.trim(), &current_provider).map_err(anyhow::Error::msg)?;
    if selection.provider != current_provider {
        let Some(factory) = factory else {
            bail!("cross-provider model switching is unavailable in replay mode");
        };
        if !factory.available(&selection.provider)? {
            bail!(
                "no credential for {}; run `niucode auth set {}`",
                selection.provider,
                selection.provider
            );
        }
        *provider = factory
            .build(&selection.provider, &selection.model)
            .with_context(|| format!("switch provider to {}", selection.provider))?;
    }
    if selection.provider == current_provider && selection.model == current_model {
        println!("{}", out.dim(format!("model already {current}")));
        return Ok(());
    }
    state.provider = Some(selection.provider.clone());
    state.model = Some(selection.model.clone());
    session.append(&Event::ModelChange {
        provider: selection.provider.clone(),
        model: selection.model.clone(),
        ts: now_ms(),
    })?;
    println!(
        "{} {}",
        out.accent("model"),
        out.paint(Tone::Bright, format!("{current} → {selection} · next turn"))
    );
    println!(
        "{}",
        out.dim("  switching between turns may miss the prompt cache")
    );
    Ok(())
}

fn handle_tools(config: &AgentConfig, out: Theme) {
    println!(
        "{} {}",
        out.accent("tools"),
        out.dim("built-in risk/effect metadata")
    );
    for spec in niucode_tools::Toolbox::specs() {
        let name = spec.function.name;
        match niucode_tools::metadata(&name) {
            Some(meta) => println!(
                "  {:<10} {:<7} {:<9} {}",
                name,
                meta.risk.as_str(),
                meta.effect.as_str(),
                if meta.read_only {
                    "read-only"
                } else {
                    "mutating"
                }
            ),
            None => println!("  {name:<10} unknown unknown unknown"),
        }
    }
    if let Some(host) = &config.extension_host {
        for spec in host.specs() {
            println!("  {:<10} unknown external  extension", spec.function.name);
        }
    }
}

fn handle_skills(
    config: &AgentConfig,
    state: &SessionUiState,
    arg: Option<String>,
    out: Theme,
) -> Result<()> {
    let workspace = Workspace::discover(&config.working_dir)?;
    let registry = skills::Registry::discover(&workspace)?;
    let Some(arg) = arg else {
        println!(
            "{} {}",
            out.accent("skills"),
            out.dim(format!(
                "global ~/.agents/skills · project {}",
                config.working_dir.join(".agents/skills").display()
            ))
        );
        if registry.list().is_empty() {
            println!("{}", out.dim("  no SKILL.md found"));
            return Ok(());
        }
        for skill in registry.list() {
            println!(
                "{} {}",
                out.paint(
                    if skill.trusted {
                        Tone::Success
                    } else {
                        Tone::Warning
                    },
                    if skill.trusted {
                        "trusted"
                    } else {
                        "untrusted"
                    }
                ),
                out.paint(
                    Tone::Bright,
                    format!(
                        "{} · {} · {}",
                        skill.name,
                        skill.scope.as_str(),
                        skill.description
                    )
                )
            );
            println!("{}", out.dim(format!("  {}", skill.path.display())));
        }
        println!(
            "{}",
            out.dim("  /skills trust <name> · /skills untrust <name>")
        );
        return Ok(());
    };
    let mut parts = arg.split_whitespace();
    let action = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    if name.is_empty() {
        bail!("usage: /skills [trust|untrust <name>]");
    }
    if state.plan_mode {
        bail!("plan mode is read-only; use /plan off before changing skill trust");
    }
    match action {
        "trust" => {
            registry.trust(name)?;
            println!(
                "{} {}",
                out.paint(Tone::Success, theme::OK),
                out.dim(format!("trusted project skill {name} for this project"))
            );
        }
        "untrust" => {
            registry.untrust(name)?;
            println!(
                "{} {}",
                out.paint(Tone::Success, theme::OK),
                out.dim(format!("untrusted project skill {name}"))
            );
        }
        _ => bail!("usage: /skills [trust|untrust <name>]"),
    }
    Ok(())
}

fn handle_permissions(
    session: &mut SessionLog,
    config: &AgentConfig,
    state: &mut SessionUiState,
    arg: Option<String>,
    out: Theme,
    err: Theme,
) -> Result<()> {
    let mut effective = config.clone();
    if let Some(profile) = state.permissions {
        profile.apply(&mut effective);
    }
    let Some(name) = arg else {
        println!(
            "{} {}",
            out.accent("permissions"),
            out.paint(
                Tone::Bright,
                state
                    .permissions
                    .map(PermissionProfile::as_str)
                    .unwrap_or("startup/custom")
            )
        );
        println!(
            "  sandbox {} · network {} · approval {}",
            effective.sandbox.mode.as_str(),
            effective.sandbox.network.as_str(),
            effective.approval.as_str()
        );
        println!("{}", out.dim("  switch with /permissions <profile>:"));
        for profile in PermissionProfile::ALL {
            let marker = if state.permissions == Some(profile) {
                "*"
            } else {
                " "
            };
            println!(
                "{}",
                out.dim(format!(
                    "  {marker} {:<12} {}",
                    profile.as_str(),
                    profile.description()
                ))
            );
        }
        return Ok(());
    };
    let Some(profile) = PermissionProfile::parse(&name) else {
        bail!("unknown permission profile {name:?}; use ask, review, full-access, or ci");
    };
    let mut applied = config.clone();
    profile.apply(&mut applied);
    state.permissions = Some(profile);
    session.append(&Event::PermissionChange {
        profile: profile.as_str().to_string(),
        sandbox: applied.sandbox.mode.as_str().to_string(),
        approval: applied.approval.as_str().to_string(),
        network: applied.sandbox.network.as_str().to_string(),
        ts: now_ms(),
    })?;
    println!(
        "{} {}",
        out.accent("permissions"),
        out.paint(
            Tone::Bright,
            format!(
                "{} · sandbox {} · network {} · approval {} · next turn",
                profile.as_str(),
                applied.sandbox.mode.as_str(),
                applied.sandbox.network.as_str(),
                applied.approval.as_str()
            )
        )
    );
    if profile == PermissionProfile::FullAccess {
        eprintln!(
            "{}",
            err.paint(
                Tone::Warning,
                "warning: full-access disables sandbox and network isolation"
            )
        );
    }
    Ok(())
}

fn handle_plan(
    session: &mut SessionLog,
    state: &mut SessionUiState,
    action: PlanAction,
    out: Theme,
) -> Result<()> {
    match action {
        PlanAction::Enter => {
            if state.plan_mode {
                println!("{}", out.dim("already in plan mode · /plan off to exit"));
                return Ok(());
            }
            state.plan_mode = true;
            session.append(&Event::PlanMode {
                active: true,
                ts: now_ms(),
            })?;
            println!(
                "{} {}",
                out.paint(Tone::Warning, theme::WARN),
                out.paint(Tone::Bright, "plan mode on · read-only")
            );
            println!(
                "{}",
                out.dim("  investigate and return a plan · /plan approve to exit")
            );
        }
        PlanAction::Exit | PlanAction::Approve => {
            if !state.plan_mode {
                println!("{}", out.dim("plan mode is already off"));
                return Ok(());
            }
            state.plan_mode = false;
            session.append(&Event::PlanMode {
                active: false,
                ts: now_ms(),
            })?;
            let label = if matches!(action, PlanAction::Approve) {
                "plan approved · execution available on the next prompt"
            } else {
                "plan mode off"
            };
            println!(
                "{} {}",
                out.paint(Tone::Success, theme::OK),
                out.paint(Tone::Bright, label)
            );
        }
        PlanAction::Status => {
            println!(
                "{} {}",
                out.accent("plan"),
                if state.plan_mode {
                    "on · read-only"
                } else {
                    "off"
                }
            );
        }
    }
    Ok(())
}

fn handle_goal(
    session: &mut SessionLog,
    state: &mut SessionUiState,
    action: GoalAction,
    out: Theme,
) -> Result<()> {
    match action {
        GoalAction::Show => match &state.goal {
            Some(goal) => {
                println!("{} {}", out.accent("goal"), out.paint(Tone::Bright, goal));
                if state.goal_run {
                    println!(
                        "{}",
                        out.dim(format!(
                            "  run active · turn {}/{} · /goal stop",
                            state.goal_turns, MAX_GOAL_TURNS
                        ))
                    );
                }
            }
            None => println!("{}", out.dim("no active goal · /goal <condition>")),
        },
        GoalAction::Run => {
            if state.plan_mode {
                bail!("goal run is disabled in plan mode; use /plan off first");
            }
            let Some(goal) = state.goal.clone() else {
                bail!("set a goal first: /goal <condition>");
            };
            state.goal_run = true;
            state.goal_turns = 0;
            state.pending_prompt = Some(goal::start_prompt(&goal));
            println!(
                "{} {}",
                out.accent("goal run"),
                out.paint(Tone::Bright, format!("max {MAX_GOAL_TURNS} turns"))
            );
            println!("{}", out.dim("  /goal stop to stop after the current turn"));
        }
        GoalAction::Stop => {
            if state.goal_run {
                state.goal_run = false;
                state.pending_prompt = None;
                println!("{}", out.dim("goal run stopped"));
            } else {
                println!("{}", out.dim("goal run is not active"));
            }
        }
        GoalAction::Set(text) => {
            let text = text.trim();
            if text.is_empty() {
                println!("{}", out.dim("goal is empty; nothing changed"));
                return Ok(());
            }
            let text = text.chars().take(4_000).collect::<String>();
            session.append(&Event::GoalSet {
                text: text.clone(),
                ts: now_ms(),
            })?;
            state.goal = Some(text.clone());
            println!(
                "{} {}",
                out.accent("goal"),
                out.paint(Tone::Bright, text.clone())
            );
            if state.plan_mode {
                println!(
                    "{}",
                    out.dim("  plan mode is read-only · /plan off then /goal run")
                );
                return Ok(());
            }
            state.goal_run = true;
            state.goal_turns = 0;
            state.pending_prompt = Some(goal::start_prompt(&text));
            println!(
                "{} {}",
                out.accent("goal run"),
                out.paint(
                    Tone::Bright,
                    format!("starting · max {MAX_GOAL_TURNS} turns · /goal stop")
                )
            );
        }
        GoalAction::Clear => {
            state.goal_run = false;
            state.pending_prompt = None;
            if state.goal.take().is_some() {
                session.append(&Event::GoalCleared { ts: now_ms() })?;
                println!("{} goal cleared", out.paint(Tone::Success, theme::OK));
            } else {
                println!("{}", out.dim("no active goal"));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn goal_after_turn(
    provider: &dyn Provider,
    session: &mut SessionLog,
    state: &mut SessionUiState,
    outcome: &RunOutcome,
    model: &str,
    out: Theme,
    err: Theme,
) -> Result<()> {
    let Some(goal) = state.goal.clone() else {
        state.goal_run = false;
        return Ok(());
    };
    let verdict = goal::evaluate(provider, model, &goal, &outcome.final_text).await?;
    session.append(&Event::GoalVerdict {
        status: verdict.status.as_str().to_string(),
        reason: verdict.reason.clone(),
        ts: now_ms(),
    })?;

    match verdict.status {
        GoalStatus::Met => {
            state.goal_run = false;
            state.goal_turns = 0;
            state.pending_prompt = None;
            state.goal = None;
            session.append(&Event::GoalCleared { ts: now_ms() })?;
            println!(
                "{} {}",
                out.paint(Tone::Success, theme::OK),
                out.paint(
                    Tone::Bright,
                    format!("goal met · {}", clean_text(&verdict.reason))
                )
            );
        }
        GoalStatus::Impossible => {
            state.goal_run = false;
            state.pending_prompt = None;
            eprintln!(
                "{} {}",
                err.paint(Tone::Warning, theme::WARN),
                err.paint(
                    Tone::Warning,
                    format!("goal impossible · {}", clean_text(&verdict.reason))
                )
            );
        }
        GoalStatus::NotMet => {
            state.goal_turns += 1;
            if state.goal_turns >= MAX_GOAL_TURNS {
                state.goal_run = false;
                state.pending_prompt = None;
                eprintln!(
                    "{} {}",
                    err.paint(Tone::Warning, theme::WARN),
                    err.dim(format!(
                        "goal budget exhausted ({MAX_GOAL_TURNS} turns) · {}",
                        clean_text(&verdict.reason)
                    ))
                );
            } else {
                state.pending_prompt = Some(goal::continue_prompt(&goal));
                eprintln!(
                    "{} {}",
                    out.accent("↻ goal not met"),
                    out.dim(format!(
                        "turn {}/{} · {}",
                        state.goal_turns,
                        MAX_GOAL_TURNS,
                        clean_text(&verdict.reason)
                    ))
                );
            }
        }
    }
    Ok(())
}

fn reset_terminal_modes() {
    let mut out = std::io::stdout();
    // Do not send LeaveAlternateScreen here: the TUI never enters the
    // alternate screen at startup. Ghostty (and some terminals) restore a
    // saved cursor position on 1049l, which makes the banner and prompt paint
    // over earlier scrollback instead of continuing at the bottom.
    let _ = execute!(
        out,
        DisableMouseCapture,
        DisableBracketedPaste,
        Show,
        crossterm::style::ResetColor
    );
    let _ = std::io::Write::flush(&mut out);
}

fn trusted_skill_names(config: &AgentConfig) -> Vec<String> {
    Workspace::discover(&config.working_dir)
        .ok()
        .and_then(|workspace| skills::Registry::discover(&workspace).ok())
        .map(|registry| {
            registry
                .list()
                .iter()
                .filter(|skill| skill.trusted)
                .map(|skill| skill.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

fn editor_footer(state: &SessionUiState, config: &AgentConfig) -> String {
    let provider = effective_provider(state, config);
    let model = effective_model(state, config);
    let permission = state
        .permissions
        .map(PermissionProfile::as_str)
        .unwrap_or("startup");
    let cwd = config
        .working_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(".");
    let mut footer = format!(
        "{cwd} · {provider}/{model} · permissions {permission} · context {}",
        config.context.budget_tokens
    );
    if state.plan_mode {
        footer.push_str(" · plan on");
    }
    if let Some(goal) = &state.goal {
        let goal = goal.chars().take(40).collect::<String>();
        footer.push_str(&format!(" · goal {goal}"));
    }
    if let Some(runtime) = &config.computer {
        let runtime = runtime.lock().unwrap();
        let state = if runtime.is_armed() { "✓" } else { "paused" };
        footer.push_str(&format!(
            " · computer {} {state}",
            runtime.policy().as_str()
        ));
    }
    footer
}

async fn run_init_generate(
    provider: &dyn Provider,
    session: &mut SessionLog,
    config: &AgentConfig,
    state: &mut SessionUiState,
    out: Theme,
    err: Theme,
) -> Result<()> {
    let workspace = Workspace::discover(&config.working_dir)?;
    let isolated_dir = workspace.proposals.join("init");
    std::fs::create_dir_all(&isolated_dir)
        .with_context(|| format!("create {}", isolated_dir.display()))?;
    let mut isolated_session = SessionLog::create(&isolated_dir)?;
    isolated_session.append(&Event::SessionStart {
        id: isolated_session.id().to_string(),
        cwd: config.working_dir.display().to_string(),
        model: config.model.clone(),
        provider: config.provider.clone(),
        ts: now_ms(),
    })?;
    isolated_session.append(&Event::Policy {
        sandbox: "read-only".into(),
        approval: "never".into(),
        network: "off".into(),
        backend: "none".into(),
        ts: now_ms(),
    })?;
    let scan = crate::init::scan(&config.working_dir);
    let prompt = crate::init::prompt(&config.working_dir, &scan);
    let mut isolated = AgentConfig::new(config.model.clone(), config.working_dir.clone());
    isolated.provider = config.provider.clone();
    isolated.max_steps = 8;
    isolated.sandbox = SandboxPolicy::new(SandboxMode::ReadOnly, NetworkMode::Off);
    isolated.approval = ApprovalMode::Never;
    isolated.allow_unsandboxed = false;
    isolated.allowed_tools = Some(vec!["read".into(), "grep".into(), "glob".into()]);
    isolated.context.budget_tokens = 32_000;
    isolated.context_budget_auto = false;
    isolated.plan_mode = true;
    isolated.system_prompt =
        "You generate a project-specific AGENTS.md. Inspect only; never modify files.".into();
    println!(
        "{} {}",
        out.accent("init"),
        out.dim(format!(
            "isolated read-only session {}",
            isolated_session.path().display()
        ))
    );
    let mut approver = |_request: &ApprovalRequest| false;
    let mut observer = |event: AgentEvent| {
        if let AgentEvent::Warning(warning) = event {
            eprintln!(
                "{} {}",
                err.paint(Tone::Warning, theme::WARN),
                err.dim(warning)
            );
        }
    };
    let run = tokio::time::timeout(
        Duration::from_secs(120),
        niucode_agent::run(
            provider,
            &mut isolated_session,
            &isolated,
            &prompt,
            &mut approver,
            &mut observer,
        ),
    )
    .await;
    let _ = isolated_session.append(&Event::SessionEnd { ts: now_ms() });
    let generation = match run {
        Ok(Ok(outcome)) => Some(outcome.final_text),
        Ok(Err(error)) => {
            eprintln!(
                "{} {}",
                err.paint(Tone::Warning, theme::WARN),
                err.dim(format!("init model failed: {error:#}; using scan fallback"))
            );
            None
        }
        Err(_) => {
            eprintln!(
                "{} {}",
                err.paint(Tone::Warning, theme::WARN),
                err.dim("init timed out after 120s; using scan fallback")
            );
            None
        }
    };
    let proposal = generation
        .as_deref()
        .and_then(crate::init::normalize)
        .unwrap_or_else(|| {
            if generation.is_some() {
                eprintln!(
                    "{} {}",
                    err.paint(Tone::Warning, theme::WARN),
                    err.dim("model output missed required sections; using scan fallback")
                );
            }
            crate::init::fallback(&config.working_dir, &scan)
        });
    let old = crate::init::existing(&config.working_dir);
    println!(
        "{} {}",
        out.accent("proposal"),
        out.dim(format!(
            "{} lines · {} bytes · /init apply to write",
            proposal.lines().count(),
            proposal.len()
        ))
    );
    if old.trim().is_empty() {
        println!("{}", out.dim("new file: AGENTS.md"));
        for line in proposal.lines() {
            println!("{}", out.paint(Tone::Bright, format!("+ {line}")));
        }
    } else {
        let diff = TextDiff::from_lines(&old, &proposal);
        for line in diff
            .unified_diff()
            .header("a/AGENTS.md", "b/AGENTS.md")
            .to_string()
            .lines()
        {
            let tone = if line.starts_with('+') {
                Tone::Success
            } else if line.starts_with('-') {
                Tone::Danger
            } else {
                Tone::Muted
            };
            println!("{}", out.paint(tone, line));
        }
    }
    if let Some(warning) = crate::init::size_warning(&proposal) {
        eprintln!(
            "{} {}",
            err.paint(Tone::Warning, theme::WARN),
            err.dim(warning)
        );
    }
    state.init_proposal = Some(proposal.clone());
    session.append(&Event::InitProposal {
        content: proposal,
        path: "AGENTS.md".into(),
        ts: now_ms(),
    })?;
    Ok(())
}

async fn run_init_apply(
    _session: &mut SessionLog,
    config: &AgentConfig,
    state: &mut SessionUiState,
    out: Theme,
    err: Theme,
) -> Result<()> {
    if state.plan_mode {
        bail!("plan mode is read-only; use /plan off before /init apply");
    }
    if config.sandbox.mode == SandboxMode::ReadOnly {
        bail!("sandbox is read-only; cannot apply /init proposal");
    }
    let Some(proposal) = state.init_proposal.clone() else {
        bail!("no /init proposal in this session; run /init first");
    };
    let path = config.working_dir.join("AGENTS.md");
    let existing = crate::init::existing(&config.working_dir);
    let backup = if !existing.trim().is_empty() {
        let workspace = Workspace::discover(&config.working_dir)?;
        std::fs::create_dir_all(&workspace.backups)?;
        let backup = workspace.backups.join(format!("AGENTS-{}.md", now_ms()));
        std::fs::copy(&path, &backup).with_context(|| format!("backup {}", path.display()))?;
        Some(backup)
    } else {
        None
    };
    let toolbox = Toolbox::with_policy(&config.working_dir, config.sandbox)?;
    if path.is_file() {
        let _ = toolbox
            .execute("read", &serde_json::json!({"path": "AGENTS.md"}))
            .await;
    }
    let result = toolbox
        .execute(
            "write",
            &serde_json::json!({"path": "AGENTS.md", "content": proposal}),
        )
        .await;
    if !result.ok {
        bail!("apply AGENTS.md failed: {}", result.output);
    }
    println!(
        "{} {}",
        out.paint(Tone::Success, theme::OK),
        out.dim(match &backup {
            Some(backup) => format!("wrote {} · backup {}", path.display(), backup.display()),
            None => format!("wrote {}", path.display()),
        })
    );
    let _ = err;
    reload_instructions(config, out, err)?;
    Ok(())
}

fn reload_instructions(config: &AgentConfig, out: Theme, err: Theme) -> Result<()> {
    let loaded = niucode_agent::load_instructions(&config.working_dir);
    for warning in &loaded.warnings {
        eprintln!(
            "{} {}",
            err.paint(Tone::Warning, theme::WARN),
            err.dim(warning)
        );
    }
    if loaded.sources.is_empty() {
        println!(
            "{} {}",
            out.accent("reload"),
            out.dim("no AGENTS.md found · built-in prompt only")
        );
    } else {
        println!(
            "{} {}",
            out.accent("reload"),
            out.paint(
                Tone::Bright,
                format!(
                    "{} instruction source(s) · digest {}",
                    loaded.sources.len(),
                    loaded.digest
                )
            )
        );
        for source in &loaded.sources {
            println!("{}", out.dim(format!("  {source}")));
        }
    }
    println!(
        "{}",
        out.dim("  effective on the next turn · extensions stay session-scoped")
    );
    Ok(())
}

fn format_age(modified: u64) -> String {
    let now = now_ms() / 1000;
    let age = now.saturating_sub(modified);
    if age < 60 {
        "just now".to_string()
    } else if age < 3_600 {
        format!("{}m ago", age / 60)
    } else if age < 86_400 {
        format!("{}h ago", age / 3_600)
    } else {
        format!("{}d ago", age / 86_400)
    }
}

fn resolve_session_path(directory: &Path, arg: &str) -> Result<PathBuf> {
    let raw = PathBuf::from(arg);
    let mut candidates = Vec::new();
    if raw.is_absolute() {
        candidates.push(raw);
    } else {
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join(&raw));
        }
        candidates.push(directory.join(&raw));
    }
    for mut candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
        if candidate.extension().is_none() {
            candidate.set_extension("jsonl");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    bail!("session not found: {arg} · use /resume to list recent sessions")
}

#[derive(Debug, Clone)]
struct RecentSession {
    path: PathBuf,
    modified: u64,
    title: String,
    provider: String,
    model: String,
}

fn session_summary(path: &Path) -> (String, String, String) {
    let mut provider = String::new();
    let mut model = String::new();
    let mut user = None;
    let mut goal = None;
    if let Ok(file) = File::open(path) {
        for line in BufReader::new(file).lines().take(400).flatten() {
            let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            match event.get("type").and_then(|value| value.as_str()) {
                Some("session_start") => {
                    provider = event
                        .get("provider")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .to_string();
                    model = event
                        .get("model")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .to_string();
                }
                Some("user_message") if user.is_none() => {
                    user = event
                        .get("text")
                        .and_then(|value| value.as_str())
                        .map(str::to_string);
                }
                Some("goal_set") if goal.is_none() => {
                    goal = event
                        .get("text")
                        .and_then(|value| value.as_str())
                        .map(str::to_string);
                }
                _ => {}
            }
            if user.is_some() {
                break;
            }
        }
    }
    let title = user
        .or(goal)
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("session")
                .to_string()
        })
        .replace(['\n', '\r', '\t'], " ");
    let title = title.trim().chars().take(72).collect::<String>();
    (title, provider, model)
}

fn sessions_in(directory: &Path) -> Vec<RecentSession> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                return None;
            }
            let modified = entry
                .metadata()
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs())
                .unwrap_or(0);
            let (title, provider, model) = session_summary(&path);
            Some(RecentSession {
                path,
                modified,
                title,
                provider,
                model,
            })
        })
        .collect()
}

fn recent_sessions(directory: &Path, all: bool) -> Vec<RecentSession> {
    let mut sessions = if all {
        let projects = directory
            .parent()
            .and_then(Path::parent)
            .filter(|path| path.file_name().is_some_and(|name| name == "projects"));
        match projects {
            Some(projects) => std::fs::read_dir(projects)
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .filter_map(|entry| {
                            let directory = entry.path().join("sessions");
                            directory.is_dir().then(|| sessions_in(&directory))
                        })
                        .flatten()
                        .collect()
                })
                .unwrap_or_default(),
            None => sessions_in(directory),
        }
    } else {
        sessions_in(directory)
    };
    sessions.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| right.path.cmp(&left.path))
    });
    sessions.truncate(10);
    sessions
}

#[allow(clippy::too_many_arguments)]
fn print_resume_list(sessions: &[RecentSession], list_all: bool, out: Theme) {
    println!(
        "{} {}",
        out.accent("resume"),
        out.dim(if list_all {
            "all projects · /resume <path|id>"
        } else {
            "current project · /resume --all for all projects"
        })
    );
    for session in sessions {
        println!(
            "{} {}",
            out.paint(Tone::Bright, format!("  {}", session.title)),
            out.dim(format!(
                "  {}/{} · {}",
                if session.provider.is_empty() {
                    "?"
                } else {
                    &session.provider
                },
                if session.model.is_empty() {
                    "?"
                } else {
                    &session.model
                },
                format_age(session.modified)
            ))
        );
    }
}

fn pick_session(sessions: &[RecentSession], list_all: bool, out: Theme) -> Result<Option<PathBuf>> {
    struct Guard {
        active: bool,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            if self.active {
                let _ = execute!(std::io::stdout(), LeaveAlternateScreen, Show);
                let _ = terminal::disable_raw_mode();
            }
        }
    }
    terminal::enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen, Show)?;
    let guard = Guard { active: true };
    let mut selected = 0_usize;
    let draw = |selected: usize| -> std::io::Result<()> {
        let mut screen = String::from("\x1b[2J\x1b[H");
        screen.push_str(&format!(
            "{} {}\r\n",
            out.accent("resume"),
            out.dim(if list_all {
                "all projects · ↑↓/j/k · enter resume · esc cancel"
            } else {
                "current project · ↑↓/j/k · enter resume · esc cancel · --all for all projects"
            })
        ));
        for (index, session) in sessions.iter().enumerate() {
            let marker = if index == selected { "▸" } else { " " };
            let title = session.title.chars().take(60).collect::<String>();
            let line = format!(
                "{marker} {title}  ·  {}/{} · {}",
                if session.provider.is_empty() {
                    "?"
                } else {
                    &session.provider
                },
                if session.model.is_empty() {
                    "?"
                } else {
                    &session.model
                },
                format_age(session.modified)
            );
            if index == selected {
                screen.push_str(&out.paint(Tone::Bright, line));
            } else {
                screen.push_str(&line);
            }
            screen.push_str("\r\n");
        }
        screen.push_str("\r\n  enter resume · esc cancel · ↑↓ select\r\n");
        print!("{screen}");
        std::io::stdout().flush()?;
        Ok(())
    };
    draw(selected)?;
    loop {
        let event = event::read()?;
        let TerminalEvent::Key(key) = event else {
            continue;
        };
        if key.kind != crossterm::event::KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
                draw(selected)?;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(sessions.len().saturating_sub(1));
                draw(selected)?;
            }
            KeyCode::Enter => {
                let path = sessions[selected].path.clone();
                drop(guard);
                return Ok(Some(path));
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                drop(guard);
                return Ok(None);
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                drop(guard);
                return Ok(None);
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resume_session(
    session: &mut SessionLog,
    config: &AgentConfig,
    history: &mut Vec<ChatMessage>,
    allow_session: &mut bool,
    last_tool: &mut Option<ToolEcho>,
    state: &mut SessionUiState,
    arg: Option<&str>,
    out: Theme,
    err: Theme,
) -> Result<()> {
    let directory = session
        .path()
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let list_all = matches!(arg, Some("--all") | Some("all"));
    let path = if let Some(arg) = arg.filter(|value| !matches!(*value, "--all" | "all")) {
        resolve_session_path(&directory, arg)?
    } else {
        let sessions = recent_sessions(&directory, list_all);
        if sessions.is_empty() {
            println!("{}", out.dim("no sessions found in this directory"));
            return Ok(());
        }
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            print_resume_list(&sessions, list_all, out);
            return Ok(());
        }
        let Some(path) = pick_session(&sessions, list_all, out)? else {
            return Ok(());
        };
        path
    };
    if path == session.path() {
        println!("{}", out.dim("already on this session"));
        return Ok(());
    }
    let loaded = load(&path)?;
    for warning in &loaded.warnings {
        eprintln!(
            "{} {}",
            err.paint(Tone::Warning, theme::WARN),
            err.dim(warning)
        );
    }
    let new_history = project_history(&loaded.events)?;
    let mut new_state = SessionUiState::from_events(&loaded.events);
    new_state.clear_history = false;
    let new_session = SessionLog::open(&path)?;
    *session = new_session;
    *history = new_history;
    *state = new_state;
    *allow_session = false;
    *last_tool = None;
    reset_screen(err);
    println!(
        "{} {}",
        out.paint(Tone::Success, theme::OK),
        out.dim(format!(
            "resumed · {} · {} events",
            session.path().display(),
            loaded.events.len()
        ))
    );
    if let Some(Event::SessionStart { cwd, model, .. }) = loaded
        .events
        .iter()
        .find(|event| matches!(event, Event::SessionStart { .. }))
    {
        if cwd != &config.working_dir.display().to_string() {
            eprintln!(
                "{}",
                err.paint(
                    Tone::Warning,
                    format!(
                        "warning: session cwd {cwd} differs from current {}",
                        config.working_dir.display()
                    )
                )
            );
        }
        if model != &config.model {
            eprintln!(
                "{}",
                err.paint(
                    Tone::Warning,
                    format!(
                        "warning: session model {model} differs from current {}",
                        config.model
                    )
                )
            );
        }
    }
    Ok(())
}

fn notify_session(hooks: Option<&SharedHooks>, path: &Path) {
    if let Some(hooks) = hooks {
        hooks.borrow_mut().on_session_start(path);
    }
}

fn new_session(
    session: &SessionLog,
    config: &AgentConfig,
    custom: Option<PathBuf>,
) -> Result<SessionLog> {
    let directory = session
        .path()
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let new_session = match custom {
        Some(path) => SessionLog::create_at(path)?,
        None => SessionLog::create(directory)?,
    };
    new_session.append(&Event::SessionStart {
        id: new_session.id().to_string(),
        cwd: config.working_dir.display().to_string(),
        model: config.model.clone(),
        provider: config.provider.clone(),
        ts: now_ms(),
    })?;
    new_session.append(&Event::Policy {
        sandbox: config.sandbox.mode.as_str().into(),
        approval: config.approval.as_str().into(),
        network: config.sandbox.network.as_str().into(),
        backend: config.sandbox.backend().into(),
        ts: now_ms(),
    })?;
    Ok(new_session)
}

fn fork_session(session: &SessionLog, at: Option<&str>) -> Result<session_ops::ForkReport> {
    let source = session.path().to_path_buf();
    let at = match at {
        Some(at) => at.to_string(),
        None => {
            let loaded = load(&source)?;
            if loaded.events.is_empty() {
                bail!("current session has no events to fork");
            }
            EventId::from_index(loaded.events.len() - 1).to_string()
        }
    };
    let directory = source
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    session_ops::fork(&source, &at, directory)
}

fn print_status(session: &SessionLog, config: &AgentConfig, out: Theme) -> Result<()> {
    let loaded = load(session.path())?;
    let mut usage = Usage::default();
    let mut turns = 0_usize;
    let mut compactions = 0_usize;
    let mut instructions = 0_usize;
    let mut subagents = 0_usize;
    let mut extensions = 0_usize;
    for event in &loaded.events {
        match event {
            Event::UserMessage { .. } => turns += 1,
            Event::Compaction { .. } => compactions += 1,
            Event::Instructions { .. } => instructions += 1,
            Event::SubagentSpawn { .. } => subagents += 1,
            Event::ExtensionMount { .. } => extensions += 1,
            Event::TurnEnd { usage: turn, .. } => {
                usage.input_tokens += turn.input_tokens;
                usage.output_tokens += turn.output_tokens;
                usage.cached_tokens += turn.cached_tokens;
                usage.reasoning_tokens += turn.reasoning_tokens;
            }
            _ => {}
        }
    }

    let ui = SessionUiState::from_events(&loaded.events);
    let mut effective = config.clone();
    if let Some(profile) = ui.permissions {
        profile.apply(&mut effective);
    }
    if effective.context_budget_auto {
        effective.context.budget_tokens = niucode_provider::derived_context_budget(
            ui.provider.as_deref().unwrap_or(&config.provider),
            ui.model.as_deref().unwrap_or(&config.model),
        );
    }
    println!("{}", out.accent("status"));
    println!(
        "  model      {}/{}",
        ui.provider.as_deref().unwrap_or(&config.provider),
        ui.model.as_deref().unwrap_or(&config.model)
    );
    println!("  cwd        {}", config.working_dir.display());
    println!(
        "  sandbox    {} · network {} · approval {} · backend {} · permissions {}",
        effective.sandbox.mode.as_str(),
        effective.sandbox.network.as_str(),
        effective.approval.as_str(),
        effective.sandbox.backend(),
        ui.permissions
            .map(PermissionProfile::as_str)
            .unwrap_or("startup")
    );
    println!(
        "  context    budget {} · trigger {}% · keep {} turns",
        effective.context.budget_tokens,
        config.context.compact_at_percent,
        config.context.keep_recent_turns
    );
    println!("  session    {}", session.path().display());
    if let Ok(workspace) = Workspace::discover(&config.working_dir) {
        println!(
            "  workspace  {} ({})",
            workspace.project_dir.display(),
            workspace.project_key
        );
    }
    println!(
        "  plan       {}",
        if ui.plan_mode {
            "on · read-only"
        } else {
            "off"
        }
    );
    println!("  goal       {}", ui.goal.as_deref().unwrap_or("none"));
    if std::env::var_os("TMUX").is_some() {
        println!(
            "  mouse      menu-scoped · events {} · tmux detected · run `tmux set -g mouse on` if 0",
            mouse_events_seen()
        );
    } else {
        println!(
            "  mouse      menu-scoped · events {} · absolute + relative hover",
            mouse_events_seen()
        );
    }
    println!(
        "  events     {} decoded · {} warnings",
        loaded.events.len(),
        loaded.warnings.len()
    );
    println!(
        "  turns      {turns} · compactions {compactions} · instructions {instructions} · subagents {subagents} · extensions {extensions}"
    );
    if usage.input_tokens > 0 {
        println!(
            "  cache      {}% · in {} · out {} · cached {}",
            usage.cache_hit_percent(),
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_tokens
        );
    }
    Ok(())
}

fn print_session(session: &SessionLog, out: Theme) -> Result<()> {
    let loaded = load(session.path())?;
    let turns = loaded
        .events
        .iter()
        .filter(|event| matches!(event, Event::UserMessage { .. }))
        .count();
    println!("{}", out.accent("session"));
    println!("  path       {}", session.path().display());
    if std::env::var_os("TMUX").is_some() {
        println!(
            "  mouse      menu-scoped · events {} · tmux detected · run `tmux set -g mouse on` if 0",
            mouse_events_seen()
        );
    } else {
        println!(
            "  mouse      menu-scoped · events {} · absolute + relative hover",
            mouse_events_seen()
        );
    }
    println!(
        "  events     {} decoded · {} warnings",
        loaded.events.len(),
        loaded.warnings.len()
    );
    println!("  turns      {turns}");
    Ok(())
}

fn reset_screen(theme: Theme) {
    if std::io::stdout().is_terminal() {
        let mut out = std::io::stdout();
        let _ = execute!(out, Clear(ClearType::All), MoveTo(0, 0));
    }
    if std::io::stderr().is_terminal() {
        eprintln!(
            "{}",
            theme.accent(format!(
                "NIUCODE // v{} · fresh session",
                env!("CARGO_PKG_VERSION")
            ))
        );
        eprintln!(
            "{}",
            theme.dim("native scrollback · :help commands · Ctrl-C cancels the turn")
        );
    }
}

fn prompt_approval(
    request: &ApprovalRequest,
    out: Theme,
    err: Theme,
    keys: ApprovalKeys,
) -> ApprovalChoice {
    if let Some(preview) = &request.preview {
        if preview.is_diff() && std::io::stdout().is_terminal() && std::io::stderr().is_terminal() {
            match review_diff(request, preview, out, keys) {
                Ok(Some(choice)) => return choice,
                Ok(None) => {}
                Err(error) => eprintln!(
                    "{} {}",
                    err.paint(Tone::Warning, theme::WARN),
                    err.dim(format!("diff pager unavailable: {error}"))
                ),
            }
        }
        println!();
        eprintln!(
            "{} {}",
            err.paint(Tone::Magenta, "approval"),
            err.paint(Tone::Bright, clean_text(&request.summary))
        );
        eprint!("{}", render_plain(preview, plain_width()));
    } else {
        println!();
        eprintln!(
            "{} {}",
            err.paint(Tone::Magenta, "approval"),
            err.paint(Tone::Bright, clean_text(&request.summary))
        );
    }

    prompt_keys(err, keys)
}

fn prompt_keys(err: Theme, keys: ApprovalKeys) -> ApprovalChoice {
    if terminal::enable_raw_mode().is_ok() {
        eprint!(
            "{}  {}  {}  ",
            err.paint(Tone::Accent, format!("[{}]llow", keys.allow)),
            err.paint(Tone::Danger, format!("[{}]eny", keys.deny)),
            err.paint(Tone::Magenta, format!("[{}]ession", keys.session))
        );
        let _ = std::io::stderr().flush();
        let choice = loop {
            match event::read() {
                Ok(TerminalEvent::Key(KeyEvent {
                    code, modifiers, ..
                })) => {
                    if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
                        break ApprovalChoice::Deny;
                    }
                    if let Some(choice) = choice_for_key(code, keys) {
                        break choice;
                    }
                    if code == KeyCode::Esc {
                        break ApprovalChoice::Deny;
                    }
                }
                Ok(_) => {}
                Err(_) => break ApprovalChoice::Deny,
            }
        };
        let _ = terminal::disable_raw_mode();
        eprintln!();
        return choice;
    }

    eprint!(
        "{} {} {}: ",
        err.paint(Tone::Accent, format!("[{}] allow", keys.allow)),
        err.paint(Tone::Danger, format!("[{}] deny", keys.deny)),
        err.paint(Tone::Magenta, format!("[{}] session", keys.session))
    );
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return ApprovalChoice::Deny;
    }
    let input = line.trim().to_lowercase();
    let first = input.chars().next();
    if input == "allow" || input == "yes" || first == Some(keys.allow.to_ascii_lowercase()) {
        ApprovalChoice::Allow
    } else if input == "session" || first == Some(keys.session.to_ascii_lowercase()) {
        ApprovalChoice::Session
    } else {
        ApprovalChoice::Deny
    }
}

fn plain_width() -> usize {
    terminal::size()
        .map(|(columns, _)| columns.saturating_sub(1) as usize)
        .unwrap_or(100)
        .max(40)
}

fn flush_usage(theme: Theme, pending: &mut Option<Usage>, turn_no: usize) {
    let Some(usage) = pending.take() else {
        return;
    };
    eprintln!(
        "{}",
        theme.dim(format!(
            "{} turn {} · in {} · out {} · cache {}%",
            theme::TURN,
            turn_no,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_hit_percent()
        ))
    );
}

fn render_tool_result(
    theme: Theme,
    name: &str,
    ok: bool,
    summary: &str,
    output: &str,
    last: &mut Option<ToolEcho>,
) {
    let (symbol, tone) = if ok {
        (theme::OK, Tone::Success)
    } else {
        (theme::ERR, Tone::Danger)
    };
    let summary = clean_text(summary);
    eprintln!(
        "{} {} {}",
        theme.paint(tone, symbol),
        theme.paint(Tone::Bright, name),
        theme.dim(&summary)
    );

    let body = output_body(summary.as_str(), output);
    if !body.is_empty() {
        if should_fold(&body, output.len()) {
            if !ok {
                for line in folded_error_tail(&body) {
                    eprintln!(
                        "{} {}",
                        theme.dim(theme::BRANCH),
                        theme.paint(Tone::Danger, clean_text(line))
                    );
                }
            }
            eprintln!(
                "{} {}",
                theme.dim(theme::BRANCH),
                theme.dim(format!(
                    "{} lines · {} · :last to expand",
                    body.len(),
                    human_bytes(output.len())
                ))
            );
        } else {
            for line in &body {
                eprintln!("{} {}", theme.dim(theme::BRANCH), clean_text(line));
            }
        }
    }
    *last = Some(ToolEcho {
        name: name.to_string(),
        summary,
        output: output.to_string(),
    });
}

fn output_body<'a>(summary: &str, output: &'a str) -> Vec<&'a str> {
    let body = if output == summary {
        ""
    } else {
        match output.split_once('\n') {
            Some((first, rest)) if first == summary => rest,
            _ => output,
        }
    };
    body.lines().collect()
}

fn should_fold(lines: &[&str], bytes: usize) -> bool {
    lines.len() > TOOL_OUTPUT_FOLD_LINES || bytes > TOOL_OUTPUT_FOLD_BYTES
}

fn folded_error_tail<'a>(lines: &[&'a str]) -> Vec<&'a str> {
    let mut tail: Vec<&'a str> = lines
        .iter()
        .copied()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && trimmed != "stdout:" && trimmed != "stderr:"
        })
        .rev()
        .take(3)
        .collect();
    tail.reverse();
    tail
}

fn human_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn print_last(theme: Theme, last: Option<&ToolEcho>) {
    let Some(echo) = last else {
        eprintln!("{}", theme.dim("no tool output yet"));
        return;
    };
    println!();
    println!(
        "{}",
        theme.paint(
            Tone::Magenta,
            format!(
                "╭─ last tool · {} · {}",
                echo.name,
                clean_text(&echo.summary)
            )
        )
    );
    if echo.output.is_empty() {
        println!("{} {}", theme.dim(theme::BRANCH), theme.dim("(no output)"));
    } else {
        for line in echo.output.lines() {
            println!("{} {}", theme.dim(theme::BRANCH), clean_text(line));
        }
    }
    println!("{}", theme.dim("╰─ end"));
}

fn run_turn_prompts(hooks: &Option<SharedHooks>, outcome: &RunOutcome) {
    let Some(hooks) = hooks else {
        return;
    };
    let prompts = hooks.borrow_mut().after_turn(outcome);
    for prompt in prompts {
        println!("\n{}", prompt.question);
        for (index, option) in prompt.options.iter().enumerate() {
            println!("  {}. {option}", index + 1);
        }
        if !std::io::stdin().is_terminal() {
            continue;
        }
        loop {
            print!("> ");
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_err() {
                break;
            }
            let choice = line.trim().parse::<usize>().ok();
            match choice {
                Some(choice) if (1..=prompt.options.len()).contains(&choice) => {
                    hooks.borrow_mut().on_answer(&prompt.id, choice - 1);
                    break;
                }
                _ => eprintln!("enter 1..{}", prompt.options.len()),
            }
        }
    }
}

fn print_help(theme: Theme, commands: &CommandRegistry) {
    println!("{}", theme.accent("commands"));
    for spec in commands.all() {
        println!(
            "  {} {}",
            theme.paint(Tone::Bright, format!("{:<9}", spec.name)),
            spec.description
        );
    }
    println!();
    println!(
        "{}",
        theme.dim("  aliases: exit · quit · :help · :last · :q")
    );
    println!(
        "{}",
        theme.dim("  other text (including plain prompts) goes to the model")
    );
}

fn compact_json(value: &serde_json::Value) -> String {
    let rendered = serde_json::to_string(value).unwrap_or_default();
    clean_text(&rendered).chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_the_summary_line_from_tool_output() {
        assert_eq!(
            output_body("exit: 1", "exit: 1\nstdout:\nhello\n"),
            vec!["stdout:", "hello"]
        );
        assert!(output_body("exit: 1", "exit: 1").is_empty());
        assert_eq!(output_body("summary", "other\n"), vec!["other"]);
    }

    #[test]
    fn folds_only_past_the_threshold() {
        let short = vec!["line"; TOOL_OUTPUT_FOLD_LINES];
        assert!(!should_fold(&short, 10));
        let long = vec!["line"; TOOL_OUTPUT_FOLD_LINES + 1];
        assert!(should_fold(&long, 10));
        assert!(should_fold(&["line"], TOOL_OUTPUT_FOLD_BYTES + 1));
        assert_eq!(
            folded_error_tail(&["stdout:", "step 1", "stderr:", "err 1", "err 2", "err 3", ""]),
            vec!["err 1", "err 2", "err 3"]
        );
    }

    #[test]
    fn human_bytes_is_stable() {
        assert_eq!(human_bytes(12), "12 B");
        assert_eq!(human_bytes(2_048), "2.0 KB");
        assert_eq!(human_bytes(2 * 1024 * 1024), "2.0 MB");
    }

    #[test]
    fn session_ui_state_uses_the_latest_plan_and_goal_events() {
        let state = SessionUiState::from_events(&[
            Event::PlanMode {
                active: true,
                ts: 1,
            },
            Event::GoalSet {
                text: "all tests pass".into(),
                ts: 2,
            },
            Event::PlanMode {
                active: false,
                ts: 3,
            },
        ]);
        assert!(!state.plan_mode);
        assert_eq!(state.goal.as_deref(), Some("all tests pass"));

        let state = SessionUiState::from_events(&[
            Event::PlanMode {
                active: true,
                ts: 1,
            },
            Event::GoalSet {
                text: "all tests pass".into(),
                ts: 2,
            },
            Event::GoalCleared { ts: 3 },
        ]);
        assert!(state.plan_mode);
        assert!(state.goal.is_none());
    }

    #[test]
    fn session_summary_uses_the_first_user_message() {
        let directory =
            std::env::temp_dir().join(format!("niucode-session-summary-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("temp dir");
        let path = directory.join("s.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"type\":\"session_start\",\"id\":\"s\",\"cwd\":\"/tmp\",\"model\":\"gpt-4o\",\"provider\":\"openai\",\"ts\":1}\n",
                "{\"type\":\"user_message\",\"text\":\"fix the parser\",\"ts\":2}\n"
            ),
        )
        .expect("session");
        let (title, provider, model) = session_summary(&path);
        assert_eq!(title, "fix the parser");
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");
        std::fs::remove_dir_all(directory).expect("cleanup");
    }

    struct FooterComputer;

    impl niucode_computer::ComputerBackend for FooterComputer {
        fn observe(
            &mut self,
            _window: Option<niucode_computer::WindowId>,
        ) -> Result<niucode_computer::Capture, Box<dyn std::error::Error + Send + Sync>> {
            Err("unused".into())
        }

        fn perform(
            &mut self,
            _action: &niucode_computer::ComputerAction,
            _observation: &niucode_computer::Observation,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Ok(())
        }
    }

    #[test]
    fn footer_shows_computer_policy_and_stop_state() {
        let mut config = AgentConfig::new("test", ".");
        let runtime = Arc::new(std::sync::Mutex::new(niucode_agent::ComputerRuntime::new(
            Box::new(FooterComputer),
            niucode_computer::ComputerPolicy::Control,
        )));
        runtime.lock().unwrap().arm().expect("arm");
        config.computer = Some(runtime.clone());
        let state = SessionUiState::from_events(&[]);
        let footer = editor_footer(&state, &config);
        assert!(footer.contains("computer control ✓"), "{footer}");
        runtime.lock().unwrap().emergency_stop().expect("stop");
        let footer = editor_footer(&state, &config);
        assert!(footer.contains("computer off paused"), "{footer}");
    }

    #[test]
    fn permission_profiles_map_to_policy() {
        let mut config = AgentConfig::new("test", ".");
        PermissionProfile::Ask.apply(&mut config);
        assert_eq!(config.sandbox.mode, SandboxMode::WorkspaceWrite);
        assert_eq!(config.sandbox.network, NetworkMode::Off);
        assert_eq!(config.approval, ApprovalMode::Ask);

        PermissionProfile::Review.apply(&mut config);
        assert_eq!(config.approval, ApprovalMode::OnFailure);

        PermissionProfile::FullAccess.apply(&mut config);
        assert_eq!(config.sandbox.mode, SandboxMode::FullAccess);
        assert_eq!(config.sandbox.network, NetworkMode::On);
        assert_eq!(config.approval, ApprovalMode::Never);

        PermissionProfile::Ci.apply(&mut config);
        assert_eq!(config.sandbox.mode, SandboxMode::ReadOnly);
        assert_eq!(config.sandbox.network, NetworkMode::Off);
        assert_eq!(config.approval, ApprovalMode::Never);
    }

    #[test]
    fn session_change_notifies_hooks() {
        use crate::TurnHooks;
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::sync::{Arc, Mutex};

        struct Recorder(Arc<Mutex<Option<PathBuf>>>);
        impl TurnHooks for Recorder {
            fn on_session_start(&mut self, path: &Path) {
                *self.0.lock().unwrap() = Some(path.to_path_buf());
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let hooks: SharedHooks = Rc::new(RefCell::new(Box::new(Recorder(seen.clone()))));
        notify_session(Some(&hooks), Path::new("/tmp/thread.jsonl"));
        assert_eq!(
            seen.lock().unwrap().as_deref(),
            Some(Path::new("/tmp/thread.jsonl"))
        );
    }

    #[test]
    fn at_path_only_expands_existing_files() {
        let directory = tempfile::tempdir().unwrap();
        let image = directory.path().join("shot.png");
        std::fs::write(&image, b"\x89PNG\r\n\x1a\nimage").unwrap();
        let (clean, paths) = split_at_paths(&format!("look @{} now", image.display()));
        assert_eq!(clean, "look now");
        assert_eq!(paths, [image]);
        let (clean, paths) = split_at_paths("mail me@example.com @missing");
        assert_eq!(clean, "mail me@example.com @missing");
        assert!(paths.is_empty());
    }

    #[test]
    fn parses_permission_aliases() {
        assert_eq!(
            PermissionProfile::parse(":ask"),
            Some(PermissionProfile::Ask)
        );
        assert_eq!(
            PermissionProfile::parse("full-access"),
            Some(PermissionProfile::FullAccess)
        );
        assert_eq!(
            PermissionProfile::parse("full"),
            Some(PermissionProfile::FullAccess)
        );
        assert_eq!(PermissionProfile::parse("nope"), None);
    }
}
