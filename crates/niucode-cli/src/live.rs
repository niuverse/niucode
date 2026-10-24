//! TUI client for SessionRuntime. This loop alone owns the active terminal.
use crate::{
    details::Details,
    diff::render_plain,
    editor::{Input, LineEditor},
    review::{choice_for_key, ApprovalChoice},
    terminal::{DetailScreen, TerminalSession},
    text_layout::wrap,
    theme::Theme,
    ApprovalKeys, SharedHooks,
};
use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use niucode_agent::{
    run_interactive_with_attachments, runtime_channel, AgentConfig, AgentEvent, ApprovalRequest,
    RunOutcome, RuntimeClient, RuntimeCommand, RuntimeEvent,
};
use niucode_attachments::AttachmentStore;
use niucode_core::{AttachmentRef, SessionLog};
use niucode_provider::{ChatMessage, Provider};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const BUFFER_BYTES: usize = 262_144;

struct View {
    screen: DetailScreen,
    content: ViewContent,
    keys: ApprovalKeys,
}
enum ViewContent {
    History(Details),
    Approval {
        id: String,
        request: ApprovalRequest,
        previous: Option<Box<Details>>,
        lines: Vec<String>,
        offset: usize,
    },
}
enum ViewAction {
    Stay,
    Close,
    Cancel,
    Approve {
        id: String,
        allowed: bool,
        session: bool,
    },
}
impl View {
    fn history(session: &SessionLog, width: usize, keys: ApprovalKeys) -> Result<Self> {
        Ok(Self {
            screen: DetailScreen::enter()?,
            content: ViewContent::History(Details::open(session.path(), width)?),
            keys,
        })
    }
    fn approval(
        id: String,
        request: ApprovalRequest,
        width: usize,
        keys: ApprovalKeys,
    ) -> Result<Self> {
        let lines = Self::approval_lines(&request, width);
        Ok(Self {
            screen: DetailScreen::enter()?,
            content: ViewContent::Approval {
                id,
                request,
                previous: None,
                lines,
                offset: 0,
            },
            keys,
        })
    }
    fn show_approval(&mut self, id: String, request: ApprovalRequest, width: usize) {
        let lines = Self::approval_lines(&request, width);
        let old = std::mem::replace(
            &mut self.content,
            ViewContent::Approval {
                id,
                request,
                lines,
                offset: 0,
                previous: None,
            },
        );
        if let (ViewContent::History(history), ViewContent::Approval { previous, .. }) =
            (old, &mut self.content)
        {
            *previous = Some(Box::new(history));
        }
    }
    fn resume_history(&mut self, width: usize) -> Result<bool> {
        if let ViewContent::Approval { previous, .. } = &mut self.content {
            if let Some(mut history) = previous.take() {
                history.refresh(width)?;
                self.content = ViewContent::History(*history);
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn approval_lines(request: &ApprovalRequest, width: usize) -> Vec<String> {
        let body = request
            .preview
            .as_ref()
            .map(|p| render_plain(p, width))
            .unwrap_or_else(|| request.summary.clone());
        wrap(&body, width)
    }
    fn draw(&mut self) -> Result<()> {
        match &self.content {
            ViewContent::History(d) => self.screen.draw(&d.title, &d.lines, d.offset)?,
            ViewContent::Approval {
                request,
                lines,
                offset,
                ..
            } => self.screen.draw(
                &format!(
                    "{} allow / {} deny / {} session · Ctrl-C cancel · {}",
                    self.keys.allow, self.keys.deny, self.keys.session, request.summary
                ),
                lines,
                *offset,
            )?,
        }
        Ok(())
    }
    fn scroll(&mut self, delta: isize) {
        let (offset, count) = match &mut self.content {
            ViewContent::History(d) => (&mut d.offset, d.lines.len()),
            ViewContent::Approval { offset, lines, .. } => (offset, lines.len()),
        };
        *offset = offset
            .saturating_add_signed(delta)
            .min(count.saturating_sub(1));
    }
    fn handle(&mut self, event: Event, width: usize, height: usize) -> Result<ViewAction> {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    return Ok(ViewAction::Cancel);
                }
                if let ViewContent::Approval { id, .. } = &self.content {
                    // Modified shortcuts must neither approve nor hide a pending request.
                    if !key.modifiers.is_empty() {
                        return Ok(ViewAction::Stay);
                    }
                    let choice = choice_for_key(key.code, self.keys).or_else(|| {
                        matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
                            .then_some(ApprovalChoice::Deny)
                    });
                    if let Some(choice) = choice {
                        let (allowed, session) = match choice {
                            ApprovalChoice::Allow => (true, false),
                            ApprovalChoice::Session => (true, true),
                            ApprovalChoice::Deny => (false, false),
                        };
                        return Ok(ViewAction::Approve {
                            id: id.clone(),
                            allowed,
                            session,
                        });
                    }
                }
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') => return Ok(ViewAction::Close),
                    KeyCode::Up | KeyCode::Char('k') => self.scroll(-1),
                    KeyCode::Down | KeyCode::Char('j') => self.scroll(1),
                    KeyCode::PageUp => self.scroll(-(height.saturating_sub(2).max(1) as isize)),
                    KeyCode::PageDown | KeyCode::Char(' ') => {
                        self.scroll(height.saturating_sub(2).max(1) as isize)
                    }
                    KeyCode::Char('[' | ']') => {
                        if let ViewContent::History(d) = &mut self.content {
                            d.navigate(if key.code == KeyCode::Char('[') {
                                -1
                            } else {
                                1
                            })?;
                        }
                    }
                    _ => {}
                }
            }
            Event::Resize(_, _) => match &mut self.content {
                ViewContent::History(d) => d.refresh(width)?,
                ViewContent::Approval {
                    request,
                    lines,
                    offset,
                    ..
                } => {
                    *lines = Self::approval_lines(request, width);
                    *offset = (*offset).min(lines.len().saturating_sub(1));
                }
            },
            _ => {}
        }
        Ok(ViewAction::Stay)
    }
}

struct Transcript {
    pending: String,
    deferred: VecDeque<String>,
    bytes: usize,
    omitted: bool,
}
impl Transcript {
    fn new() -> Self {
        Self {
            pending: String::new(),
            deferred: VecDeque::new(),
            bytes: 0,
            omitted: false,
        }
    }
    fn print(&mut self, terminal: &mut TerminalSession, viewing: bool, text: String) -> Result<()> {
        if !viewing {
            terminal.print(&text)?;
        } else if self.bytes + text.len() <= BUFFER_BYTES {
            self.bytes += text.len();
            self.deferred.push_back(text);
        } else {
            self.omitted = true;
        }
        Ok(())
    }
    fn delta(&mut self, terminal: &mut TerminalSession, viewing: bool, text: &str) -> Result<()> {
        self.pending.push_str(text);
        while let Some(index) = self.pending.find('\n') {
            let rest = self.pending.split_off(index + 1);
            let line = std::mem::replace(&mut self.pending, rest);
            self.print(terminal, viewing, line.trim_end_matches('\n').into())?;
        }
        if self.pending.len() > 8192 {
            self.flush(terminal, viewing)?;
        }
        Ok(())
    }
    fn flush(&mut self, terminal: &mut TerminalSession, viewing: bool) -> Result<()> {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            self.print(terminal, viewing, line)?;
        }
        if !viewing {
            terminal.flush_table()?;
            terminal.flush_math()?;
        }
        Ok(())
    }
    fn resume(&mut self, terminal: &mut TerminalSession) -> Result<()> {
        terminal.flush_table()?;
        terminal.clear()?;
        while let Some(line) = self.deferred.pop_front() {
            terminal.print(&line)?;
        }
        self.bytes = 0;
        if std::mem::take(&mut self.omitted) {
            terminal.print(
                "Some output arrived while viewing details; Ctrl-O opens the recorded history.",
            )?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    provider: &dyn Provider,
    session: &SessionLog,
    config: &AgentConfig,
    prompt: &str,
    history: Vec<ChatMessage>,
    attachments: Vec<AttachmentRef>,
    store: Option<&AttachmentStore>,
    editor: &mut LineEditor,
    allow_session: &mut bool,
    hooks: Option<SharedHooks>,
    approval_keys: ApprovalKeys,
) -> Result<RunOutcome> {
    let (mut client, endpoint) = runtime_channel();
    let mut terminal = editor.start(Theme::detect_stdout())?;
    let run = run_interactive_with_attachments(
        provider,
        session,
        config,
        prompt,
        history,
        attachments,
        store,
        endpoint,
    );
    tokio::pin!(run);
    let mut ticks = tokio::time::interval(Duration::from_millis(25));
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let started = Instant::now();
    let input_run_id = niucode_core::now_ms();
    let mut sequence = 0_u64;
    let mut status = String::from("requesting");
    let mut streaming = false;
    let mut running_tool: Option<(String, Instant)> = None;
    let mut tools_completed = 0;
    let mut seen_tools = std::collections::HashSet::new();
    let mut tool_times: std::collections::HashMap<String, std::time::Duration> =
        std::collections::HashMap::new();
    let mut transcript = Transcript::new();
    let mut view: Option<View> = None;
    let mut unacknowledged: Vec<(String, String)> = Vec::new();
    let mut completed = None;
    loop {
        tokio::select! {
            // Drain events before accepting completion so final output is not lost.
            event = client.events.recv(), if !client.events.is_closed() || !client.events.is_empty() => {
                if let Some(event) = event {
                    if let RuntimeEvent::Agent(agent_event) = &event {
                        if let Some(hooks) = &hooks {
                            hooks.borrow_mut().on_event(agent_event);
                        }
                    }
                    match event {
                        RuntimeEvent::Agent(AgentEvent::TextDelta(mut text)) => {
                            if let Some(hooks) = &hooks {
                                hooks.borrow_mut().filter_text(&mut text);
                            }
                            status = "responding".into();
                            streaming = true;
                            transcript.delta(&mut terminal, view.is_some(), &text)?;
                        }
                        RuntimeEvent::Agent(AgentEvent::ReasoningDelta(_)) => status = "thinking · Ctrl-O recorded details".into(),
                        RuntimeEvent::Agent(AgentEvent::ToolCall { name, input }) => {
                            transcript.flush(&mut terminal, view.is_some())?;
                            streaming = false;
                            let _ = input;
                            if seen_tools.insert(name.clone()) {
                                let line = crate::presentation::tool_call(
                                    &name,
                                    &serde_json::Value::Null,
                                    terminal.size().0,
                                );
                                transcript.print(&mut terminal, view.is_some(), line)?;
                            }
                            running_tool = Some((name.clone(), Instant::now()));
                            status = format!("running {name}");
                        }
                        RuntimeEvent::Agent(AgentEvent::ToolResult { name, ok, summary, output }) => {
                            tools_completed += 1;
                            streaming = false;
                            if let Some((started_name, since)) = running_tool.take() {
                                *tool_times.entry(started_name).or_default() += since.elapsed();
                            }
                            status = format!("{name} finished");
                            if !ok {
                                let width = terminal.size().0;
                                let line = crate::presentation::tool_result(
                                    &name,
                                    false,
                                    &summary,
                                    width,
                                );
                                transcript.print(&mut terminal, view.is_some(), line)?;
                                if let Some(tail) = crate::presentation::tool_error_tail(&summary, &output, width) {
                                    transcript.print(&mut terminal, view.is_some(), tail)?;
                                }
                            }
                        }
                        RuntimeEvent::Agent(AgentEvent::Warning(message)) | RuntimeEvent::Notice(message) => {
                            transcript.print(&mut terminal, view.is_some(), format!("! {message}"))?;
                        }
                        RuntimeEvent::Agent(AgentEvent::Compaction { before_tokens, after_tokens, .. }) => {
                            transcript.print(&mut terminal, view.is_some(), format!("⇣ context {before_tokens} → {after_tokens}"))?;
                        }
                        RuntimeEvent::Agent(AgentEvent::TurnEnd { .. }) => {
                            streaming = false;
                            running_tool = None;
                            seen_tools.clear();
                            transcript.flush(&mut terminal, view.is_some())?;
                        }
                        RuntimeEvent::Agent(AgentEvent::Phase(phase)) => status = phase.as_str().into(),
                        RuntimeEvent::Input { id, queued, applied } => {
                            if let Some(index) = unacknowledged.iter().position(|(pending, _)| pending == &id) {
                                let (_, text) = unacknowledged.remove(index);
                                transcript.print(&mut terminal, view.is_some(), format!("you ▸ {text}"))?;
                            }
                            status = if applied { "input received" } else if queued { "task queued" } else { "input pending" }.into();
                            transcript.print(&mut terminal, view.is_some(), format!("↳ {status}"))?;
                        }
                        RuntimeEvent::InputRejected { id, reason } => {
                            if let Some(index) = unacknowledged.iter().position(|(pending, _)| pending == &id) {
                                let (_, text) = unacknowledged.remove(index);
                                transcript.print(&mut terminal, view.is_some(), format!("! input rejected: {reason}\n{text}"))?;
                                if editor.composer.text().is_empty() { editor.composer.insert(&text).map_err(anyhow::Error::msg)?; }
                            }
                        }
                        RuntimeEvent::TaskDone { id, .. } => {
                            transcript.flush(&mut terminal, view.is_some())?;
                            if let Some(review) = niucode_agent::task_review_by_id(&niucode_core::load(session.path())?.events, &id) {
                                let summary = crate::presentation::completion(&review, terminal.size().0);
                                transcript.print(&mut terminal, view.is_some(), summary)?;
                                if let Some(timings) =
                                    crate::presentation::tool_timings(&tool_times, terminal.size().0)
                                {
                                    transcript.print(&mut terminal, view.is_some(), timings)?;
                                }
                                tool_times.clear();
                            }
                        }
                        RuntimeEvent::Approval { request_id, request, .. } => {
                            if *allow_session {
                                send(&client, RuntimeCommand::Approve { request_id, allowed: true })?;
                            } else {
                                if let Some(active) = &mut view {
                                    active.show_approval(request_id, request, terminal.size().0);
                                } else {
                                    terminal.clear()?;
                                    view = Some(View::approval(
                                        request_id,
                                        request,
                                        terminal.size().0,
                                        approval_keys,
                                    )?);
                                }
                            }
                        }
                    }
                }
            }
            result = &mut run, if completed.is_none() => { completed = Some(result); }
            _ = tokio::signal::ctrl_c() => {
                crate::stop_computer(config);
                client.cancel();
            }
            _ = ticks.tick() => {
                // Bound input work per tick. The level-triggered TTY backend requires a positive poll timeout; it does not poll at Duration::ZERO.
                for _ in 0..32 {
                    let Some(event) = terminal.poll(Duration::from_millis(1))? else { break; };
                    if let Some(active) = &mut view {
                        match active.handle(event,terminal.size().0,terminal.size().1)? {
                            ViewAction::Stay => {},
                            action => {
                                let mut restored = false;
                                match action {
                                    ViewAction::Cancel => {
                                        crate::stop_computer(config);
                                        client.cancel();
                                    }
                                    ViewAction::Approve { id, allowed, session } => {
                                        *allow_session |= session;
                                        send(&client,RuntimeCommand::Approve { request_id:id,allowed })?;
                                        restored = active.resume_history(terminal.size().0)?;
                                    }
                                    _ => {}
                                }
                                if !restored { view.take(); transcript.resume(&mut terminal)?; }
                            }
                        }
                    } else if let Some(input) = editor.handle(event, terminal.size().0, "niucode ▸ ") {
                        let queued = matches!(&input, Input::Queue(_));
                        match input {
                            Input::Interrupted | Input::Eof => {
                                crate::stop_computer(config);
                                client.cancel();
                            }
                            Input::Details => { terminal.clear()?; view = Some(View::history(session, terminal.size().0, approval_keys)?); }
                            Input::Line(text) | Input::Queue(text) => {
                                // Local commands cannot silently become steering instructions.
                                if text.starts_with('/') || unacknowledged.len() >= 32 {
                                    transcript.print(&mut terminal, false, "Local commands are available after this task; Ctrl-O details, Ctrl-C cancel.".into())?;
                                    editor.composer.insert(&text).map_err(anyhow::Error::msg)?;
                                    continue;
                                }
                                sequence += 1;
                                let id = format!("{input_run_id}-{sequence}");
                                let command = RuntimeCommand::Submit { id: id.clone(), text: text.clone(), queued };
                                if let Err(e) = send(&client, command) {
                                    editor.composer.insert(&text).map_err(anyhow::Error::msg)?;
                                    transcript.print(&mut terminal, false, e.to_string())?;
                                } else { unacknowledged.push((id, text)); }
                            }
                        }
                    }
                }
                if let Some(active) = &mut view { active.draw()?; }
                else {
                    let footer_status = if let Some((name, since)) = &running_tool {
                        format!("{name} · {:.1}s", since.elapsed().as_secs_f32())
                    } else {
                        status.clone()
                    };
                    let footer = crate::presentation::activity(
                        &footer_status,
                        tools_completed,
                        started.elapsed().as_millis(),
                        Theme::detect_stdout().is_ansi()
                            && crate::presentation::animation_enabled(),
                        terminal.size().0,
                        streaming,
                    );
                    editor.draw(&mut terminal, "niucode ▸ ", &footer, &transcript.pending)?;
                }
            }
        }
        if completed.is_some() && client.events.is_empty() {
            break;
        }
    }
    view.take();
    transcript.resume(&mut terminal)?;
    transcript.flush(&mut terminal, false)?;
    for (id, text) in unacknowledged {
        terminal.print(&format!(
            "! input {id} was not acknowledged; preserved below:\n{text}"
        ))?;
        if editor.composer.text().is_empty() {
            editor.composer.insert(&text).map_err(anyhow::Error::msg)?;
        }
    }
    if completed.as_ref().is_some_and(Result::is_err) {
        if let Some(review) =
            niucode_agent::task_review(&niucode_core::load(session.path())?.events)
        {
            terminal.print(&crate::presentation::completion(&review, terminal.size().0))?;
        }
    }
    terminal.finish()?;
    let result = completed.expect("completed before exit");
    if let Some(hooks) = &hooks {
        if let Ok(outcome) = &result {
            hooks.borrow_mut().on_turn_end(outcome);
        }
    }
    result
}

fn send(client: &RuntimeClient, command: RuntimeCommand) -> Result<()> {
    client
        .commands
        .try_send(command)
        .map_err(|e| anyhow::anyhow!("runtime command not accepted: {e}"))
}

pub fn show_details(session: &SessionLog) -> Result<()> {
    let mut terminal = TerminalSession::enter(false, Theme::detect_stdout().is_ansi())?;
    let mut view = View::history(session, terminal.size().0, ApprovalKeys::default())?;
    loop {
        view.draw()?;
        if let Some(event) = terminal.poll(Duration::from_millis(100))? {
            if !matches!(
                view.handle(event, terminal.size().0, terminal.size().1)?,
                ViewAction::Stay
            ) {
                break;
            }
        }
    }
    drop(view);
    terminal.finish()?;
    Ok(())
}
