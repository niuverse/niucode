//! Local web surface for a session log.
//!
//! Loopback only. `/api/events` is a chunked SSE tail of the append-only log.
//! `POST /api/prompt` runs one read-only, approval-free agent turn; browser
//! approvals and streaming deltas are later slices (ADR-0036).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use base64::Engine;
use niucode_agent::{
    project_history, run_with_attachments, run_with_history, AgentConfig, ApprovalMode,
};
use niucode_attachments::{check_budget, AttachmentStore};
use niucode_core::AttachmentRef;
use niucode_core::{load, now_ms, Event, SessionLog};
use niucode_provider::Provider;
use niucode_tools::{NetworkMode, SandboxMode, SandboxPolicy};
use pulldown_cmark::{html, Options as MarkdownOptions, Parser};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;

use crate::provider_factory::ProviderFactory;

const INDEX: &str = include_str!("../assets/web.html");
const UNAUTHORIZED: &str = "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>niucode · token required</title><body style=\"font:15px system-ui;background:#0d1117;color:#e6edf3;padding:40px\"><h1>Token required</h1><p>Open the tokenized URL printed by <code>niucode web</code> (or <code>niuse web</code>).</p><p>Bare loopback URLs are rejected so other local pages cannot drive the agent.</p></body></html>";

type ContextProvider = Arc<dyn Fn() -> Value + Send + Sync>;
type NewSessionProvider = Arc<dyn Fn() -> Result<PathBuf> + Send + Sync>;
type SwitchSessionProvider = Arc<dyn Fn(&str) -> Result<PathBuf> + Send + Sync>;

pub struct Options {
    pub session: PathBuf,
    pub port: u16,
    pub preset: String,
    pub model: Option<String>,
    pub cwd: PathBuf,
    pub max_steps: usize,
    /// Override the agent system prompt; used by niuse's product shell.
    pub system_prompt: Option<String>,
    /// Product brand shown in the page header.
    pub brand: Option<String>,
    /// Optional product context: title, brand, and sidebar sections.
    pub context: Option<Value>,
    /// Optional live context. When set, `/api/context` calls it per request.
    pub context_provider: Option<ContextProvider>,
    /// Allow `POST /api/new` to start a fresh session in the same directory.
    pub allow_new: bool,
    /// Optional embedder hook for product-specific new-session semantics.
    pub new_session_provider: Option<NewSessionProvider>,
    /// Optional embedder hook for switching to an allowlisted session id.
    pub switch_session_provider: Option<SwitchSessionProvider>,
    /// Native desktop policy attached to browser prompts.
    pub computer: String,
}

struct PendingApproval {
    tool: String,
    sender: mpsc::Sender<bool>,
}

struct State {
    path: Mutex<PathBuf>,
    snapshot: Mutex<Vec<u8>>,
    provider: Option<Arc<dyn Provider>>,
    preset: String,
    model: String,
    cwd: PathBuf,
    max_steps: usize,
    system_prompt: Option<String>,
    context: Vec<u8>,
    context_provider: Option<ContextProvider>,
    brand: String,
    token: Option<String>,
    allow_new: bool,
    allowed_tools: Mutex<HashSet<String>>,
    new_session_provider: Option<NewSessionProvider>,
    switch_session_provider: Option<SwitchSessionProvider>,
    computer: Option<Arc<Mutex<niucode_agent::ComputerRuntime>>>,
    busy: AtomicBool,
    live: broadcast::Sender<Value>,
    pending: Mutex<HashMap<String, PendingApproval>>,
    next_approval: AtomicU64,
}

pub async fn serve(options: Options) -> Result<()> {
    let path = options.session;
    let model = match options.model {
        Some(model) => model,
        None => ProviderFactory::default_model(&options.preset)
            .unwrap_or_else(|_| "deepseek-flash".into()),
    };
    let provider = ProviderFactory::new(false, None)?
        .build(&options.preset, &model)
        .context("build web provider")?;
    let mut computer_config = AgentConfig::new(model.clone(), options.cwd.clone());
    computer_config.provider = options.preset.clone();
    crate::configure_computer(&mut computer_config, &options.computer)?;
    let computer = computer_config.computer;
    if !path.exists() {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());
        let log = match parent {
            Some(parent) => SessionLog::create(parent)?,
            None => SessionLog::create(Path::new("."))?,
        };
        log.append(&Event::SessionStart {
            id: log.id().to_string(),
            cwd: options.cwd.display().to_string(),
            model: model.clone(),
            provider: options.preset.clone(),
            ts: now_ms(),
        })?;
        log.append(&Event::Policy {
            sandbox: "read-only".into(),
            approval: "never".into(),
            network: "off".into(),
            backend: "web".into(),
            ts: now_ms(),
        })?;
        std::fs::rename(log.path(), &path)
            .with_context(|| format!("move session into {}", path.display()))?;
    }
    let loaded = load(&path).with_context(|| format!("load session {}", path.display()))?;
    let events = loaded
        .events
        .iter()
        .map(|event| {
            let mut value = serde_json::to_value(event).unwrap_or(Value::Null);
            add_html(&mut value);
            value
        })
        .collect::<Vec<_>>();
    let snapshot = serde_json::to_vec(&json!({
        "id": path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("session"),
        "path": path.display().to_string(),
        "warnings": loaded.warnings,
        "events": events,
    }))?;
    let (live, _) = broadcast::channel(256);
    let token = random_token();
    let listener = TcpListener::bind(("127.0.0.1", options.port))
        .await
        .context("bind 127.0.0.1")?;
    let address = listener.local_addr()?;
    println!(
        "web: http://{address}/?token={token} · {} events · workspace-write + ask · Ctrl-C to stop",
        loaded.events.len()
    );
    serve_listener(
        listener,
        Arc::new(State {
            path: Mutex::new(path),
            snapshot: Mutex::new(snapshot),
            provider: Some(provider),
            preset: options.preset,
            model,
            cwd: options.cwd,
            max_steps: options.max_steps,
            system_prompt: options.system_prompt,
            context: serde_json::to_vec(&options.context.unwrap_or_else(
                || json!({"brand": options.brand.clone().unwrap_or_else(|| "niucode".into())}),
            ))?,
            context_provider: options.context_provider,
            brand: options.brand.unwrap_or_else(|| "niucode".into()),
            token: Some(token),
            allow_new: options.allow_new,
            allowed_tools: Mutex::new(HashSet::new()),
            new_session_provider: options.new_session_provider,
            switch_session_provider: options.switch_session_provider,
            computer,
            busy: AtomicBool::new(false),
            live,
            pending: Mutex::new(HashMap::new()),
            next_approval: AtomicU64::new(1),
        }),
    )
    .await
}

async fn serve_listener(listener: TcpListener, state: Arc<State>) -> Result<()> {
    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();
        tokio::spawn(async move {
            let _ = handle(stream, state).await;
        });
    }
}

async fn read_request(stream: &mut TcpStream) -> std::io::Result<(String, String)> {
    const MAX_REQUEST: usize = 16 * 1024 * 1024;
    let mut bytes = Vec::with_capacity(8192);
    let mut chunk = [0_u8; 8192];
    let mut header_end = None;
    let mut content_length = 0;
    loop {
        let count = stream.read(&mut chunk).await?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if header_end.is_none() {
            if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                header_end = Some(position);
                let headers = String::from_utf8_lossy(&bytes[..position]);
                content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0)
                    .min(MAX_REQUEST);
            }
        }
        if let Some(position) = header_end {
            if bytes.len() >= position + 4 + content_length {
                break;
            }
        }
        if bytes.len() > MAX_REQUEST {
            break;
        }
    }
    let request = String::from_utf8_lossy(&bytes);
    let (head, body) = request.split_once("\r\n\r\n").unwrap_or((&request, ""));
    Ok((head.to_string(), body.to_string()))
}

#[cfg(not(test))]
const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const REQUEST_READ_TIMEOUT: Duration = Duration::from_millis(200);

async fn handle(mut stream: TcpStream, state: Arc<State>) -> Result<()> {
    let (head, body) =
        match tokio::time::timeout(REQUEST_READ_TIMEOUT, read_request(&mut stream)).await {
            Ok(request) => request?,
            Err(_) => {
                return send(
                    &mut stream,
                    "408 Request Timeout",
                    "application/json",
                    br#"{"error":"request read timed out"}"#.to_vec(),
                )
                .await;
            }
        };
    let mut parts = head.lines().next().unwrap_or("").split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    let route = path.split('?').next().unwrap_or(path);
    if let Some(host) = header(&head, "host") {
        if !loopback_host(host) {
            return send(
                &mut stream,
                "403 Forbidden",
                "application/json",
                br#"{"error":"non-loopback host"}"#.to_vec(),
            )
            .await;
        }
    }
    if method == "POST" {
        if let Some(origin) = header(&head, "origin") {
            if !loopback_origin(origin) {
                return send(
                    &mut stream,
                    "403 Forbidden",
                    "application/json",
                    br#"{"error":"cross-origin request rejected"}"#.to_vec(),
                )
                .await;
            }
        }
    }
    let query_authorized = state
        .token
        .as_deref()
        .is_some_and(|token| query_token(path) == Some(token));
    let cookie_authorized = state
        .token
        .as_deref()
        .is_some_and(|token| cookie_token(&head) == Some(token));
    if state.token.is_some() && !query_authorized && !cookie_authorized {
        if route == "/" {
            return send(
                &mut stream,
                "401 Unauthorized",
                "text/html; charset=utf-8",
                UNAUTHORIZED.as_bytes().to_vec(),
            )
            .await;
        }
        return send(
            &mut stream,
            "401 Unauthorized",
            "application/json",
            br#"{"error":"missing or invalid token"}"#.to_vec(),
        )
        .await;
    }
    let current_path = state.path.lock().unwrap().clone();
    if method == "GET" && route == "/api/events" {
        return stream_events(stream, current_path, state.live.subscribe()).await;
    }
    if method == "POST" && route == "/api/attach" {
        return attach(stream, state, &body).await;
    }
    if method == "POST" && route == "/api/prompt" {
        return prompt(stream, state, &body).await;
    }
    if method == "GET" && route == "/api/context" {
        let body = if let Some(provider) = &state.context_provider {
            serde_json::to_vec(&provider())?
        } else {
            state.context.clone()
        };
        return send(&mut stream, "200 OK", "application/json", body).await;
    }
    if method == "GET" && route == "/api/export" {
        let markdown = match load(&current_path) {
            Ok(loaded) => export_markdown(&loaded.events),
            Err(error) => {
                return send(
                    &mut stream,
                    "500 Internal Server Error",
                    "application/json",
                    serde_json::to_vec(&json!({"error": error.to_string()}))?,
                )
                .await;
            }
        };
        return send_download(&mut stream, markdown).await;
    }
    if method == "GET" && route == "/api/status" {
        let status = json!({
            "brand": state.brand,
            "provider": state.preset,
            "model": state.model,
            "cwd": state.cwd.display().to_string(),
            "sandbox": "workspace-write",
            "approval": "ask",
            "network": "off",
            "busy": state.busy.load(Ordering::Relaxed),
            "allow_new": state.allow_new,
            "computer": computer_status(&state.computer),
        });
        return send(
            &mut stream,
            "200 OK",
            "application/json",
            serde_json::to_vec(&status)?,
        )
        .await;
    }
    if method == "POST" && route == "/api/new" {
        return new_session(stream, state).await;
    }
    if method == "POST" && route == "/api/switch" {
        return switch_session(stream, state, &body).await;
    }
    if method == "POST" && route == "/api/approval" {
        return approval(stream, state, &body).await;
    }
    if method == "POST" && route == "/api/computer" {
        return computer_control(stream, state, &body).await;
    }
    let snapshot = state.snapshot.lock().unwrap().clone();
    if route == "/" && query_authorized && !cookie_authorized {
        if let Some(token) = &state.token {
            return send_with_cookie(&mut stream, token).await;
        }
    }
    let (status, kind, body) = response(method, route, &snapshot);
    send(&mut stream, status, kind, body).await
}

async fn prompt(mut stream: TcpStream, state: Arc<State>, body: &str) -> Result<()> {
    let Some(provider) = state.provider.clone() else {
        return send(
            &mut stream,
            "503 Service Unavailable",
            "application/json",
            br#"{"error":"no provider"}"#.to_vec(),
        )
        .await;
    };
    let prompt = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("prompt")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default();
    if prompt.trim().is_empty() {
        return send(
            &mut stream,
            "400 Bad Request",
            "application/json",
            br#"{"error":"prompt is required"}"#.to_vec(),
        )
        .await;
    }
    let value = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
    let attachments: Vec<AttachmentRef> = match value.get("attachments").cloned() {
        Some(value) => match serde_json::from_value(value) {
            Ok(attachments) => attachments,
            Err(_) => {
                return send(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    br#"{"error":"attachments must be a list of attachment references"}"#.to_vec(),
                )
                .await;
            }
        },
        None => Vec::new(),
    };
    if !attachments.is_empty() {
        let store = match session_store(&state) {
            Ok(store) => store,
            Err(error) => {
                return send(
                    &mut stream,
                    "500 Internal Server Error",
                    "application/json",
                    serde_json::to_vec(&json!({"error": error.to_string()}))?,
                )
                .await;
            }
        };
        for attachment in &attachments {
            if let Err(error) = store.load(attachment) {
                return send(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    serde_json::to_vec(&json!({"error": error.to_string()}))?,
                )
                .await;
            }
        }
        if let Err(error) = check_budget(&attachments, 64_000) {
            return send(
                &mut stream,
                "400 Bad Request",
                "application/json",
                serde_json::to_vec(&json!({"error": error.to_string()}))?,
            )
            .await;
        }
    }
    if state.busy.swap(true, Ordering::SeqCst) {
        return send(
            &mut stream,
            "409 Conflict",
            "application/json",
            br#"{"error":"a turn is already running"}"#.to_vec(),
        )
        .await;
    }
    let turn = state.clone();
    // A web turn uses the callback engine, whose future is not `Send`; run it
    // on a dedicated current-thread runtime instead of the connection task.
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(error) => {
                eprintln!("web: runtime: {error:#}");
                turn.busy.store(false, Ordering::SeqCst);
                return;
            }
        };
        let result = runtime.block_on(run_prompt(turn.clone(), provider, prompt, attachments));
        turn.busy.store(false, Ordering::SeqCst);
        let _ = turn.live.send(json!({"type": "context_changed"}));
        if let Err(error) = result {
            eprintln!("web: prompt failed: {error:#}");
        }
    });
    send(
        &mut stream,
        "202 Accepted",
        "application/json",
        br#"{"ok":true}"#.to_vec(),
    )
    .await
}

async fn attach(mut stream: TcpStream, state: Arc<State>, body: &str) -> Result<()> {
    let value = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
    let data = value
        .get("data")
        .and_then(Value::as_str)
        .filter(|data| !data.is_empty());
    let Some(data) = data else {
        return send(
            &mut stream,
            "400 Bad Request",
            "application/json",
            br#"{\"error\":\"data is required\"}"#.to_vec(),
        )
        .await;
    };
    let bytes = match base64::engine::general_purpose::STANDARD.decode(data) {
        Ok(bytes) => bytes,
        Err(_) => {
            return send(
                &mut stream,
                "400 Bad Request",
                "application/json",
                br#"{\"error\":\"data must be base64\"}"#.to_vec(),
            )
            .await;
        }
    };
    let name = value.get("name").and_then(Value::as_str);
    let alt = value.get("alt").and_then(Value::as_str);
    let store = session_store(&state)?;
    let reference = match store.put(&bytes, name, alt) {
        Ok(reference) => reference,
        Err(error) => {
            return send(
                &mut stream,
                "400 Bad Request",
                "application/json",
                serde_json::to_vec(&json!({"error": error.to_string()}))?,
            )
            .await;
        }
    };
    send(
        &mut stream,
        "200 OK",
        "application/json",
        serde_json::to_vec(&reference)?,
    )
    .await
}

fn session_store(state: &State) -> Result<AttachmentStore> {
    let path = state.path.lock().unwrap().clone();
    let root = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("attachments");
    Ok(AttachmentStore::open(root)?)
}

fn computer_status(runtime: &Option<Arc<Mutex<niucode_agent::ComputerRuntime>>>) -> Value {
    runtime.as_ref().map_or_else(
        || json!({"policy": "off", "armed": false, "configured": false}),
        |runtime| {
            let runtime = runtime.lock().unwrap();
            json!({
                "policy": runtime.policy().as_str(),
                "armed": runtime.is_armed(),
                "configured": true,
            })
        },
    )
}

async fn computer_control(mut stream: TcpStream, state: Arc<State>, body: &str) -> Result<()> {
    let Some(runtime) = state.computer.clone() else {
        return send(
            &mut stream,
            "409 Conflict",
            "application/json",
            br#"{"error":"computer use is not configured for this server"}"#.to_vec(),
        )
        .await;
    };
    let action = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("action")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default();
    if action != "arm" && action != "stop" {
        return send(
            &mut stream,
            "400 Bad Request",
            "application/json",
            br#"{"error":"action must be arm or stop"}"#.to_vec(),
        )
        .await;
    }
    // Do not hold the std mutex across the response await: the handler future
    // must stay Send.
    let (arm_error, release_error) = {
        let mut runtime = runtime.lock().unwrap();
        if action == "arm" {
            (runtime.arm().err().map(|error| error.to_string()), None)
        } else {
            (
                None,
                runtime
                    .emergency_stop()
                    .err()
                    .map(|error| error.to_string()),
            )
        }
    };
    if let Some(error) = arm_error {
        return send(
            &mut stream,
            "409 Conflict",
            "application/json",
            serde_json::to_vec(&json!({"error": error}))?,
        )
        .await;
    }
    let mut status = computer_status(&state.computer);
    if let Some(error) = release_error {
        status["release_error"] = Value::String(error);
    }
    send(
        &mut stream,
        "200 OK",
        "application/json",
        serde_json::to_vec(&status)?,
    )
    .await
}

async fn approval(mut stream: TcpStream, state: Arc<State>, body: &str) -> Result<()> {
    let value = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
    let id = value.get("id").and_then(Value::as_str).unwrap_or("");
    let allowed = value
        .get("allowed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let session = value
        .get("session")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let pending = state.pending.lock().unwrap().remove(id);
    if allowed && session {
        if let Some(pending) = &pending {
            state
                .allowed_tools
                .lock()
                .unwrap()
                .insert(pending.tool.clone());
        }
    }
    match pending {
        Some(pending) => {
            let _ = pending.sender.send(allowed);
            send(
                &mut stream,
                "200 OK",
                "application/json",
                br#"{"ok":true}"#.to_vec(),
            )
            .await
        }
        None => {
            send(
                &mut stream,
                "404 Not Found",
                "application/json",
                br#"{"error":"unknown approval"}"#.to_vec(),
            )
            .await
        }
    }
}

async fn switch_session(mut stream: TcpStream, state: Arc<State>, body: &str) -> Result<()> {
    let Some(provider) = state.switch_session_provider.clone() else {
        return send(
            &mut stream,
            "403 Forbidden",
            "application/json",
            br#"{"error":"session switching is not enabled"}"#.to_vec(),
        )
        .await;
    };
    let id = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| value.get("id").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default();
    if id.is_empty() {
        return send(
            &mut stream,
            "400 Bad Request",
            "application/json",
            br#"{"error":"id is required"}"#.to_vec(),
        )
        .await;
    }
    if state.busy.swap(true, Ordering::SeqCst) {
        return send(
            &mut stream,
            "409 Conflict",
            "application/json",
            br#"{"error":"a turn is already running"}"#.to_vec(),
        )
        .await;
    }
    let result = provider(&id);
    state.busy.store(false, Ordering::SeqCst);
    match result {
        Ok(path) => {
            let snapshot = session_snapshot(&path)?;
            *state.path.lock().unwrap() = path.clone();
            *state.snapshot.lock().unwrap() = snapshot;
            state.allowed_tools.lock().unwrap().clear();
            let _ = state.live.send(json!({"type": "session_switched"}));
            send(
                &mut stream,
                "200 OK",
                "application/json",
                serde_json::to_vec(&json!({
                    "id": path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("session"),
                    "path": path.display().to_string(),
                }))?,
            )
            .await
        }
        Err(error) => {
            send(
                &mut stream,
                "400 Bad Request",
                "application/json",
                serde_json::to_vec(&json!({"error": error.to_string()}))?,
            )
            .await
        }
    }
}

fn session_snapshot(path: &Path) -> Result<Vec<u8>> {
    let loaded = load(path)?;
    let events = loaded
        .events
        .iter()
        .map(|event| {
            let mut value = serde_json::to_value(event).unwrap_or(Value::Null);
            add_html(&mut value);
            value
        })
        .collect::<Vec<_>>();
    Ok(serde_json::to_vec(&json!({
        "id": path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("session"),
        "path": path.display().to_string(),
        "warnings": loaded.warnings,
        "events": events,
    }))?)
}

async fn new_session(mut stream: TcpStream, state: Arc<State>) -> Result<()> {
    if !state.allow_new {
        return send(
            &mut stream,
            "403 Forbidden",
            "application/json",
            br#"{"error":"new sessions are not enabled"}"#.to_vec(),
        )
        .await;
    }
    if state.busy.swap(true, Ordering::SeqCst) {
        return send(
            &mut stream,
            "409 Conflict",
            "application/json",
            br#"{"error":"a turn is already running"}"#.to_vec(),
        )
        .await;
    }
    let result = create_session(&state);
    state.busy.store(false, Ordering::SeqCst);
    match result {
        Ok(path) => {
            state.allowed_tools.lock().unwrap().clear();
            let body = serde_json::to_vec(&json!({
                "id": path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("session"),
                "path": path.display().to_string(),
            }))?;
            let _ = state.live.send(json!({"type": "session_switched"}));
            send(&mut stream, "200 OK", "application/json", body).await
        }
        Err(error) => {
            send(
                &mut stream,
                "500 Internal Server Error",
                "application/json",
                serde_json::to_vec(&json!({"error": error.to_string()}))?,
            )
            .await
        }
    }
}

fn create_session(state: &State) -> Result<PathBuf> {
    if let Some(provider) = &state.new_session_provider {
        let path = provider()?;
        *state.path.lock().unwrap() = path.clone();
        *state.snapshot.lock().unwrap() = serde_json::to_vec(&json!({
            "id": path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("session"),
            "path": path.display().to_string(),
            "warnings": [],
            "events": [],
        }))?;
        return Ok(path);
    }
    let current = state.path.lock().unwrap().clone();
    let parent = current
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let log = SessionLog::create(parent)?;
    let id = log.id().to_string();
    log.append(&Event::SessionStart {
        id: id.clone(),
        cwd: state.cwd.display().to_string(),
        model: state.model.clone(),
        provider: state.preset.clone(),
        ts: now_ms(),
    })?;
    log.append(&Event::Policy {
        sandbox: "workspace-write".into(),
        approval: "ask".into(),
        network: "off".into(),
        backend: "web".into(),
        ts: now_ms(),
    })?;
    let path = log.path().to_path_buf();
    drop(log);
    *state.path.lock().unwrap() = path.clone();
    *state.snapshot.lock().unwrap() = serde_json::to_vec(&json!({
        "id": id,
        "path": path.display().to_string(),
        "warnings": [],
        "events": [],
    }))?;
    Ok(path)
}

async fn run_prompt(
    state: Arc<State>,
    provider: Arc<dyn Provider>,
    prompt: String,
    attachments: Vec<AttachmentRef>,
) -> Result<()> {
    let live = state.live.clone();
    let approval_state = state.clone();
    let path = state.path.lock().unwrap().clone();
    let loaded = load(&path)?;
    let history = project_history(&loaded.events)?;
    let mut log = SessionLog::open(&path)?;
    let mut config = AgentConfig::new(state.model.clone(), state.cwd.clone());
    config.provider = state.preset.clone();
    config.max_steps = state.max_steps;
    config.sandbox = SandboxPolicy::new(SandboxMode::WorkspaceWrite, NetworkMode::Off);
    config.approval = ApprovalMode::Ask;
    config.allow_unsandboxed = false;
    config.computer = state.computer.clone();
    if let Some(prompt) = &state.system_prompt {
        config.system_prompt = prompt.clone();
    }
    let mut approver = |request: &niucode_agent::ApprovalRequest| {
        if approval_state
            .allowed_tools
            .lock()
            .unwrap()
            .contains(&request.tool)
        {
            return true;
        }
        let id = format!(
            "a-{}",
            approval_state.next_approval.fetch_add(1, Ordering::SeqCst)
        );
        let (sender, receiver) = mpsc::channel();
        approval_state.pending.lock().unwrap().insert(
            id.clone(),
            PendingApproval {
                tool: request.tool.clone(),
                sender,
            },
        );
        let _ = approval_state.live.send(json!({
            "type": "approval",
            "id": id,
            "tool": request.tool,
            "summary": request.summary,
            "preview": preview_json(request.preview.as_ref()),
            "image": request.image,
        }));
        let allowed = receiver
            .recv_timeout(Duration::from_secs(300))
            .unwrap_or(false);
        approval_state.pending.lock().unwrap().remove(&id);
        allowed
    };
    let mut streamed = String::new();
    let mut last_render = std::time::Instant::now();
    let mut last_rendered_len = 0;
    let mut observer = move |event: niucode_agent::AgentEvent| {
        match &event {
            niucode_agent::AgentEvent::TextDelta(text) => {
                streamed.push_str(text);
                // Re-render at most every 100ms or 64 new bytes. The durable
                // assistant event always renders the final text.
                let due = last_rendered_len == 0
                    || streamed.len().saturating_sub(last_rendered_len) >= 64
                    || last_render.elapsed() >= std::time::Duration::from_millis(100);
                if due {
                    last_render = std::time::Instant::now();
                    last_rendered_len = streamed.len();
                    let _ = live.send(json!({
                        "type": "text_delta",
                        "text": text,
                        "html": markdown_html(&streamed),
                    }));
                } else {
                    let _ = live.send(json!({"type": "text_delta", "text": text}));
                }
            }
            _ => {
                let _ = live.send(crate::event_json(&event));
            }
        }
    };
    if attachments.is_empty() {
        run_with_history(
            provider.as_ref(),
            &mut log,
            &config,
            &prompt,
            history,
            &mut approver,
            &mut observer,
        )
        .await?;
    } else {
        let store = session_store(&state)?;
        run_with_attachments(
            provider.as_ref(),
            &mut log,
            &config,
            &prompt,
            attachments,
            &store,
            &mut approver,
            &mut observer,
        )
        .await?;
    }
    Ok(())
}

async fn send(stream: &mut TcpStream, status: &str, kind: &str, body: Vec<u8>) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'\r\nCache-Control: no-store\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.shutdown().await?;
    Ok(())
}

async fn stream_events(
    mut stream: TcpStream,
    path: PathBuf,
    mut live: broadcast::Receiver<Value>,
) -> Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nTransfer-Encoding: chunked\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    let mut offset = 0_u64;
    let mut partial = String::new();
    let mut tick = tokio::time::interval(Duration::from_millis(350));
    loop {
        tokio::select! {
            _ = tick.tick() => {
                let bytes = match tokio::fs::read(&path).await {
                    Ok(bytes) => bytes,
                    Err(_) => continue,
                };
                if bytes.len() as u64 > offset {
                    let fresh = &bytes[offset as usize..];
                    offset = bytes.len() as u64;
                    partial.push_str(&String::from_utf8_lossy(fresh));
                    while let Some(index) = partial.find('\n') {
                        let rest = partial.split_off(index + 1);
                        let line = std::mem::replace(&mut partial, rest);
                        let line = line.trim_end_matches('\r');
                        if line.is_empty() {
                            continue;
                        }
                        write_chunk(&mut stream, &format!("data: {line}\n\n")).await?;
                    }
                } else {
                    write_chunk(&mut stream, ": ping\n\n").await?;
                }
            }
            event = live.recv() => {
                if let Ok(value) = event {
                    write_chunk(&mut stream, &format!("data: {value}\n\n")).await?;
                }
            }
        }
    }
}

async fn write_chunk(stream: &mut TcpStream, text: &str) -> Result<()> {
    let head = format!("{:x}\r\n", text.len());
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(text.as_bytes()).await?;
    stream.write_all(b"\r\n").await?;
    Ok(())
}

fn cookie_token(head: &str) -> Option<&str> {
    header(head, "cookie")?.split(';').find_map(|part| {
        part.trim()
            .strip_prefix("niucode_token=")
            .filter(|token| !token.is_empty())
    })
}

fn query_token(path: &str) -> Option<&str> {
    path.split_once('?')?.1.split('&').find_map(|part| {
        part.strip_prefix("token=")
            .filter(|token| !token.is_empty())
    })
}

async fn send_with_cookie(stream: &mut TcpStream, token: &str) -> Result<()> {
    let body = INDEX.as_bytes();
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nSet-Cookie: niucode_token={token}; HttpOnly; SameSite=Strict; Path=/\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'\r\nCache-Control: no-store\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.shutdown().await?;
    Ok(())
}

fn random_token() -> String {
    let mut bytes = [0_u8; 16];
    #[cfg(unix)]
    {
        use std::io::Read;
        if let Ok(mut file) = std::fs::File::open("/dev/urandom") {
            let _ = file.read_exact(&mut bytes);
        }
    }
    if bytes.iter().all(|byte| *byte == 0) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (nanos >> ((index % 16) * 8)) as u8;
        }
    }
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim().eq_ignore_ascii_case(name) {
            Some(value.trim())
        } else {
            None
        }
    })
}

fn loopback_host(value: &str) -> bool {
    let host = if let Some(rest) = value.strip_prefix('[') {
        rest.split(']').next().unwrap_or(rest)
    } else {
        value.split(':').next().unwrap_or(value)
    };
    matches!(
        host.to_ascii_lowercase().as_str(),
        "127.0.0.1" | "localhost" | "::1"
    )
}

fn loopback_origin(value: &str) -> bool {
    let rest = value
        .strip_prefix("http://")
        .or_else(|| value.strip_prefix("https://"));
    rest.is_some_and(loopback_host)
}

fn preview_json(preview: Option<&niucode_tools::ChangePreview>) -> Value {
    let Some(preview) = preview else {
        return Value::Null;
    };
    let body = match &preview.body {
        niucode_tools::PreviewBody::Diff { files } => json!({
            "kind": "diff",
            "files": files.iter().map(|file| json!({
                "path": file.path,
                "is_new": file.is_new,
                "added": file.added,
                "removed": file.removed,
                "hunks": file.hunks.iter().map(|hunk| json!({
                    "old_start": hunk.old_start,
                    "old_len": hunk.old_len,
                    "new_start": hunk.new_start,
                    "new_len": hunk.new_len,
                    "lines": hunk.lines.iter().map(|line| json!({
                        "kind": match line.kind {
                            niucode_tools::LineKind::Context => "context",
                            niucode_tools::LineKind::Add => "add",
                            niucode_tools::LineKind::Remove => "remove",
                        },
                        "old_line": line.old_line,
                        "new_line": line.new_line,
                        "text": line.text,
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }),
        niucode_tools::PreviewBody::Command { command } => json!({
            "kind": "command",
            "command": command,
        }),
        niucode_tools::PreviewBody::Note { message } => json!({
            "kind": "note",
            "message": message,
        }),
    };
    json!({
        "body": body,
        "added": preview.added,
        "removed": preview.removed,
        "warning": preview.warning,
    })
}

fn harden_links(html: &str) -> String {
    let mut output = String::new();
    let mut rest = html;
    while let Some(start) = rest.find("<a ") {
        output.push_str(&rest[..start]);
        let tail = &rest[start..];
        let Some(end) = tail.find('>') else {
            output.push_str(tail);
            return output;
        };
        let tag = &tail[..=end];
        let href = tag.find("href=\"").and_then(|start| {
            let rest = &tag[start + 6..];
            rest.find('"').map(|end| &rest[..end])
        });
        let mut hardened = tag.replacen(
            "<a ",
            "<a target=\"_blank\" rel=\"noreferrer noopener\" ",
            1,
        );
        if let Some(href) = href {
            if !safe_link(href) {
                let scheme = href
                    .split(':')
                    .next()
                    .unwrap_or("unknown")
                    .chars()
                    .filter(|character| character.is_ascii_alphanumeric())
                    .take(16)
                    .collect::<String>();
                hardened = hardened.replacen(
                    &format!("href=\"{href}\""),
                    &format!("href=\"#\" data-blocked=\"{scheme}:\""),
                    1,
                );
            }
        }
        output.push_str(&hardened);
        rest = &tail[end + 1..];
    }
    output.push_str(rest);
    output
}

fn safe_link(href: &str) -> bool {
    let trimmed = href.trim_start_matches(|character: char| {
        character.is_ascii_control() || character.is_ascii_whitespace()
    });
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("https://")
        || lower.starts_with("http://")
        || lower.starts_with("mailto:")
        || trimmed.starts_with('#')
}

fn strip_images(html: &str) -> String {
    let mut output = String::new();
    let mut rest = html;
    while let Some(index) = rest.find("<img") {
        output.push_str(&rest[..index]);
        let tag = &rest[index..];
        let Some(end) = tag.find('>') else {
            output.push_str(tag);
            return output;
        };
        let tag = &tag[..=end];
        let alt = tag
            .find("alt=\"")
            .and_then(|start| {
                let rest = &tag[start + 5..];
                rest.find('"').map(|end| &rest[..end])
            })
            .unwrap_or("");
        if alt.is_empty() {
            output.push_str("<span class=\"image-alt\">[image]</span>");
        } else {
            output.push_str(&format!("<span class=\"image-alt\">[image: {alt}]</span>"));
        }
        rest = &rest[index + end + 1..];
    }
    output.push_str(rest);
    output
}

fn add_html(value: &mut Value) {
    if value.get("type").and_then(Value::as_str) != Some("assistant_message") {
        return;
    }
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        value["html"] = Value::String(markdown_html(text));
    }
}

fn markdown_html(text: &str) -> String {
    let (text, code) = extract_code(text);
    let (text, math) = prepare_math(&text);
    let mut options = MarkdownOptions::empty();
    options.insert(MarkdownOptions::ENABLE_TABLES);
    options.insert(MarkdownOptions::ENABLE_STRIKETHROUGH);
    options.insert(MarkdownOptions::ENABLE_TASKLISTS);
    options.insert(MarkdownOptions::ENABLE_FOOTNOTES);
    options.insert(MarkdownOptions::ENABLE_GFM);
    let parser = Parser::new_ext(&text, options).map(|event| match event {
        pulldown_cmark::Event::Html(raw) => pulldown_cmark::Event::Text(raw),
        pulldown_cmark::Event::InlineHtml(raw) => pulldown_cmark::Event::Text(raw),
        event => event,
    });
    let mut output = String::new();
    html::push_html(&mut output, parser);
    output = strip_images(&output);
    for (index, rendered) in math.into_iter().enumerate() {
        output = output.replace(&format!("@@NIUMATH{index}@@"), &rendered);
    }
    for (index, rendered) in code.into_iter().enumerate() {
        let token = format!("@@NIUCODE{index}@@");
        output = output
            .replace(&format!("<p>{token}</p>"), &rendered)
            .replace(&token, &rendered);
    }
    harden_links(&output)
}

fn extract_code(text: &str) -> (String, Vec<String>) {
    let mut output = String::new();
    let mut blocks: Vec<String> = Vec::new();
    let mut language = String::new();
    let mut body = String::new();
    let mut in_code = false;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if !in_code && trimmed.starts_with("```") {
            in_code = true;
            language = trimmed.trim_start_matches("```").trim().to_string();
            body.clear();
            blocks.push(String::new());
            output.push_str(&format!("\n@@NIUCODE{}@@\n", blocks.len() - 1));
            continue;
        }
        if in_code && trimmed.starts_with("```") {
            let rendered = render_code(&language, &body);
            let index = blocks.len() - 1;
            blocks[index] = rendered;
            in_code = false;
            continue;
        }
        if in_code {
            body.push_str(line);
        } else {
            output.push_str(line);
        }
    }
    if in_code {
        let rendered = render_code(&language, &body);
        let index = blocks.len() - 1;
        blocks[index] = rendered;
    }
    (output, blocks)
}

fn render_code(language: &str, body: &str) -> String {
    let language = language
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        .take(24)
        .collect::<String>();
    let label = if language.is_empty() {
        "code".to_string()
    } else {
        language
    };
    let mut html = format!("<pre class=\"code\"><code class=\"language-{label}\">");
    let lines: Vec<&str> = body.split('\n').collect();
    let mut highlighter = crate::syntax::Highlighter::new();
    for (index, line) in lines.iter().enumerate() {
        for token in highlighter.line(line) {
            let class = match token.kind {
                crate::syntax::TokenKind::Keyword => "tok-keyword",
                crate::syntax::TokenKind::String => "tok-string",
                crate::syntax::TokenKind::Comment => "tok-comment",
                crate::syntax::TokenKind::Number => "tok-number",
                crate::syntax::TokenKind::Function => "tok-function",
                crate::syntax::TokenKind::Plain => "",
            };
            if class.is_empty() {
                html.push_str(&escape_html(&token.text));
            } else {
                html.push_str(&format!(
                    "<span class=\"{class}\">{}</span>",
                    escape_html(&token.text)
                ));
            }
        }
        if index + 1 < lines.len() {
            html.push('\n');
        }
    }
    html.push_str("</code></pre>");
    html
}

fn prepare_math(text: &str) -> (String, Vec<String>) {
    let mut rendered = Vec::new();
    let mut output = String::new();
    let mut code = false;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            code = !code;
            output.push_str(line);
            continue;
        }
        if code {
            output.push_str(line);
            continue;
        }
        let mut index = 0;
        let chars: Vec<char> = line.chars().collect();
        while index < chars.len() {
            if chars[index] == '\\' && index + 1 < chars.len() {
                output.push(chars[index]);
                output.push(chars[index + 1]);
                index += 2;
                continue;
            }
            if chars[index] != '$' {
                output.push(chars[index]);
                index += 1;
                continue;
            }
            let display = chars.get(index + 1) == Some(&'$');
            let marker = if display { "$$" } else { "$" };
            let offset = marker.len();
            if let Some(end) = line[index + offset..].find(marker) {
                let end = index + offset + end;
                let expression: String = chars[index + offset..end].iter().collect();
                if !expression.trim().is_empty() && !expression.contains('\n') {
                    let style = if display {
                        latex2mathml::DisplayStyle::Block
                    } else {
                        latex2mathml::DisplayStyle::Inline
                    };
                    let value = latex2mathml::latex_to_mathml(expression.trim(), style)
                        .unwrap_or_else(|_| {
                            format!(
                                "<code class=\"math\">{}</code>",
                                escape_html(expression.trim())
                            )
                        });
                    rendered.push(value);
                    output.push_str(&format!("@@NIUMATH{}@@", rendered.len() - 1));
                    index = end + marker.len();
                    continue;
                }
            }
            output.push('$');
            index += 1;
        }
    }
    (output, rendered)
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn export_markdown(events: &[Event]) -> String {
    let mut out = String::from("# niucode session\n\n");
    for event in events {
        match event {
            Event::SessionStart {
                cwd,
                model,
                provider,
                ..
            } => {
                out.push_str(&format!(
                    "- cwd: `{cwd}`\n- model: `{model}`\n- provider: `{provider}`\n\n"
                ));
            }
            Event::UserMessage { text, .. } => {
                out.push_str("## User\n\n");
                out.push_str(text);
                out.push_str("\n\n");
            }
            Event::AssistantMessage {
                text, tool_calls, ..
            } => {
                if !text.trim().is_empty() {
                    out.push_str("## Assistant\n\n");
                    out.push_str(text);
                    out.push_str("\n\n");
                }
                for call in tool_calls {
                    out.push_str(&format!("### Tool call · {}\n\n````json\n", call.name));
                    out.push_str(
                        &serde_json::to_string_pretty(&call.input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    );
                    out.push_str("\n````\n\n");
                }
            }
            Event::ToolResult { ok, output, .. } => {
                out.push_str(&format!(
                    "### Tool result ({})\n\n````text\n",
                    if *ok { "ok" } else { "error" }
                ));
                out.push_str(output);
                out.push_str("\n````\n\n");
            }
            Event::TurnEnd { usage, .. } => {
                out.push_str(&format!(
                    "_tokens: {} in / {} out / {} cached_\n\n",
                    usage.input_tokens, usage.output_tokens, usage.cached_tokens
                ));
            }
            Event::Compaction { summary, .. } => {
                out.push_str("## Compaction summary\n\n");
                out.push_str(summary);
                out.push_str("\n\n");
            }
            _ => {}
        }
    }
    out
}

async fn send_download(stream: &mut TcpStream, markdown: String) -> Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/markdown; charset=utf-8\r\nContent-Disposition: attachment; filename=\"niucode-session.md\"\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nCache-Control: no-store\r\n\r\n",
        markdown.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(markdown.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

fn response(method: &str, path: &str, snapshot: &[u8]) -> (&'static str, &'static str, Vec<u8>) {
    match (method, path) {
        ("GET", "/") => (
            "200 OK",
            "text/html; charset=utf-8",
            INDEX.as_bytes().to_vec(),
        ),
        ("GET", "/api/session") => ("200 OK", "application/json", snapshot.to_vec()),
        ("GET", "/api/health") => ("200 OK", "application/json", br#"{"ok":true}"#.to_vec()),
        ("POST", "/api/prompt") => (
            "501 Not Implemented",
            "application/json",
            br#"{"error":"interactive chat is not implemented in v0"}"#.to_vec(),
        ),
        _ => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found\n".to_vec(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn state_with(context: Vec<u8>) -> Arc<State> {
        Arc::new(State {
            path: Mutex::new(PathBuf::new()),
            snapshot: Mutex::new(br#"{"events":[]}"#.to_vec()),
            provider: None,
            preset: "deepseek".into(),
            model: "deepseek-flash".into(),
            cwd: PathBuf::from("."),
            max_steps: 1,
            system_prompt: None,
            context,
            context_provider: None,
            brand: "niucode".into(),
            token: None,
            allow_new: true,
            allowed_tools: Mutex::new(HashSet::new()),
            new_session_provider: None,
            switch_session_provider: None,
            computer: None,
            busy: AtomicBool::new(false),
            live: broadcast::channel(8).0,
            pending: Mutex::new(HashMap::new()),
            next_approval: AtomicU64::new(1),
        })
    }

    fn state_with_computer(computer: Arc<Mutex<niucode_agent::ComputerRuntime>>) -> Arc<State> {
        Arc::new(State {
            path: Mutex::new(PathBuf::new()),
            snapshot: Mutex::new(br#"{"events":[]}"#.to_vec()),
            provider: None,
            preset: "deepseek".into(),
            model: "deepseek-flash".into(),
            cwd: PathBuf::from("."),
            max_steps: 1,
            system_prompt: None,
            context: br#"{}"#.to_vec(),
            context_provider: None,
            brand: "niucode".into(),
            token: None,
            allow_new: true,
            allowed_tools: Mutex::new(HashSet::new()),
            new_session_provider: None,
            switch_session_provider: None,
            computer: Some(computer),
            busy: AtomicBool::new(false),
            live: broadcast::channel(8).0,
            pending: Mutex::new(HashMap::new()),
            next_approval: AtomicU64::new(1),
        })
    }

    fn state() -> Arc<State> {
        state_with(br#"{}"#.to_vec())
    }

    fn state_at(path: PathBuf, allow_new: bool) -> Arc<State> {
        Arc::new(State {
            path: Mutex::new(path),
            snapshot: Mutex::new(Vec::new()),
            provider: None,
            preset: "deepseek".into(),
            model: "deepseek-flash".into(),
            cwd: PathBuf::from("."),
            max_steps: 1,
            system_prompt: None,
            context: br#"{}"#.to_vec(),
            context_provider: None,
            brand: "niucode".into(),
            token: None,
            allow_new,
            allowed_tools: Mutex::new(HashSet::new()),
            new_session_provider: None,
            switch_session_provider: None,
            computer: None,
            busy: AtomicBool::new(false),
            live: broadcast::channel(8).0,
            pending: Mutex::new(HashMap::new()),
            next_approval: AtomicU64::new(1),
        })
    }

    #[test]
    fn export_markdown_keeps_user_assistant_and_tool_output() {
        let events = vec![
            Event::SessionStart {
                id: "s".into(),
                cwd: "/w".into(),
                model: "m".into(),
                provider: "deepseek".into(),
                ts: 0,
            },
            Event::UserMessage {
                text: "hello".into(),
                attachments: Vec::new(),
                ts: 0,
            },
            Event::AssistantMessage {
                text: "world".into(),
                reasoning: None,
                tool_calls: vec![niucode_core::ToolCall {
                    id: "c1".into(),
                    name: "write".into(),
                    input: serde_json::json!({"path":"a.txt"}),
                }],
                ts: 0,
            },
            Event::ToolResult {
                call_id: "c".into(),
                ok: true,
                output: "output".into(),
                ts: 0,
            },
        ];
        let markdown = export_markdown(&events);
        assert!(markdown.starts_with("# niucode session"));
        assert!(markdown.contains("## User\n\nhello"));
        assert!(markdown.contains("## Assistant\n\nworld"));
        assert!(markdown.contains("````text\noutput\n````"));
        assert!(markdown.contains("### Tool call · write"));
        assert!(markdown.contains("a.txt"));
        assert!(markdown.contains("cwd: `/w`"));
    }

    #[test]
    fn markdown_html_escapes_raw_html_and_renders_tables() {
        let html = markdown_html(
            "**bold**\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n<script>alert(1)</script>",
        );
        assert!(html.contains("<strong>bold</strong>"), "{html}");
        assert!(html.contains("<table>"), "{html}");
        assert!(html.contains("<td>1</td>"), "{html}");
        assert!(!html.contains("<script>"), "{html}");
        assert!(html.contains("&lt;script&gt;"), "{html}");
    }

    #[test]
    fn only_loopback_hosts_and_origins_are_accepted() {
        assert!(loopback_host("127.0.0.1:8799"));
        assert!(loopback_host("localhost:8799"));
        assert!(loopback_host("[::1]:8799"));
        assert!(!loopback_host("evil.example"));
        assert!(loopback_origin("http://127.0.0.1:8799"));
        assert!(loopback_origin("http://localhost:8799"));
        assert!(!loopback_origin("https://evil.example"));
        assert!(!loopback_origin("null"));
    }

    #[tokio::test]
    async fn stalled_request_bodies_time_out() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state()));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                b"POST /api/prompt HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{}",
            )
            .await
            .unwrap();
        let mut reply = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut reply))
            .await
            .unwrap()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 408"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        task.abort();
    }

    #[tokio::test]
    async fn reads_complete_request_bodies_across_chunks() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_request(&mut stream).await.unwrap()
        });
        let mut client = TcpStream::connect(address).await.unwrap();
        let body = "a".repeat(20_000);
        client
            .write_all(
                format!(
                    "POST /api/prompt HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let (head, received) = server.await.unwrap();
        assert!(head.starts_with("POST /api/prompt"));
        assert_eq!(received.len(), 20_000);
    }

    #[tokio::test]
    async fn attachment_upload_stores_base64_under_the_session() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::write(&path, "").unwrap();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state_at(path, true)));

        let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nimage");
        let body = serde_json::to_vec(&json!({"name":"shot.png","data":png})).unwrap();
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /api/attach HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.write_all(&body).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 200"), "{text}");
        assert!(text.contains("image/png"), "{text}");
        assert!(text.contains("shot.png"), "{text}");

        let body = br#"{"name":"bad.png","data":"not base64"}"#.to_vec();
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /api/attach HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.write_all(&body).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 400"));
        task.abort();
    }

    #[tokio::test]
    async fn browser_prompt_sends_an_uploaded_image() {
        use niucode_provider::{ScriptedProvider, StreamEvent};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::write(&path, "").unwrap();
        let scripted = Arc::new(ScriptedProvider::from_turns(vec![vec![
            StreamEvent::TextDelta("seen".into()),
            StreamEvent::Done {
                finish_reason: Some("stop".into()),
            },
        ]]));
        let mut state = state_at(path, true);
        Arc::get_mut(&mut state).unwrap().provider = Some(scripted.clone());
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state));

        let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nimage");
        let upload = serde_json::to_vec(&json!({"name":"shot.png","data":png})).unwrap();
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /api/attach HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    upload.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.write_all(&upload).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        let attachment: AttachmentRef =
            serde_json::from_str(text.split_once("\r\n\r\n").unwrap().1).unwrap();

        let body =
            serde_json::to_vec(&json!({"prompt":"look","attachments":[attachment]})).unwrap();
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /api/prompt HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.write_all(&body).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 202"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while scripted.requests().is_empty() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let requests = scripted.requests();
        assert_eq!(requests.len(), 1);
        let user = requests[0].messages.last().unwrap();
        assert!(user.has_images(), "request did not carry the image part");
        task.abort();
    }

    #[tokio::test]
    async fn access_tokens_are_required_and_cookie_bootstrapped() {
        let mut state = state_at(PathBuf::from("unused.jsonl"), true);
        Arc::get_mut(&mut state).unwrap().token = Some("secret".into());
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state));
        for (request, expected) in [
            (
                "GET /api/health HTTP/1.1\r\nHost: localhost\r\n\r\n",
                "HTTP/1.1 401",
            ),
            (
                "GET /api/health?token=secret HTTP/1.1\r\nHost: localhost\r\n\r\n",
                "HTTP/1.1 200",
            ),
            (
                "GET / HTTP/1.1\r\nHost: localhost\r\nCookie: niucode_token=secret\r\n\r\n",
                "HTTP/1.1 200",
            ),
        ] {
            let mut stream = TcpStream::connect(address).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut reply = Vec::new();
            stream.read_to_end(&mut reply).await.unwrap();
            assert!(
                String::from_utf8_lossy(&reply).starts_with(expected),
                "{}",
                String::from_utf8_lossy(&reply)
            );
        }
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /?token=secret HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 200"), "{text}");
        assert!(text.contains("Set-Cookie: niucode_token=secret"), "{text}");
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).contains("Token required"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        task.abort();
    }

    #[tokio::test]
    async fn cross_origin_and_foreign_host_requests_are_rejected() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state()));
        for request in [
            "POST /api/prompt HTTP/1.1\r\nHost: localhost\r\nOrigin: https://evil.example\r\nContent-Length: 0\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: evil.example\r\n\r\n",
        ] {
            let mut stream = TcpStream::connect(address).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut reply = Vec::new();
            stream.read_to_end(&mut reply).await.unwrap();
            assert!(
                String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 403"),
                "{}",
                String::from_utf8_lossy(&reply)
            );
        }
        task.abort();
    }

    #[test]
    fn links_open_safely_and_block_dangerous_schemes() {
        let html = markdown_html(
            "[ok](https://example.com) [bad](data:text/html,x) [js](javascript:alert(1))",
        );
        assert!(html.contains("target=\"_blank\""), "{html}");
        assert!(html.contains("rel=\"noreferrer noopener\""), "{html}");
        assert!(html.contains("data-blocked=\"data:"), "{html}");
        assert!(html.contains("data-blocked=\"javascript:"), "{html}");
        assert!(!html.contains("href=\"data:"), "{html}");
        assert!(!html.contains("href=\"javascript:"), "{html}");
    }

    #[test]
    fn task_lists_render_as_checkboxes() {
        let html = markdown_html("- [x] done\n- [ ] todo");
        assert!(html.contains("type=\"checkbox\""), "{html}");
        assert!(html.contains("checked"), "{html}");
        assert!(html.contains("disabled"), "{html}");
    }

    #[test]
    fn gfm_alerts_render_with_classes() {
        let html = markdown_html("> [!NOTE]\n> body");
        assert!(html.contains("markdown-alert-note"), "{html}");
    }

    #[test]
    fn footnotes_render_with_references() {
        let html = markdown_html("Text[^1]\n\n[^1]: note");
        assert!(html.contains("footnote"), "{html}");
        assert!(html.contains("note"), "{html}");
    }

    #[test]
    fn remote_images_do_not_load() {
        let html = markdown_html("![tracker](https://example.invalid/pixel.png)");
        assert!(!html.contains("<img"), "{html}");
        assert!(!html.contains("https://example.invalid"), "{html}");
        assert!(html.contains("image: tracker"), "{html}");
    }

    #[test]
    fn fenced_triple_strings_keep_their_state() {
        let html = markdown_html("```python\ntext = \"\"\"start\nstill inside\nend\"\"\" + 1\n```");
        let strings = html.matches("tok-string").count();
        assert!(strings >= 2, "{html}");
        assert!(html.contains("tok-number"), "{html}");
    }

    #[test]
    fn fenced_block_comments_keep_their_state() {
        let html = markdown_html("```rust\n/* start\nstill */ let x = 1;\n```");
        let comments = html.matches("tok-comment").count();
        assert!(comments >= 2, "{html}");
        assert!(html.contains("tok-keyword"), "{html}");
    }

    #[test]
    fn fenced_code_gets_token_classes_and_stays_escaped() {
        let html = markdown_html("```rust\nlet x = \"hi\"; // note\n<script>\n```");
        assert!(html.contains("<pre class=\"code\">"), "{html}");
        assert!(html.contains("tok-keyword"), "{html}");
        assert!(html.contains("tok-string"), "{html}");
        assert!(html.contains("tok-comment"), "{html}");
        assert!(!html.contains("<script>"), "{html}");
        assert!(html.contains("&lt;script&gt;"), "{html}");
        assert!(!html.contains("```"), "{html}");
    }

    #[test]
    fn math_is_rendered_as_mathml() {
        let html = markdown_html("energy $E = mc^2$ and\\n\\n$$\\n\\alpha + \\beta\\n$$");
        assert!(html.contains("<math"), "{html}");
        assert!(!html.contains('$') || html.contains("energy"), "{html}");
    }

    #[test]
    fn routes_are_small_and_explicit() {
        let session = br#"{"events":[]}"#;
        assert_eq!(response("GET", "/api/health", session).0, "200 OK");
        assert_eq!(response("GET", "/api/session", session).2, session.to_vec());
        assert_eq!(response("GET", "/", session).1, "text/html; charset=utf-8");
        assert_eq!(response("GET", "/nope", session).0, "404 Not Found");
        assert_eq!(
            response("POST", "/api/prompt", session).0,
            "501 Not Implemented"
        );
    }

    #[tokio::test]
    async fn serves_a_session_over_loopback_http() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state()));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /api/session HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
        assert!(text.contains("application/json"), "{text}");
        assert!(text.ends_with("{\"events\":[]}"), "{text}");
        task.abort();
    }

    #[tokio::test]
    async fn serves_product_context_over_loopback() {
        let state = state_with(
            br#"{"brand":"niuse","sections":[{"title":"identity","lines":["persona"]}]}"#.to_vec(),
        );
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /api/context HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 200"), "{text}");
        assert!(text.contains("\"brand\":\"niuse\""), "{text}");
        assert!(text.contains("persona"), "{text}");
        task.abort();
    }

    #[tokio::test]
    async fn new_session_creates_another_log() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::write(&path, "").unwrap();
        let state = Arc::new(State {
            path: Mutex::new(path.clone()),
            snapshot: Mutex::new(Vec::new()),
            provider: None,
            preset: "deepseek".into(),
            model: "deepseek-flash".into(),
            cwd: directory.path().to_path_buf(),
            max_steps: 1,
            system_prompt: None,
            context: br#"{}"#.to_vec(),
            context_provider: None,
            brand: "niucode".into(),
            token: None,
            allow_new: true,
            allowed_tools: Mutex::new(HashSet::new()),
            new_session_provider: None,
            switch_session_provider: None,
            computer: None,
            busy: AtomicBool::new(false),
            live: broadcast::channel(8).0,
            pending: Mutex::new(HashMap::new()),
            next_approval: AtomicU64::new(1),
        });
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state.clone()));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"POST /api/new HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 200"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        let new_path = state.path.lock().unwrap().clone();
        assert_ne!(new_path, path);
        assert!(new_path.exists());
        task.abort();
    }

    #[tokio::test]
    async fn live_context_provider_is_called_per_request() {
        use std::sync::atomic::AtomicUsize;
        let mut state = state_with(Vec::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        Arc::get_mut(&mut state).unwrap().context_provider = Some(Arc::new(move || {
            let n = count.fetch_add(1, Ordering::SeqCst) + 1;
            json!({"brand": format!("live-{n}")})
        }));
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state));
        for expected in ["live-1", "live-2"] {
            let mut stream = TcpStream::connect(address).await.unwrap();
            stream
                .write_all(b"GET /api/context HTTP/1.1\r\nHost: localhost\r\n\r\n")
                .await
                .unwrap();
            let mut reply = Vec::new();
            stream.read_to_end(&mut reply).await.unwrap();
            let text = String::from_utf8_lossy(&reply);
            assert!(text.contains(expected), "{text}");
        }
        task.abort();
    }

    #[tokio::test]
    async fn exports_markdown_with_an_attachment_header() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::write(
            &path,
            "{\"type\":\"user_message\",\"text\":\"hello export\",\"ts\":0}\n",
        )
        .unwrap();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state_at(path, true)));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /api/export HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 200"), "{text}");
        assert!(text.contains("text/markdown"), "{text}");
        assert!(
            text.contains("attachment; filename=\"niucode-session.md\""),
            "{text}"
        );
        assert!(text.contains("# niucode session"), "{text}");
        assert!(text.contains("hello export"), "{text}");
        task.abort();
    }

    #[tokio::test]
    async fn switch_session_uses_the_embedder_provider() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("chapter.jsonl");
        std::fs::write(
            &target,
            "{\"type\":\"user_message\",\"text\":\"chapter\",\"ts\":0}\n",
        )
        .unwrap();
        let mut state = state_at(directory.path().join("active.jsonl"), true);
        let target_for_provider = target.clone();
        Arc::get_mut(&mut state).unwrap().switch_session_provider = Some(Arc::new(move |id| {
            if id == "chapter" {
                Ok(target_for_provider.clone())
            } else {
                anyhow::bail!("unknown thread")
            }
        }));
        let state_for_assert = state.clone();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state));
        let mut events = TcpStream::connect(address).await.unwrap();
        events
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut headers = [0_u8; 512];
        let _ = events.read(&mut headers).await.unwrap();
        let mut stream = TcpStream::connect(address).await.unwrap();
        let body = br#"{"id":"chapter"}"#;
        stream
            .write_all(
                format!(
                    "POST /api/switch HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.write_all(body).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 200"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        assert_eq!(*state_for_assert.path.lock().unwrap(), target);
        let mut seen = Vec::new();
        let switched = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let mut chunk = [0_u8; 1024];
                let count = events.read(&mut chunk).await.unwrap();
                seen.extend_from_slice(&chunk[..count]);
                if String::from_utf8_lossy(&seen).contains("session_switched") {
                    return;
                }
            }
        })
        .await;
        switched.expect("session_switched event");
        task.abort();
    }

    #[tokio::test]
    async fn new_session_can_be_disabled() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(
            listener,
            state_at(PathBuf::from("unused.jsonl"), false),
        ));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"POST /api/new HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 403"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        task.abort();
    }

    struct WebComputer;
    impl niucode_computer::ComputerBackend for WebComputer {
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

    #[tokio::test]
    async fn configured_computer_can_be_stopped() {
        let mut runtime = niucode_agent::ComputerRuntime::new(
            Box::new(WebComputer),
            niucode_computer::ComputerPolicy::Control,
        );
        runtime.arm().expect("arm");
        let state = state_with_computer(Arc::new(Mutex::new(runtime)));
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state.clone()));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                b"POST /api/computer HTTP/1.1\r\nHost: localhost\r\nContent-Length: 17\r\n\r\n{\"action\":\"stop\"}",
            )
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 200"), "{text}");
        assert!(text.contains("\"armed\":false"), "{text}");
        assert!(text.contains("\"policy\":\"off\""), "{text}");
        let runtime = state.computer.as_ref().unwrap().lock().unwrap();
        assert_eq!(runtime.policy(), niucode_computer::ComputerPolicy::Off);
        assert!(!runtime.is_armed());
        task.abort();
    }

    #[tokio::test]
    async fn computer_control_requires_a_configured_runtime() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state()));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(
                b"POST /api/computer HTTP/1.1\r\nHost: localhost\r\nContent-Length: 17\r\n\r\n{\"action\":\"stop\"}",
            )
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 409"), "{text}");
        assert!(text.contains("not configured"), "{text}");
        task.abort();
    }

    #[tokio::test]
    async fn serves_model_and_policy_status() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state()));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        let text = String::from_utf8_lossy(&reply);
        assert!(text.starts_with("HTTP/1.1 200"), "{text}");
        assert!(text.contains("\"model\":\"deepseek-flash\""), "{text}");
        assert!(text.contains("\"sandbox\":\"workspace-write\""), "{text}");
        assert!(text.contains("\"approval\":\"ask\""), "{text}");
        assert!(
            text.contains("\"computer\":{\"armed\":false,\"configured\":false,\"policy\":\"off\"}"),
            "{text}"
        );
        task.abort();
    }

    #[tokio::test]
    async fn sse_streams_the_initial_log_and_new_events() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::write(&path, "{\"type\":\"session_start\",\"id\":\"one\"}\n").unwrap();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(
            listener,
            Arc::new(State {
                path: Mutex::new(path.clone()),
                snapshot: Mutex::new(Vec::new()),
                provider: None,
                preset: "deepseek".into(),
                model: "deepseek-flash".into(),
                cwd: PathBuf::from("."),
                max_steps: 1,
                system_prompt: None,
                context: br#"{}"#.to_vec(),
                context_provider: None,
                brand: "niucode".into(),
                token: None,
                allow_new: true,
                allowed_tools: Mutex::new(HashSet::new()),
                new_session_provider: None,
                switch_session_provider: None,
                computer: None,
                busy: AtomicBool::new(false),
                live: broadcast::channel(8).0,
                pending: Mutex::new(HashMap::new()),
                next_approval: AtomicU64::new(1),
            }),
        ));
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut seen = Vec::new();
        let first = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let mut chunk = [0_u8; 1024];
                let count = stream.read(&mut chunk).await.unwrap();
                seen.extend_from_slice(&chunk[..count]);
                if String::from_utf8_lossy(&seen).contains("session_start") {
                    return;
                }
            }
        })
        .await;
        first.expect("initial SSE event");
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"type\":\"assistant_message\",\"text\":\"two\"}\n")
            .unwrap();
        let second = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let mut chunk = [0_u8; 1024];
                let count = stream.read(&mut chunk).await.unwrap();
                seen.extend_from_slice(&chunk[..count]);
                if String::from_utf8_lossy(&seen).contains("assistant_message") {
                    return;
                }
            }
        })
        .await;
        second.expect("appended SSE event");
        task.abort();
    }

    #[tokio::test]
    async fn browser_prompt_runs_a_read_only_turn() {
        use niucode_provider::{ScriptedProvider, StreamEvent};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::write(&path, "").unwrap();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::from_turns(vec![vec![
            StreamEvent::TextDelta("from the browser".into()),
            StreamEvent::Done {
                finish_reason: Some("stop".into()),
            },
        ]]));
        let task = tokio::spawn(serve_listener(
            listener,
            Arc::new(State {
                path: Mutex::new(path.clone()),
                snapshot: Mutex::new(Vec::new()),
                provider: Some(provider),
                preset: "scripted".into(),
                model: "scripted".into(),
                cwd: directory.path().to_path_buf(),
                max_steps: 1,
                system_prompt: None,
                context: br#"{}"#.to_vec(),
                context_provider: None,
                brand: "niucode".into(),
                token: None,
                allow_new: true,
                allowed_tools: Mutex::new(HashSet::new()),
                new_session_provider: None,
                switch_session_provider: None,
                computer: None,
                busy: AtomicBool::new(false),
                live: broadcast::channel(8).0,
                pending: Mutex::new(HashMap::new()),
                next_approval: AtomicU64::new(1),
            }),
        ));
        let mut events = TcpStream::connect(address).await.unwrap();
        events
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut headers = [0_u8; 512];
        let _ = events.read(&mut headers).await.unwrap();
        let mut stream = TcpStream::connect(address).await.unwrap();
        let body = br#"{"prompt":"say hello"}"#;
        stream
            .write_all(
                format!(
                    "POST /api/prompt HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.write_all(body).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 202"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        let mut live = Vec::new();
        let delta = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let mut chunk = [0_u8; 1024];
                let count = events.read(&mut chunk).await.unwrap();
                live.extend_from_slice(&chunk[..count]);
                if String::from_utf8_lossy(&live).contains("text_delta") {
                    return;
                }
            }
        })
        .await;
        delta.expect("text delta over SSE");
        assert!(String::from_utf8_lossy(&live).contains("\"type\":\"phase\""));
        assert!(String::from_utf8_lossy(&live).contains("\"html\":"));
        let changed = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if String::from_utf8_lossy(&live).contains("context_changed") {
                    return;
                }
                let mut chunk = [0_u8; 1024];
                let count = events.read(&mut chunk).await.unwrap();
                live.extend_from_slice(&chunk[..count]);
            }
        })
        .await;
        changed.expect("context_changed after turn");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            if text.contains("from the browser") {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "turn did not finish");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        task.abort();
    }

    #[tokio::test]
    async fn approval_post_releases_a_pending_turn() {
        let state = state();
        let state_for_assert = state.clone();
        let (sender, receiver) = mpsc::channel();
        state.pending.lock().unwrap().insert(
            "a-1".into(),
            PendingApproval {
                tool: "write".into(),
                sender,
            },
        );
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(listener, state));
        let mut stream = TcpStream::connect(address).await.unwrap();
        let body = br#"{"id":"a-1","allowed":true,"session":true}"#;
        stream
            .write_all(
                format!(
                    "POST /api/approval HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.write_all(body).await.unwrap();
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 200"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        assert!(receiver.recv_timeout(Duration::from_secs(1)).unwrap());
        assert!(state_for_assert
            .allowed_tools
            .lock()
            .unwrap()
            .contains("write"));
        task.abort();
    }

    #[tokio::test]
    async fn browser_approval_allows_a_write_turn() {
        use niucode_provider::{ScriptedProvider, StreamEvent, ToolCallDelta};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.jsonl");
        std::fs::write(&path, "").unwrap();
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::from_turns(vec![
            vec![
                StreamEvent::ToolCallDelta(ToolCallDelta {
                    index: 0,
                    id: Some("write-1".into()),
                    name: Some("write".into()),
                    arguments: serde_json::json!({"path":"approved.txt","content":"yes"})
                        .to_string(),
                }),
                StreamEvent::Done {
                    finish_reason: Some("tool_calls".into()),
                },
            ],
            vec![
                StreamEvent::TextDelta("written".into()),
                StreamEvent::Done {
                    finish_reason: Some("stop".into()),
                },
            ],
        ]));
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(serve_listener(
            listener,
            Arc::new(State {
                path: Mutex::new(path.clone()),
                snapshot: Mutex::new(Vec::new()),
                provider: Some(provider),
                preset: "scripted".into(),
                model: "scripted".into(),
                cwd: directory.path().to_path_buf(),
                max_steps: 2,
                system_prompt: None,
                context: br#"{}"#.to_vec(),
                context_provider: None,
                brand: "niucode".into(),
                token: None,
                allow_new: true,
                allowed_tools: Mutex::new(HashSet::new()),
                new_session_provider: None,
                switch_session_provider: None,
                computer: None,
                busy: AtomicBool::new(false),
                live: broadcast::channel(8).0,
                pending: Mutex::new(HashMap::new()),
                next_approval: AtomicU64::new(1),
            }),
        ));
        let mut events = TcpStream::connect(address).await.unwrap();
        events
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut headers = [0_u8; 512];
        let _ = events.read(&mut headers).await.unwrap();
        let mut prompt = TcpStream::connect(address).await.unwrap();
        let body = br#"{"prompt":"write it"}"#;
        prompt.write_all(format!("POST /api/prompt HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes()).await.unwrap();
        prompt.write_all(body).await.unwrap();
        let mut reply = Vec::new();
        prompt.read_to_end(&mut reply).await.unwrap();
        assert!(String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 202"));
        let mut seen = Vec::new();
        let approval = "a-1".to_string();
        let seen_approval = tokio::time::timeout(Duration::from_secs(4), async {
            loop {
                let mut chunk = [0_u8; 1024];
                let count = events.read(&mut chunk).await.unwrap();
                seen.extend_from_slice(&chunk[..count]);
                if String::from_utf8_lossy(&seen).contains("\"type\":\"approval\"") {
                    return;
                }
            }
        })
        .await;
        seen_approval.expect("approval event");
        let mut approve = TcpStream::connect(address).await.unwrap();
        let body = format!(r#"{{"id":"{approval}","allowed":true}}"#);
        approve.write_all(format!("POST /api/approval HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes()).await.unwrap();
        approve.write_all(body.as_bytes()).await.unwrap();
        let mut reply = Vec::new();
        approve.read_to_end(&mut reply).await.unwrap();
        assert!(
            String::from_utf8_lossy(&reply).starts_with("HTTP/1.1 200"),
            "{}",
            String::from_utf8_lossy(&reply)
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if std::fs::read_to_string(directory.path().join("approved.txt"))
                .ok()
                .as_deref()
                == Some("yes")
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "approved write missing"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        task.abort();
    }

    #[test]
    fn index_dom_ids_are_unique_and_referenced_ids_exist() {
        use std::collections::HashSet;
        let mut ids = HashSet::new();
        let mut rest = INDEX;
        while let Some(index) = rest.find("id=\"") {
            rest = &rest[index + 4..];
            if let Some(end) = rest.find('"') {
                assert!(
                    ids.insert(rest[..end].to_string()),
                    "duplicate id {}",
                    &rest[..end]
                );
                rest = &rest[end..];
            }
        }
        let mut refs = HashSet::new();
        let mut rest = INDEX;
        while let Some(index) = rest.find("getElementById('") {
            rest = &rest[index + 16..];
            if let Some(end) = rest.find('\'') {
                refs.insert(rest[..end].to_string());
                rest = &rest[end..];
            }
        }
        let missing = refs.difference(&ids).collect::<Vec<_>>();
        assert!(missing.is_empty(), "missing ids: {missing:?}");
    }

    #[test]
    fn index_uses_sse_and_has_no_remote_assets() {
        assert!(INDEX.contains("EventSource('/api/events')"));
        assert!(INDEX.contains("Enter to send"));
        assert!(INDEX.contains("fetch('/api/prompt'"));
        assert!(INDEX.contains("text_delta"));
        assert!(INDEX.contains("renderApproval"));
        assert!(INDEX.contains("approval-shot"));
        assert!(INDEX.contains("renderPreview"));
        assert!(INDEX.contains("renderToolCall"));
        assert!(INDEX.contains("tool-card"));
        assert!(INDEX.contains("textarea"));
        assert!(INDEX.contains("Shift+Enter"));
        assert!(INDEX.contains("prompt.history"));
        assert!(INDEX.contains("copy-code"));
        assert!(INDEX.contains("context_changed"));
        assert!(INDEX.contains("'/help'"));
        assert!(INDEX.contains("unknown command"));
        assert!(INDEX.contains("cmd-menu"));
        assert!(INDEX.contains("session log unchanged"));
        assert!(INDEX.contains("usageTotals"));
        assert!(INDEX.contains("/api/export"));
        assert!(INDEX.contains("filterTranscript"));
        assert!(INDEX.contains("scrollToLatest"));
        assert!(INDEX.contains("prefers-reduced-motion"));
        assert!(INDEX.contains("niucode.theme"));
        assert!(INDEX.contains("body.contrast"));
        assert!(INDEX.contains("role=\"log\""));
        assert!(INDEX.contains("prefers-contrast: more"));
        assert!(INDEX.contains("button.chip"));
        assert!(INDEX.contains("aside.open"));
        assert!(INDEX.contains("ctx-backdrop"));
        assert!(INDEX.contains("trapContextFocus"));
        assert!(INDEX.contains("pre.copyable"));
        assert!(INDEX.contains("api/attach"));
        assert!(INDEX.contains("pendingAttachments"));
        assert!(INDEX.contains("addMessageCopy"));
        assert!(INDEX.contains("streamRendered"));
        assert!(INDEX.contains("ctrlKey"));
        assert!(INDEX.contains("getComputedStyle"));
        assert!(INDEX.contains("/new"));
        assert!(INDEX.contains("/threads"));
        assert!(INDEX.contains("/export"));
        assert!(INDEX.contains("Allow tool"));
        assert!(INDEX.contains("expires in"));
        assert!(INDEX.contains("switchThread"));
        assert!(INDEX.contains("session_switched"));
        assert!(INDEX.contains("diff-line"));
        assert!(INDEX.contains("e.html"));
        assert!(INDEX.contains("math"));
        assert!(INDEX.contains("tok-keyword"));
        assert!(INDEX.contains("/api/context"));
        assert!(INDEX.contains("/api/status"));
        assert!(INDEX.contains("id=\"computer\""));
        assert!(INDEX.contains("beginTurn"));
        assert!(INDEX.contains("toggle-context"));
        assert!(INDEX.contains("ctx-filter"));
        assert!(INDEX.contains("ctx-section"));
        assert!(INDEX.contains("loadContext"));
        assert!(INDEX.contains("localStorage"));
        assert!(INDEX.contains("textarea"));
        assert!(INDEX.contains("Shift+Enter"));
        assert!(INDEX.contains("prompt.history"));
        assert!(INDEX.contains("copy-code"));
        assert!(INDEX.contains("context_changed"));
        assert!(INDEX.contains("'/help'"));
        assert!(INDEX.contains("unknown command"));
        assert!(INDEX.contains("cmd-menu"));
        assert!(INDEX.contains("session log unchanged"));
        assert!(INDEX.contains("usageTotals"));
        assert!(INDEX.contains("/api/export"));
        assert!(INDEX.contains("filterTranscript"));
        assert!(INDEX.contains("scrollToLatest"));
        assert!(INDEX.contains("prefers-reduced-motion"));
        assert!(INDEX.contains("niucode.theme"));
        assert!(INDEX.contains("body.contrast"));
        assert!(INDEX.contains("role=\"log\""));
        assert!(INDEX.contains("prefers-contrast: more"));
        assert!(INDEX.contains("button.chip"));
        assert!(INDEX.contains("aside.open"));
        assert!(INDEX.contains("ctx-backdrop"));
        assert!(INDEX.contains("trapContextFocus"));
        assert!(INDEX.contains("pre.copyable"));
        assert!(INDEX.contains("api/attach"));
        assert!(INDEX.contains("pendingAttachments"));
        assert!(INDEX.contains("addMessageCopy"));
        assert!(INDEX.contains("streamRendered"));
        assert!(INDEX.contains("ctrlKey"));
        assert!(INDEX.contains("getComputedStyle"));
        assert!(INDEX.contains("/new"));
        assert!(INDEX.contains("/threads"));
        assert!(INDEX.contains("/export"));
        assert!(INDEX.contains("Allow tool"));
        assert!(INDEX.contains("expires in"));
        assert!(INDEX.contains("switchThread"));
        assert!(INDEX.contains("session_switched"));
        assert!(INDEX.contains("stepTranscriptMatch"));
        assert!(INDEX.contains("id=\"log-prev\""));
        assert!(INDEX.contains("id=\"log-next\""));
        assert!(INDEX.contains("% cache"));
        assert!(!INDEX.contains("http://"));
        assert!(!INDEX.contains("https://"));
    }
}
