//! Task facts to compact transcript and full evidence. No terminal or log writes.
use crate::text_layout::{clean_text, clip};
use niucode_agent::TaskReview;
use niucode_core::ProcessOutcome;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

pub fn startup(
    project: &str,
    model: &str,
    sandbox: &str,
    network: &str,
    approval: &str,
    width: usize,
) -> Vec<String> {
    vec![
        clip(
            &format!(
                "niucode tui · {} · {}",
                clip(&clean_text(project), 20),
                clip(&clean_text(model), 32)
            ),
            width,
        ),
        clip(
            &format!("{sandbox} · net {network} · {approval} · /help"),
            width,
        ),
    ]
}

pub fn completion(review: &TaskReview, width: usize) -> String {
    let failed = review
        .checks
        .iter()
        .filter(|c| !matches!(c.outcome, ProcessOutcome::Exited { code: 0 }))
        .count();
    let warnings = failed + review.tool_failures + review.denied + review.conflicts.len();
    let label = match review.status.as_deref() {
        Some("complete") if warnings == 0 => "✓ done",
        Some("complete") => "! finished with issues",
        Some("cancelled") => "! cancelled",
        Some("error") => "✗ failed",
        _ => "! interrupted",
    };
    let mut lines = vec![format!(
        "{label} · {:.1}s",
        review.elapsed_ms as f64 / 1000.0
    )];
    let relevant = !review.changed.is_empty()
        || review.writes > 0
        || !review.checks.is_empty()
        || warnings > 0;
    if relevant {
        let checks = if review.checks.is_empty() {
            "no checks".into()
        } else {
            format!("{} commands, {failed} failed", review.checks.len())
        };
        lines.push(if review.changed.is_empty() && review.writes > 0 {
            format!("{} recorded writes · {checks}", review.writes)
        } else {
            format!("{} confirmed changes · {checks}", review.changed.len())
        });
        let note = if !review.conflicts.is_empty() {
            "conflicts"
        } else if review.denied > 0 {
            "approval denied"
        } else if !review.unknown.is_empty() {
            "partial coverage"
        } else if review.checks.iter().any(|c| !c.fresh) {
            "checks may be stale"
        } else {
            "evidence"
        };
        lines.push(format!("{note} · Ctrl-O details"));
    }
    lines
        .iter()
        .map(|s| clip(s, width))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn evidence(review: &TaskReview) -> String {
    let mut body = format!(
        "# Task {}\nStatus: {} · {:.1}s\ncwd: {}\n\n",
        review.id,
        review.status.as_deref().unwrap_or("interrupted"),
        review.elapsed_ms as f64 / 1000.0,
        review.cwd
    );
    for file in &review.changed {
        body.push_str(&format!(
            "## {} · {}\n{}\n",
            file.path,
            if file.attributed {
                "recorded file tool"
            } else {
                "workspace change · origin unknown"
            },
            file.diff.as_deref().unwrap_or(
                "Full text diff unavailable (binary, missing or outside captured coverage)."
            )
        ));
    }
    body.push_str("# Command evidence\n");
    if review.checks.is_empty() {
        body.push_str("No recorded command checks.\n");
    }
    for check in &review.checks {
        let result = match check.outcome {
            ProcessOutcome::Exited { code } => format!("exit {code}"),
            ProcessOutcome::Signaled { signal } => format!("signal {signal:?}"),
            ProcessOutcome::TimedOut => "timed out".into(),
        };
        body.push_str(&format!(
            "{} · {result} · {}\n$ {}\n",
            check.call_id,
            if check.fresh {
                "captured workspace unchanged since command"
            } else {
                "final workspace not verified / result may be stale"
            },
            check.command
        ));
    }
    body.push_str(
        "Command exit status is evidence for that command, not proof of all acceptance criteria.\n",
    );
    if !review.conflicts.is_empty() {
        body.push_str(&format!("\n# Conflicts\n{}\n", review.conflicts.join("\n")));
    }
    body.push_str(&format!(
        "\nTool failures: {} · denied approvals: {}\n",
        review.tool_failures, review.denied
    ));
    if !review.unknown.is_empty() {
        body.push_str(&format!(
            "\n# Coverage unknown\n{}\n",
            review.unknown.join("\n")
        ));
    }
    body.push_str(
        "\nMissing evidence is not proof of no changes. No automatic retry of uncertain writes.\n",
    );
    body
}

/// One compact line for a tool call. Never includes file contents or outputs.
pub fn tool_call(name: &str, input: &Value, width: usize) -> String {
    let detail = tool_detail(input);
    let line = match detail {
        Some(detail) => format!("→ {name} · {detail}"),
        None => format!("→ {name}"),
    };
    clip(&clean_text(&line), width)
}

/// One compact line for a finished tool; failures stay explicit.
pub fn tool_result(name: &str, ok: bool, summary: &str, width: usize) -> String {
    let mark = if ok { "✓" } else { "✗" };
    let summary = clip(&clean_text(summary), width.saturating_sub(name.len() + 4));
    if summary.is_empty() {
        clip(&format!("{mark} {name}"), width)
    } else {
        clip(&format!("{mark} {name} · {summary}"), width)
    }
}

/// Extract the last meaningful non-empty error line from a failed tool's output
/// when it adds detail beyond `summary`.
pub fn tool_error_tail(summary: &str, output: &str, width: usize) -> Option<String> {
    let summary_clean = clean_text(summary);
    let tail = output.lines().map(str::trim).rfind(|line| {
        !line.is_empty()
            && *line != "stdout:"
            && *line != "stderr:"
            && clean_text(line) != summary_clean
    })?;
    let formatted = format!("  ↳ {}", clean_text(tail));
    Some(clip(&formatted, width))
}

fn format_tool_duration(elapsed: Duration) -> String {
    let ms = elapsed.as_millis();
    if ms > 0 && ms < 100 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", elapsed.as_secs_f64())
    }
}

fn tool_detail(input: &Value) -> Option<String> {
    let object = input.as_object()?;
    for key in [
        "path", "command", "query", "pattern", "url", "prompt", "goal",
    ] {
        if let Some(value) = object.get(key).and_then(Value::as_str) {
            let value = clean_text(value);
            if !value.is_empty() {
                return Some(clip(&value, 72));
            }
        }
    }
    None
}

/// Aggregate per-tool durations into one bounded, deterministic line.
pub fn tool_timings(times: &HashMap<String, Duration>, width: usize) -> Option<String> {
    if times.is_empty() {
        return None;
    }
    let mut entries: Vec<_> = times.iter().collect();
    entries.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));
    let parts = entries
        .into_iter()
        .take(6)
        .map(|(name, elapsed)| format!("{name} {}", format_tool_duration(*elapsed)))
        .collect::<Vec<_>>();
    Some(clip(
        &clean_text(&format!("tools · {}", parts.join(" · "))),
        width,
    ))
}

/// Motion can be disabled without changing the status text.
pub fn animation_enabled() -> bool {
    std::env::var_os("NIUCODE_REDUCED_MOTION").is_none() && std::env::var_os("NO_MOTION").is_none()
}

pub fn activity(
    state: &str,
    tools: usize,
    elapsed_ms: u128,
    animate: bool,
    width: usize,
    streaming: bool,
) -> String {
    const FRAMES: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
    let frame = if animate {
        FRAMES[(elapsed_ms / 125 % FRAMES.len() as u128) as usize]
    } else {
        "·"
    };
    let state = clip(&clean_text(state), width.saturating_sub(34).max(8));
    let caret = if streaming {
        if animate && (elapsed_ms / 250).is_multiple_of(2) {
            " ▍"
        } else {
            " ▏"
        }
    } else {
        ""
    };
    clip(
        &format!(
            "{frame} {state}{caret} · {}s · {tools} tools · Ctrl-C cancel",
            elapsed_ms / 1000
        ),
        width,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use niucode_agent::ReviewedFile;
    #[test]
    fn quiet_chat_and_large_reviews_have_bounded_transcripts() {
        let mut review = TaskReview {
            status: Some("complete".into()),
            unknown: vec!["unobserved".into(); 130],
            ..Default::default()
        };
        assert_eq!(completion(&review, 80), "✓ done · 0.0s");
        for count in [1, 100, 1000] {
            review.changed = (0..count)
                .map(|_| ReviewedFile {
                    path: "long路径".repeat(100),
                    attributed: false,
                    diff: None,
                })
                .collect();
            for width in [20, 40, 80, 120] {
                let text = completion(&review, width);
                assert!(crate::text_layout::wrap(&text, width).len() <= 3);
                assert!(!text.contains("long"));
            }
        }
        assert!(evidence(&review).contains(&review.changed[0].path));
    }
    #[test]
    fn long_commands_remain_in_details_and_never_expand_completion() {
        let command = "cargo test ".repeat(5000);
        let review = TaskReview {
            status: Some("complete".into()),
            checks: vec![niucode_agent::CommandCheck {
                call_id: "check".into(),
                command: command.clone(),
                outcome: ProcessOutcome::Exited { code: 1 },
                fresh: false,
            }],
            ..Default::default()
        };
        for width in [20, 40, 80, 120] {
            let summary = completion(&review, width);
            assert!(crate::text_layout::wrap(&summary, width).len() <= 3);
            assert!(!summary.contains("✓"));
            assert!(!summary.contains("cargo"));
        }
        assert!(evidence(&review).contains(&command));
    }

    #[test]
    fn failures_denials_and_truncation_never_present_success() {
        let mut review = TaskReview {
            status: Some("complete".into()),
            denied: 1,
            ..Default::default()
        };
        assert!(!completion(&review, 80).contains("✓"));
        review.denied = 0;
        for status in ["cancelled", "error", "unknown"] {
            review.status = Some(status.into());
            assert!(!completion(&review, 80).contains("✓"));
        }
    }
    #[test]
    fn animation_is_fixed_width_and_has_a_static_fallback() {
        assert_ne!(
            activity("thinking", 0, 0, true, 80, false),
            activity("thinking", 0, 125, true, 80, false)
        );
        assert_eq!(
            activity("thinking", 0, 0, false, 80, false),
            activity("thinking", 0, 125, false, 80, false)
        );
        let first = activity("responding", 0, 0, true, 80, true);
        let second = activity("responding", 0, 250, true, 80, true);
        assert!(first.contains('▍') || first.contains('▏'), "{first}");
        assert_ne!(first, second);
        assert!(!activity("responding", 0, 0, false, 80, false).contains('▍'));
        assert!(
            animation_enabled()
                == (std::env::var_os("NIUCODE_REDUCED_MOTION").is_none()
                    && std::env::var_os("NO_MOTION").is_none())
        );
    }

    #[test]
    fn tool_timings_are_aggregated_and_bounded() {
        let mut times = HashMap::new();
        times.insert("bash".to_string(), Duration::from_millis(1200));
        times.insert("write".to_string(), Duration::from_millis(400));
        times.insert("read".to_string(), Duration::from_millis(25));
        let line = tool_timings(&times, 80).unwrap();
        assert!(line.starts_with("tools · bash 1.2s"), "{line}");
        assert!(line.contains("write 0.4s"), "{line}");
        assert!(line.contains("read 25ms"), "{line}");
        assert!(tool_timings(&HashMap::new(), 80).is_none());
        assert!(tool_timings(&times, 18).unwrap().chars().count() <= 18);
    }

    #[test]
    fn tool_error_tail_picks_last_meaningful_line() {
        assert_eq!(
            tool_error_tail(
                "exit: 1",
                "exit: 1\nstdout:\ncompiling...\nstderr:\nerror[E0425]: cannot find value `x`\n",
                80
            ),
            Some("  ↳ error[E0425]: cannot find value `x`".to_string())
        );
        assert_eq!(tool_error_tail("exit: 1", "exit: 1\n", 80), None);
    }

    #[test]
    fn tool_lines_stay_compact_and_name_the_target() {
        use serde_json::json;
        assert_eq!(
            tool_call(
                "write",
                &json!({"path": "src/main.rs", "content": "secret"}),
                80
            ),
            "→ write · src/main.rs"
        );
        assert_eq!(
            tool_call("bash", &json!({"command": "cargo test"}), 80),
            "→ bash · cargo test"
        );
        assert_eq!(tool_call("read", &json!({}), 80), "→ read");
        assert_eq!(
            tool_result("write", true, "2 files", 80),
            "✓ write · 2 files"
        );
        assert_eq!(tool_result("bash", false, "exit 1", 80), "✗ bash · exit 1");
    }
}
