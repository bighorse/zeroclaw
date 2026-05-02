//! SOP enforcement hook.
//!
//! Closes a critical gap in the `SopEngine` design: `sop_execute` is
//! a *call*, not a *control flow*. Once an LLM has called it, the
//! engine records an `active_runs[run_id] = SopRun{status: Running}`
//! entry, but nothing prevents the LLM from then issuing any tool
//! call it likes — including `file_write` straight to the final
//! deliverable path. Per-step quality gates (`requires_confirmation`,
//! prompt-side P0 checks in `notes`) only fire when the LLM actually
//! routes through `sop_advance`. If the LLM skips that, every gate
//! is silently bypassed.
//!
//! This was observed live on PharmaClaw v3 (`run-1777677621534-0001`):
//!
//!   23:20:21  sop_execute (run starts at step 1, status=Running)
//!   23:21:58  file_write → reports/proposal_part1.md   ← bypassed
//!   23:22:04  shell pandoc                              ← (failed but tried)
//!   23:23:50  file_write → reports/proposal_part2.md   ← bypassed
//!   23:23:55  sop_advance (called once, after report done)
//!
//! The "report" was assembled directly from yesterday's leftover
//! `case_library/insight/zhangli_sysucc/` artifacts; PI verification,
//! prior-art retrieval, weakness rebuttal, PM review — none of them
//! ran. The LLM chose the locally optimal "stitch and ship" path.
//!
//! ## What this hook enforces
//!
//! When **any** SOP run is in `active_runs` (status `Pending`,
//! `Running`, `WaitingApproval`, or `PausedCheckpoint`), every
//! `file_write` and `file_edit` call is checked: the `path` argument
//! must start with one of the configured allowlist prefixes (default:
//! `case_library/`, `scripts/`, `state/`). Anything else is `Cancel`'d
//! with a message explaining why. The cancellation message tells the
//! LLM exactly which prefixes are acceptable, so it can self-correct
//! without an opaque tool failure.
//!
//! ## What this hook does NOT enforce
//!
//! - It does not require `sop_advance` to be called between writes.
//!   That would need per-step output-path inference and is brittle —
//!   the path-prefix gate alone closes the worst escape hatch
//!   (`reports/`, project root, arbitrary paths).
//! - It does not touch reads (`file_read`, `glob_search`, etc.) —
//!   the LLM still needs unhindered observability.
//! - It does not touch `shell`. That's a deliberate trade-off:
//!   parsing shell command lines for write targets is unreliable.
//!   Shell mutating use should be guarded by `[autonomy]
//!   allowed_commands` separately.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::config::schema::SopEnforcementConfig;
use crate::hooks::traits::{HookHandler, HookResult};
use crate::sop::{SopEngine, SopRunStatus};

/// Tools whose `path` argument is enforced. Reads and SOP control
/// tools are intentionally absent.
const ENFORCED_TOOLS: &[&str] = &["file_write", "file_edit"];

pub struct SopEnforcementHook {
    config: SopEnforcementConfig,
    engine: Arc<Mutex<SopEngine>>,
}

impl SopEnforcementHook {
    pub fn new(config: SopEnforcementConfig, engine: Arc<Mutex<SopEngine>>) -> Self {
        Self { config, engine }
    }

    /// Returns `Some((run_id, status))` summary if there is at least one
    /// SOP run in a live state (`Pending` / `Running` / `WaitingApproval`
    /// / `PausedCheckpoint`); `None` otherwise. We snapshot under the
    /// lock and drop it before doing any string work.
    fn snapshot_active_run(&self) -> Option<(String, String)> {
        let engine = self.engine.lock().ok()?;
        engine
            .active_runs()
            .values()
            .find(|r| {
                matches!(
                    r.status,
                    SopRunStatus::Pending
                        | SopRunStatus::Running
                        | SopRunStatus::WaitingApproval
                        | SopRunStatus::PausedCheckpoint
                )
            })
            .map(|r| (r.run_id.clone(), r.status.to_string()))
    }

    /// Test whether `path` is permitted under the configured allowlist.
    /// Both the path and prefixes are normalised by stripping leading
    /// `./` so the LLM's idiomatic `./foo` matches a `foo/` prefix.
    fn path_is_allowed(&self, path: &str) -> bool {
        let cleaned = path.trim_start_matches("./");
        self.config.allowed_path_prefixes.iter().any(|prefix| {
            let p = prefix.trim_start_matches("./");
            cleaned.starts_with(p)
        })
    }

    fn extract_path(args: &Value) -> Option<&str> {
        args.get("path").and_then(|v| v.as_str())
    }

    fn cancel_message(&self, tool: &str, path: &str, run_id: &str, status: &str) -> String {
        format!(
            "🚫 SOP enforcement: `{tool}` to `{path}` rejected.\n\n\
             An SOP run is active (`{run_id}`, status `{status}`) but the requested write \
             path is outside the SOP-managed file tree. While a SOP run is in progress, \
             mutating tools may only target paths under one of these prefixes: {prefixes}.\n\n\
             To proceed:\n\
             • If this write belongs to a SOP step, write under `case_library/insight/<pi_name>/` \
               (or `scripts/` for intermediate Python helpers), then call `sop_advance` to \
               record the step result.\n\
             • If this run should be abandoned (e.g. user changed their mind), call \
               `sop_advance` with `status=\"failed\"` to terminate the run, then this write \
               will be allowed.\n\
             • Never write the final report straight to `reports/` or project root — \
               the SOP step 8 (报告组装) is what produces that.",
            prefixes = self
                .config
                .allowed_path_prefixes
                .iter()
                .map(|p| format!("`{p}`"))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

#[async_trait]
impl HookHandler for SopEnforcementHook {
    fn name(&self) -> &str {
        "sop-enforcement"
    }

    fn priority(&self) -> i32 {
        // Run early so the rejection short-circuits before any other
        // mutating-tool hook (audit, metrics) processes the call.
        100
    }

    async fn before_tool_call(&self, name: String, args: Value) -> HookResult<(String, Value)> {
        if !self.config.enabled {
            return HookResult::Continue((name, args));
        }
        if !ENFORCED_TOOLS.contains(&name.as_str()) {
            return HookResult::Continue((name, args));
        }
        let Some(path) = Self::extract_path(&args) else {
            // Tool will fail on its own with a clearer "missing path" error.
            return HookResult::Continue((name, args));
        };
        let Some((run_id, status)) = self.snapshot_active_run() else {
            // No active SOP run → no enforcement.
            return HookResult::Continue((name, args));
        };
        if self.path_is_allowed(path) {
            return HookResult::Continue((name, args));
        }
        tracing::warn!(
            hook = "sop-enforcement",
            tool = %name,
            path = %path,
            run_id = %run_id,
            status = %status,
            "rejecting mutating tool call: path outside SOP allowlist"
        );
        HookResult::Cancel(self.cancel_message(&name, path, &run_id, &status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::SopConfig;
    use crate::sop::types::{SopEvent, SopTriggerSource};
    use crate::sop::{SopEngine, SopRunStatus};
    use serde_json::json;

    fn cfg(enabled: bool, prefixes: &[&str]) -> SopEnforcementConfig {
        SopEnforcementConfig {
            enabled,
            allowed_path_prefixes: prefixes.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn empty_engine() -> Arc<Mutex<SopEngine>> {
        Arc::new(Mutex::new(SopEngine::new(SopConfig::default())))
    }

    fn engine_with_active_run(status: SopRunStatus) -> Arc<Mutex<SopEngine>> {
        let engine = SopEngine::new(SopConfig::default());
        let arc = Arc::new(Mutex::new(engine));
        // Inject an active run by hand. We avoid going through start_run
        // because that would require loading a real SOP definition.
        {
            let mut eng = arc.lock().unwrap();
            let run = crate::sop::types::SopRun {
                run_id: "run-test-0001".to_string(),
                sop_name: "test-sop".to_string(),
                trigger_event: SopEvent {
                    source: SopTriggerSource::Manual,
                    topic: None,
                    payload: None,
                    timestamp: "2026-05-02T00:00:00Z".to_string(),
                },
                status,
                current_step: 1,
                total_steps: 11,
                started_at: "2026-05-02T00:00:00Z".to_string(),
                completed_at: None,
                step_results: vec![],
                waiting_since: None,
                llm_calls_saved: 0,
            };
            eng.active_runs_mut_for_test().insert(run.run_id.clone(), run);
        }
        arc
    }

    #[tokio::test]
    async fn disabled_hook_passes_through() {
        let hook = SopEnforcementHook::new(cfg(false, &["case_library/"]), empty_engine());
        let result = hook
            .before_tool_call(
                "file_write".into(),
                json!({"path": "anywhere.md", "content": "x"}),
            )
            .await;
        assert!(matches!(result, HookResult::Continue(_)));
    }

    #[tokio::test]
    async fn no_active_run_passes_through() {
        let hook =
            SopEnforcementHook::new(cfg(true, &["case_library/"]), empty_engine());
        let result = hook
            .before_tool_call(
                "file_write".into(),
                json!({"path": "reports/anywhere.md", "content": "x"}),
            )
            .await;
        assert!(
            matches!(result, HookResult::Continue(_)),
            "no active run → write should be allowed"
        );
    }

    #[tokio::test]
    async fn read_tools_are_never_enforced() {
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_active_run(SopRunStatus::Running),
        );
        for tool in ["file_read", "glob_search", "shell", "memory_recall"] {
            let result = hook
                .before_tool_call(tool.into(), json!({"path": "anywhere.md"}))
                .await;
            assert!(
                matches!(result, HookResult::Continue(_)),
                "{tool} must pass through enforcement"
            );
        }
    }

    #[tokio::test]
    async fn write_to_allowed_prefix_passes() {
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/", "scripts/"]),
            engine_with_active_run(SopRunStatus::Running),
        );
        for path in [
            "case_library/insight/zhangli/proposal.md",
            "scripts/pi_intel.py",
            "./case_library/insight/zhangli/budget.md", // leading ./ tolerated
        ] {
            let result = hook
                .before_tool_call(
                    "file_write".into(),
                    json!({"path": path, "content": "x"}),
                )
                .await;
            assert!(
                matches!(result, HookResult::Continue(_)),
                "{path} should be allowed"
            );
        }
    }

    #[tokio::test]
    async fn write_outside_prefix_is_cancelled_with_helpful_message() {
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/", "scripts/"]),
            engine_with_active_run(SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "file_write".into(),
                json!({"path": "reports/proposal.md", "content": "x"}),
            )
            .await;
        match result {
            HookResult::Cancel(msg) => {
                assert!(msg.contains("SOP enforcement"));
                assert!(msg.contains("reports/proposal.md"));
                assert!(msg.contains("run-test-0001"));
                assert!(msg.contains("`case_library/`"));
                assert!(msg.contains("sop_advance"));
            }
            HookResult::Continue(_) => panic!("expected Cancel"),
        }
    }

    #[tokio::test]
    async fn waiting_approval_state_still_enforced() {
        // Critical: PR #17's WaitingApproval guard means sop_advance
        // returns Err during this state. The LLM's natural retry path
        // is to switch to file_write — exactly what we must keep
        // gated.
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_active_run(SopRunStatus::WaitingApproval),
        );
        let result = hook
            .before_tool_call(
                "file_write".into(),
                json!({"path": "reports/proposal.md", "content": "x"}),
            )
            .await;
        assert!(matches!(result, HookResult::Cancel(_)));
    }

    #[tokio::test]
    async fn missing_path_arg_passes_through_for_clearer_error() {
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_active_run(SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call("file_write".into(), json!({"content": "no path field"}))
            .await;
        // We let the tool itself produce the "missing path" error
        // rather than masking it with an enforcement message.
        assert!(matches!(result, HookResult::Continue(_)));
    }
}
