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
//!
//! ## Step 8 size gate (PharmaClaw V4-V11 recurrence)
//!
//! Closes a recurring V4/V7/V8/V9/V10/V11 工程 bug observed across
//! 6 PharmaClaw runs: at SOP step 8 (报告组装 / report assembly), the
//! LLM treats the chapter files (`pi_intel.md` / `disease_scan.md` /
//! `hypotheses.md` / `technical_route.md` / `budget.md` /
//! `output_matching.md` / `pm_review.md`, ~67KB combined) as "report
//! is assembled", and produces only a 4-5KB index `proposal.md` that
//! points at the chapter files instead of merging them. The `notes`
//! prompt for step 8 already says "must be ≥20KB" but it's a prompt
//! warning that the LLM routinely skips.
//!
//! The hook intercepts `sop_advance status=completed` while the
//! active run's `current_step == 8` and `sop_name` matches a
//! configured deliverable-style SOP (default: anything in
//! `step8_size_gate_sop_names`). It globs the latest `*proposal.md`
//! under `case_library/insight/*/`, checks `len() >= 20480`, and
//! cancels the advance with a self-correction message if not.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::config::schema::SopEnforcementConfig;
use crate::hooks::traits::{HookHandler, HookResult};
use crate::sop::{SopEngine, SopRunStatus};

/// Tools whose `path` argument is enforced. Reads and SOP control
/// tools are intentionally absent.
const ENFORCED_TOOLS: &[&str] = &["file_write", "file_edit"];

/// Tool whose `status=completed` triggers the step 8 size gate.
const STEP8_GATE_TOOL: &str = "sop_advance";

/// Minimum acceptable size for a step 8 deliverable proposal.md.
/// Typical full立项 reports are 25-35KB after merging chapter files.
const STEP8_PROPOSAL_MIN_BYTES: u64 = 20 * 1024;

/// Glob pattern for step 8 deliverable proposal.md files. Matches
/// PharmaClaw `jl-insight-research-proposal` convention:
/// `case_library/insight/<pi_name>/<date>_proposal.md` and variants
/// like `proposal.md` / `final_proposal.md`.
const STEP8_PROPOSAL_GLOB: &str = "case_library/insight/*/*proposal.md";

/// SOP names that produce a deliverable proposal.md at step 8.
/// Other SOPs reach step 8 doing different work — gating them on
/// proposal.md would be a false positive.
const STEP8_GATED_SOPS: &[&str] = &["jl-insight-research-proposal"];

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

    /// Returns `Some((run_id, sop_name, current_step))` for the live run
    /// if its `sop_name` is in [`STEP8_GATED_SOPS`] and `current_step ==
    /// 8`. Used by the step 8 size gate.
    fn snapshot_step8_gated_run(&self) -> Option<(String, String, u32)> {
        let engine = self.engine.lock().ok()?;
        engine
            .active_runs()
            .values()
            .find(|r| {
                matches!(r.status, SopRunStatus::Running)
                    && r.current_step == 8
                    && STEP8_GATED_SOPS.contains(&r.sop_name.as_str())
            })
            .map(|r| (r.run_id.clone(), r.sop_name.clone(), r.current_step))
    }

    /// If the call is `sop_advance status=completed` while a step-8-gated
    /// SOP run is at step 8, glob for the latest `*proposal.md` and
    /// require its size to be at least [`STEP8_PROPOSAL_MIN_BYTES`].
    /// Returns `Some(cancel_message)` if the gate should fire,
    /// `None` if the call should pass through.
    fn check_step8_size_gate(&self, name: &str, args: &Value) -> Option<String> {
        if name != STEP8_GATE_TOOL {
            return None;
        }
        let status = args.get("status").and_then(|v| v.as_str())?;
        if status != "completed" {
            return None;
        }
        let (run_id, sop_name, _) = self.snapshot_step8_gated_run()?;

        let proposals: Vec<_> = match glob::glob(STEP8_PROPOSAL_GLOB) {
            Ok(g) => g.flatten().collect(),
            Err(_) => return None,
        };

        if proposals.is_empty() {
            return Some(format!(
                "🚫 SOP step 8 size gate: no `*proposal.md` found under `{glob_pattern}`.\n\n\
                 Active run `{run_id}` (`{sop_name}`) is at step 8 (报告组装) and you called \
                 `sop_advance status=completed`, but no merged proposal.md exists yet.\n\n\
                 Step 8 must produce a single final `case_library/insight/<pi_name>/<date>_proposal.md` \
                 ≥{min_kb}KB (typical 25-35KB) by merging the chapter files \
                 (pi_intel.md / disease_scan.md / hypotheses.md / technical_route.md / \
                 budget.md / output_matching.md / pm_review.md).\n\n\
                 To proceed: file_write the merged proposal.md, then re-call sop_advance.",
                glob_pattern = STEP8_PROPOSAL_GLOB,
                min_kb = STEP8_PROPOSAL_MIN_BYTES / 1024,
            ));
        }

        let latest = proposals
            .iter()
            .filter_map(|p| {
                std::fs::metadata(p)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|t| (p, t))
            })
            .max_by_key(|(_, t)| *t)?
            .0;

        let size = std::fs::metadata(latest).ok()?.len();
        if size >= STEP8_PROPOSAL_MIN_BYTES {
            return None;
        }

        Some(format!(
            "🚫 SOP step 8 size gate: latest `proposal.md` is too small.\n\n\
             • Path:     `{path}`\n\
             • Size:     {size} bytes\n\
             • Required: ≥{required} bytes (≈{min_kb}KB, typical 25-35KB)\n\
             • Run:      `{run_id}` (`{sop_name}`, step 8)\n\n\
             This is a recurring V4-V11 bug: the LLM treats the chapter files \
             (pi_intel.md / disease_scan.md / hypotheses.md / technical_route.md / \
             budget.md / output_matching.md / pm_review.md, ~67KB combined) as \
             'report assembled', producing only a 4-5KB index proposal.md that \
             points at the chapter files instead of merging them. PI delivery \
             experience: 'this looks unfinished'.\n\n\
             To proceed:\n\
             • file_write parts (`part1.md` / `part2.md` / `part3.md`) under \
               `case_library/insight/<pi_name>/`.\n\
             • shell `cat case_library/insight/<pi_name>/part*.md > \
               case_library/insight/<pi_name>/<date>_proposal.md`.\n\
             • Re-call `sop_advance status=completed` once the merged \
               proposal.md is ≥{min_kb}KB.",
            path = latest.display(),
            size = size,
            required = STEP8_PROPOSAL_MIN_BYTES,
            min_kb = STEP8_PROPOSAL_MIN_BYTES / 1024,
            run_id = run_id,
            sop_name = sop_name,
        ))
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

        // Gate 1: step 8 size gate on sop_advance status=completed
        if let Some(reason) = self.check_step8_size_gate(&name, &args) {
            tracing::warn!(
                hook = "sop-enforcement",
                tool = %name,
                "rejecting sop_advance: step 8 proposal.md size gate"
            );
            return HookResult::Cancel(reason);
        }

        // Gate 2: path allowlist on file_write / file_edit
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
        engine_with_run("test-sop", 1, status)
    }

    fn engine_with_run(
        sop_name: &str,
        current_step: u32,
        status: SopRunStatus,
    ) -> Arc<Mutex<SopEngine>> {
        let engine = SopEngine::new(SopConfig::default());
        let arc = Arc::new(Mutex::new(engine));
        // Inject an active run by hand. We avoid going through start_run
        // because that would require loading a real SOP definition.
        {
            let mut eng = arc.lock().unwrap();
            let run = crate::sop::types::SopRun {
                run_id: "run-test-0001".to_string(),
                sop_name: sop_name.to_string(),
                trigger_event: SopEvent {
                    source: SopTriggerSource::Manual,
                    topic: None,
                    payload: None,
                    timestamp: "2026-05-02T00:00:00Z".to_string(),
                },
                status,
                current_step,
                total_steps: 11,
                started_at: "2026-05-02T00:00:00Z".to_string(),
                completed_at: None,
                step_results: vec![],
                waiting_since: None,
                llm_calls_saved: 0,
            };
            eng.active_runs_mut_for_test()
                .insert(run.run_id.clone(), run);
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
        let hook = SopEnforcementHook::new(cfg(true, &["case_library/"]), empty_engine());
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
                .before_tool_call("file_write".into(), json!({"path": path, "content": "x"}))
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

    // ------------------------------------------------------------------
    // Step 8 size gate tests
    // ------------------------------------------------------------------
    //
    // These tests run with the process cwd temporarily switched to a
    // tempdir so the glob `case_library/insight/*/*proposal.md`
    // resolves against test fixtures. We serialize them with a mutex
    // because cwd is process-global and tokio runs tests in parallel.

    use std::path::PathBuf;
    use std::sync::Mutex as StdMutex;
    use std::sync::OnceLock;

    fn cwd_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    struct CwdGuard {
        original: PathBuf,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl CwdGuard {
        fn enter(target: &std::path::Path) -> Self {
            let guard = cwd_lock().lock().unwrap_or_else(|e| e.into_inner());
            let original = std::env::current_dir().expect("cwd readable");
            std::env::set_current_dir(target).expect("set cwd");
            Self {
                original,
                _guard: guard,
            }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    fn write_proposal(case_dir: &std::path::Path, pi_name: &str, kb: usize) -> PathBuf {
        let pi_dir = case_dir.join("case_library").join("insight").join(pi_name);
        std::fs::create_dir_all(&pi_dir).expect("create case dir");
        let proposal = pi_dir.join("2026-05-03_proposal.md");
        let payload: String = "x".repeat(kb * 1024);
        std::fs::write(&proposal, payload).expect("write proposal");
        proposal
    }

    #[tokio::test]
    async fn step8_size_gate_blocks_undersized_proposal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal(tmp.path(), "zhang_hong", 5); // 5KB < 20KB
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_run("jl-insight-research-proposal", 8, SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "completed", "output": "proposal done"}),
            )
            .await;
        match result {
            HookResult::Cancel(msg) => {
                assert!(msg.contains("step 8 size gate"), "msg: {msg}");
                assert!(msg.contains("5120 bytes"), "msg: {msg}");
                assert!(msg.contains("zhang_hong"), "msg: {msg}");
            }
            HookResult::Continue(_) => panic!("expected Cancel for 5KB proposal"),
        }
    }

    #[tokio::test]
    async fn step8_size_gate_passes_full_proposal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal(tmp.path(), "zhang_hong", 30); // 30KB ≥ 20KB
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_run("jl-insight-research-proposal", 8, SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "completed", "output": "proposal done"}),
            )
            .await;
        assert!(
            matches!(result, HookResult::Continue(_)),
            "30KB proposal should pass"
        );
    }

    #[tokio::test]
    async fn step8_size_gate_no_proposal_blocks() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // No proposal.md created at all.
        std::fs::create_dir_all(tmp.path().join("case_library/insight")).unwrap();
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_run("jl-insight-research-proposal", 8, SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "completed", "output": "proposal done"}),
            )
            .await;
        match result {
            HookResult::Cancel(msg) => {
                assert!(msg.contains("no `*proposal.md` found"), "msg: {msg}");
            }
            HookResult::Continue(_) => panic!("expected Cancel when no proposal exists"),
        }
    }

    #[tokio::test]
    async fn step8_size_gate_only_fires_at_step_8() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal(tmp.path(), "zhang_hong", 5);
        let _cwd = CwdGuard::enter(tmp.path());

        // Active run is at step 7, not step 8 → gate must not fire
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_run("jl-insight-research-proposal", 7, SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "completed", "output": "step 7 done"}),
            )
            .await;
        assert!(
            matches!(result, HookResult::Continue(_)),
            "step 7 advance should not trigger size gate"
        );
    }

    #[tokio::test]
    async fn step8_size_gate_only_fires_for_gated_sops() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal(tmp.path(), "zhang_hong", 5);
        let _cwd = CwdGuard::enter(tmp.path());

        // Different SOP (not in STEP8_GATED_SOPS) → gate must not fire
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_run("some-other-sop", 8, SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "completed", "output": "step 8 of unrelated sop"}),
            )
            .await;
        assert!(
            matches!(result, HookResult::Continue(_)),
            "unrelated SOP should bypass step 8 size gate"
        );
    }

    #[tokio::test]
    async fn step8_size_gate_ignores_non_completed_status() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal(tmp.path(), "zhang_hong", 5);
        let _cwd = CwdGuard::enter(tmp.path());

        // sop_advance with status=failed should not trigger size gate
        let hook = SopEnforcementHook::new(
            cfg(true, &["case_library/"]),
            engine_with_run("jl-insight-research-proposal", 8, SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "failed", "reason": "abandon"}),
            )
            .await;
        assert!(
            matches!(result, HookResult::Continue(_)),
            "status=failed should bypass size gate"
        );
    }
}
