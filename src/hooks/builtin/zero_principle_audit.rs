//! Zero-Principle Audit Hook (PharmaClaw 第零原则审计).
//!
//! Enforces the "zero principle" used by medical SOPs: **before any
//! `file_write` or `shell` tool call inside a medical SOP turn, the
//! agent MUST have called `file_read` on a `SKILL.md` file**.
//!
//! Without this hook, an LLM could skip reading the canonical SKILL.md
//! recipe and synthesise a script from pre-trained knowledge — a
//! violation of the PharmaClaw zero principle.
//!
//! ## Status
//!
//! This is a **skeleton (stub)** implementation. The trait is wired up
//! and the hook is registered, but the gating logic in
//! [`ZeroPrincipleAuditHook::before_tool_call`] is currently a
//! pass-through. Full enforcement (turn-state tracking, SKILL.md path
//! detection, `HookResult::Cancel` on violation) lands in a follow-up
//! PR per the spec in `pharmaclaw-05-zero-principle-hook.md`.
//!
//! ## Activation
//!
//! Only active for agent turns started by `sop_execute` whose SOP
//! declares `medical = true`. Non-medical turns pass through with zero
//! overhead. The medical flag plumbing (SOP → turn metadata → hook)
//! is not yet wired and will land alongside the full implementation.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::schema::ZeroPrincipleAuditConfig;
use crate::hooks::traits::{HookHandler, HookResult};

/// Per-turn audit state. Tracks which SKILL.md paths have been read
/// during the current turn so subsequent `file_write` / `shell` calls
/// can be validated.
///
/// TODO(phase1-impl): populate via `before_tool_call` when `file_read`
/// of a path containing `SKILL.md` is observed; clear on session end.
#[derive(Debug, Default)]
#[allow(dead_code)] // populated in follow-up PR
struct TurnAuditState {
    skill_paths_read: HashSet<PathBuf>,
    medical: bool,
}

/// Hook that enforces the PharmaClaw zero-principle on medical SOP turns.
///
/// See module documentation for design rationale and the staged
/// rollout plan.
pub struct ZeroPrincipleAuditHook {
    config: ZeroPrincipleAuditConfig,
    /// Per-turn state, keyed by turn id. Currently unused — populated
    /// in the follow-up PR that wires up turn metadata.
    #[allow(dead_code)]
    turns: Arc<Mutex<HashMap<String, TurnAuditState>>>,
}

impl ZeroPrincipleAuditHook {
    pub fn new(config: ZeroPrincipleAuditConfig) -> Self {
        if config.enabled {
            tracing::info!(
                hook = "zero-principle-audit",
                gated_tools = ?config.gated_tools,
                "zero-principle-audit hook initialised (skeleton — enforcement pending)"
            );
        }
        Self {
            config,
            turns: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl HookHandler for ZeroPrincipleAuditHook {
    fn name(&self) -> &str {
        "zero-principle-audit"
    }

    fn priority(&self) -> i32 {
        // Run before webhook-audit (-100) so violations are blocked before
        // they are reported to external audit systems. Higher than 0 keeps
        // it ahead of arbitrary user-defined hooks.
        100
    }

    async fn before_tool_call(&self, name: String, args: Value) -> HookResult<(String, Value)> {
        // STUB: short-circuit when disabled or not yet implemented.
        // Follow-up PR will:
        //   1. Look up turn metadata (medical flag) to decide if active
        //   2. If tool == file_read and args.path contains "SKILL.md",
        //      record the path in TurnAuditState.skill_paths_read
        //   3. If tool ∈ gated_tools and skill_paths_read is empty,
        //      return HookResult::Cancel(zero_principle_violation_msg(...))
        //   4. Otherwise Continue
        //
        // For now: pass-through. Hook is a no-op even when enabled,
        // but config validation and registration are exercised.
        if !self.config.enabled {
            return HookResult::Continue((name, args));
        }
        tracing::trace!(
            hook = "zero-principle-audit",
            tool = %name,
            "stub: passing through (enforcement pending)"
        );
        HookResult::Continue((name, args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_disabled_hook() -> ZeroPrincipleAuditHook {
        ZeroPrincipleAuditHook::new(ZeroPrincipleAuditConfig {
            enabled: false,
            gated_tools: vec!["file_write".into(), "shell".into()],
            allowed_paths: vec![],
        })
    }

    fn make_enabled_hook() -> ZeroPrincipleAuditHook {
        ZeroPrincipleAuditHook::new(ZeroPrincipleAuditConfig {
            enabled: true,
            gated_tools: vec!["file_write".into(), "shell".into()],
            allowed_paths: vec![],
        })
    }

    #[test]
    fn hook_name_is_stable() {
        let hook = make_disabled_hook();
        assert_eq!(hook.name(), "zero-principle-audit");
    }

    #[test]
    fn priority_is_above_webhook_audit() {
        // webhook-audit uses -100. We must run before it so blocked
        // violations don't reach the external audit endpoint.
        let hook = make_disabled_hook();
        assert!(hook.priority() > -100);
    }

    #[tokio::test]
    async fn disabled_hook_passes_through() {
        let hook = make_disabled_hook();
        let result = hook
            .before_tool_call("file_write".into(), serde_json::json!({"path": "/tmp/x"}))
            .await;
        assert!(!result.is_cancel());
    }

    #[tokio::test]
    async fn enabled_stub_still_passes_through() {
        // Enforcement is intentionally deferred. This test asserts the
        // current contract; it will be REPLACED (not merely extended)
        // when the follow-up PR lands real gating logic.
        let hook = make_enabled_hook();
        let result = hook
            .before_tool_call("file_write".into(), serde_json::json!({"path": "/tmp/x"}))
            .await;
        assert!(!result.is_cancel());
    }
}
