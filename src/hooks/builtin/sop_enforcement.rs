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

use std::path::PathBuf;
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

/// Per-chapter minimum sizes for the prerequisites diagnostic (A-fix
/// for the 2026-05-09 v_zhang dedup-stuck case). When Gate 1 rejects
/// a step 8 sop_advance, we list each chapter file's status against
/// these thresholds so the LLM has actionable repair signal instead
/// of just "size gate" → loop → dedup abort. Numbers are calibrated
/// from the gao_yuan / li_juanmei completed runs (median chapter
/// sizes were 8-15KB; we set thresholds at the 25th percentile so
/// "barely complete" still passes).
const CHAPTER_CHECKS: &[(&str, u64)] = &[
    ("pi_intel.md", 6 * 1024),
    ("disease_scan.md", 8 * 1024),
    ("hypotheses.md", 10 * 1024),
    ("technical_route.md", 6 * 1024),
    ("budget.md", 4 * 1024),
    ("output_matching.md", 4 * 1024),
    ("pm_review.md", 3 * 1024),
];

#[derive(Debug)]
enum ChapterStatus {
    Missing,
    Undersized { actual: u64, required: u64 },
    Ok { size: u64 },
}

#[derive(Debug)]
struct ChapterDiag {
    file: &'static str,
    status: ChapterStatus,
}

impl ChapterDiag {
    fn render_line(&self) -> String {
        match &self.status {
            ChapterStatus::Missing => format!("  ❌ MISSING:    {}", self.file),
            ChapterStatus::Undersized { actual, required } => format!(
                "  ⚠️  UNDERSIZED: {} ({} bytes < required {})",
                self.file, actual, required
            ),
            ChapterStatus::Ok { size } => format!("  ✅ ok ({} bytes): {}", size, self.file),
        }
    }

    fn is_blocker(&self) -> bool {
        matches!(
            self.status,
            ChapterStatus::Missing | ChapterStatus::Undersized { .. }
        )
    }
}

/// Glob pattern for step 8 deliverable proposal.md files. Matches
/// PharmaClaw `jl-insight-research-proposal` convention:
/// `case_library/insight/<pi_name>/<date>_proposal.md` and variants
/// like `proposal.md` / `final_proposal.md`.
const STEP8_PROPOSAL_GLOB: &str = "case_library/insight/*/*proposal.md";

/// SOP names that produce a deliverable proposal.md at step 8.
/// Other SOPs reach step 8 doing different work — gating them on
/// proposal.md would be a false positive.
const STEP8_GATED_SOPS: &[&str] = &["jl-insight-research-proposal"];

/// Patterns that indicate residual template placeholders the LLM
/// forgot to substitute. Origin: PI 高原 v_gao_v5 一审反馈 #1 整改
/// 单 P0-1: 报告第 1 页有 `[疾病方向]` `<auto-filled>` 残留致评审秒杀.
/// LLM self-review systematically blind to these — must be detected
/// by deterministic regex.
///
/// Patterns are kept narrow on purpose: we only catch shapes that are
/// almost certainly template residue, not normal markdown brackets
/// like `[PMID:12345]` or `[See §2.3]`. Specifically:
///   - Pure CJK inside square brackets (real Chinese template tokens).
///   - Angle-bracket auto-fill markers (`<auto-filled>`, `<TBD>`, ...).
///   - Triple-dash data placeholders (`---{var}---`, `---data---`).
const PLACEHOLDER_RESIDUE_PATTERNS: &[(&str, &str)] = &[
    // Pure-CJK bracket placeholders like [疾病方向] [关键靶点] [假说1].
    // Real template residue in PharmaClaw is virtually always all-CJK.
    // This avoids false positives on `[PMID:12345]` / `[作者 Year]`.
    (
        r"\[[\u{4e00}-\u{9fff}]+\]",
        "pure-CJK bracket placeholder (e.g. [疾病方向])",
    ),
    // Auto-fill markers from various templates.
    (
        r"<auto-filled>|<auto-fill>|<待填>|<TBD>|<TODO>|<XXX>",
        "auto-fill marker",
    ),
    // Triple-dash data placeholders.
    (
        r"---\{[a-z_]+\}---|---data---",
        "triple-dash data placeholder",
    ),
];

pub struct SopEnforcementHook {
    config: SopEnforcementConfig,
    engine: Arc<Mutex<SopEngine>>,
    /// Workspace root used to anchor relative globs (e.g.
    /// [`STEP8_PROPOSAL_GLOB`]). The daemon's CWD is the repo root,
    /// not the workspace, so unanchored globs miss `case_library/`
    /// entirely. Without this anchor every step 8 size-gate check
    /// returned "no proposal.md found" regardless of actual files.
    workspace_dir: PathBuf,
}

impl SopEnforcementHook {
    pub fn new(
        config: SopEnforcementConfig,
        engine: Arc<Mutex<SopEngine>>,
        workspace_dir: PathBuf,
    ) -> Self {
        Self {
            config,
            engine,
            workspace_dir,
        }
    }

    /// Diagnostic helper: pick the most-recently-modified case dir
    /// under `case_library/insight/*/` and return per-chapter status
    /// against [`CHAPTER_CHECKS`]. The most-recent dir is the active
    /// case (Step 1 always writes pi_intel.md first), so this is a
    /// reliable signal even when the bot hasn't yet created
    /// proposal.md. Used to convert Gate 1's "size gate" reject into
    /// actionable "missing/undersized files X, Y, Z" diagnosis so
    /// the bot can pivot instead of looping into the dedup abort.
    fn diagnose_case_dir(&self) -> Option<(PathBuf, Vec<ChapterDiag>)> {
        use std::time::SystemTime;
        let case_glob = self.workspace_dir.join("case_library/insight/*/");
        let case_glob_str = case_glob.to_string_lossy();
        let mut candidates: Vec<(PathBuf, SystemTime)> = match glob::glob(&case_glob_str) {
            Ok(g) => g
                .flatten()
                .filter(|p| p.is_dir())
                .filter_map(|p| {
                    std::fs::metadata(&p)
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .map(|t| (p, t))
                })
                .collect(),
            Err(_) => return None,
        };
        candidates.sort_by_key(|(_, t)| *t);
        let case_dir = candidates.last()?.0.clone();

        let mut diags: Vec<ChapterDiag> = Vec::with_capacity(CHAPTER_CHECKS.len());
        for (name, min_bytes) in CHAPTER_CHECKS {
            let path = case_dir.join(name);
            let size = std::fs::metadata(&path).ok().map(|m| m.len());
            let status = match size {
                None => ChapterStatus::Missing,
                Some(s) if s < *min_bytes => ChapterStatus::Undersized {
                    actual: s,
                    required: *min_bytes,
                },
                Some(s) => ChapterStatus::Ok { size: s },
            };
            diags.push(ChapterDiag { file: name, status });
        }
        Some((case_dir, diags))
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
    /// if its `sop_name` is in [`STEP8_GATED_SOPS`] and `current_step >=
    /// 8`. Used by the step 8 size gate.
    ///
    /// Why `>= 8` (2026-05-09 v_ma_yiming bypass fix): the original
    /// `== 8` guard let the LLM bypass the gate by sop_advance'ing past
    /// step 8 with an empty/undersized proposal.md and then writing the
    /// final proposal.md during step 9-11 archival. Once the run was
    /// past step 8 the gate could never re-arm. Expanding to `>= 8`
    /// means every sop_advance from step 8 onward (including the final
    /// step 11 → completed transition) re-checks proposal.md size.
    fn snapshot_step8_gated_run(&self) -> Option<(String, String, u32)> {
        let engine = self.engine.lock().ok()?;
        engine
            .active_runs()
            .values()
            .find(|r| {
                matches!(r.status, SopRunStatus::Running)
                    && r.current_step >= 8
                    && STEP8_GATED_SOPS.contains(&r.sop_name.as_str())
            })
            .map(|r| (r.run_id.clone(), r.sop_name.clone(), r.current_step))
    }

    /// If the call is `sop_advance` (status `completed` or `skipped`) while a
    /// step-8-gated SOP run is at step 8 or beyond, glob for the latest
    /// `*proposal.md` and require its size to be at least
    /// [`STEP8_PROPOSAL_MIN_BYTES`]. Returns `Some(cancel_message)` if the
    /// gate should fire, `None` if the call should pass through.
    ///
    /// Why `skipped` is also gated (2026-05-09 v_ma_yiming bypass fix): the
    /// original guard only checked `status == "completed"`, so the LLM could
    /// bypass the gate by calling `sop_advance status=skipped` at step 8 and
    /// then writing a small proposal.md during the archival phase. `failed`
    /// is the only escape hatch left because it terminates the entire run as
    /// `SopRunStatus::Failed` — that's the legitimate "this SOP cannot
    /// produce a deliverable" path, not the "skip the size check" path.
    fn check_step8_size_gate(&self, name: &str, args: &Value) -> Option<String> {
        if name != STEP8_GATE_TOOL {
            return None;
        }
        let status = args.get("status").and_then(|v| v.as_str())?;
        if status == "failed" {
            return None;
        }
        if status != "completed" && status != "skipped" {
            return None;
        }
        let (run_id, sop_name, current_step) = self.snapshot_step8_gated_run()?;

        let glob_pattern_path = self.workspace_dir.join(STEP8_PROPOSAL_GLOB);
        let glob_pattern_str = glob_pattern_path.to_string_lossy();
        let proposals: Vec<_> = match glob::glob(&glob_pattern_str) {
            Ok(g) => g.flatten().collect(),
            Err(_) => return None,
        };

        if proposals.is_empty() {
            // Diagnose what's actually present in the active case dir
            // so the LLM can repair specifically rather than retry the
            // same sop_advance call (the recurring 2026-05-09 v_zhang
            // dedup-abort failure mode).
            let diagnosis = match self.diagnose_case_dir() {
                Some((case_dir, diags)) => {
                    let chapter_lines: Vec<String> =
                        diags.iter().map(|d| d.render_line()).collect();
                    let blockers: Vec<&str> = diags
                        .iter()
                        .filter(|d| d.is_blocker())
                        .map(|d| d.file)
                        .collect();
                    format!(
                        "Active case dir: `{case}`\n\n\
                         Chapter file status (each step's required output):\n{lines}\n\n\
                         📋 BLOCKERS — file_write these BEFORE retrying sop_advance:\n  {blockers}",
                        case = case_dir.display(),
                        lines = chapter_lines.join("\n"),
                        blockers = if blockers.is_empty() {
                            "(none — only proposal.md missing)".to_string()
                        } else {
                            blockers.join(", ")
                        },
                    )
                }
                None => "(could not locate active case dir)".to_string(),
            };

            return Some(format!(
                "🚫 SOP step 8 size gate: no `*proposal.md` found under `{glob_pattern}`.\n\n\
                 Active run `{run_id}` (`{sop_name}`) is at step {current_step} (step 8 报告组装 is \
                 the deliverable producer) and you called `sop_advance status={status}`, but no \
                 merged proposal.md exists yet.\n\n\
                 {diagnosis}\n\n\
                 ▶️  REPAIR PLAN (do these in order, do NOT retry sop_advance until done):\n\
                 1. For each MISSING / UNDERSIZED chapter above, file_write its content. The \
                    chapter sizes you wrote in earlier steps did not pass; reopen and expand them.\n\
                 2. After all chapters are ✅, file_write the merged proposal.md \
                    (`case_library/insight/<pi_name>/<date>_proposal.md`, ≥{min_kb}KB, typical \
                    25-35KB) by combining the 7 chapter files.\n\
                 3. Then re-call sop_advance status=completed.\n\n\
                 ⚠️ Calling sop_advance with the same args without doing the above will trigger \
                 the agent dedup detector and abort the conversation. The fix is content, not retry.",
                glob_pattern = glob_pattern_str,
                min_kb = STEP8_PROPOSAL_MIN_BYTES / 1024,
                diagnosis = diagnosis,
                current_step = current_step,
                status = status,
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

        // Same actionable-diagnosis logic as the no-proposal branch:
        // list each chapter's status so the LLM can identify which
        // ones are still under-sized and need expansion before the
        // proposal.md merge will pass.
        let diagnosis = match self.diagnose_case_dir() {
            Some((_case_dir, diags)) => {
                let chapter_lines: Vec<String> = diags.iter().map(|d| d.render_line()).collect();
                let blockers: Vec<&str> = diags
                    .iter()
                    .filter(|d| d.is_blocker())
                    .map(|d| d.file)
                    .collect();
                format!(
                    "Chapter file status:\n{lines}\n\n\
                     📋 Likely root cause: the missing/undersized chapters were never written or \
                     are stubs. Repair sequence: {blockers_or_done}",
                    lines = chapter_lines.join("\n"),
                    blockers_or_done = if blockers.is_empty() {
                        "all chapters look fine — proposal.md just needs a fuller merge \
                         (cat the chapter files together to reach ≥20KB)."
                            .to_string()
                    } else {
                        format!(
                            "expand {} first, then re-merge into proposal.md.",
                            blockers.join(" + ")
                        )
                    },
                )
            }
            None => "(could not locate active case dir)".to_string(),
        };

        Some(format!(
            "🚫 SOP step 8 size gate: latest `proposal.md` is too small.\n\n\
             • Path:     `{path}`\n\
             • Size:     {size} bytes\n\
             • Required: ≥{required} bytes (≈{min_kb}KB, typical 25-35KB)\n\
             • Run:      `{run_id}` (`{sop_name}`, step {current_step})\n\n\
             {diagnosis}\n\n\
             This is a recurring V4-V11 bug: the LLM treats the chapter files \
             (pi_intel.md / disease_scan.md / hypotheses.md / technical_route.md / \
             budget.md / output_matching.md / pm_review.md, ~67KB combined) as \
             'report assembled', producing only a 4-5KB index proposal.md that \
             points at the chapter files instead of merging them. PI delivery \
             experience: 'this looks unfinished'.\n\n\
             ▶️  To proceed:\n\
             • For each ❌ MISSING or ⚠️ UNDERSIZED chapter above, file_write its content first.\n\
             • Then file_write the merged proposal.md by combining all 7 chapters.\n\
             • Re-call `sop_advance status=completed` once the merged proposal.md is ≥{min_kb}KB.\n\n\
             ⚠️ Retrying sop_advance with the same args without writing the missing files will \
             trigger the agent dedup detector and abort the conversation. The fix is content, not retry.",
            path = latest.display(),
            size = size,
            required = STEP8_PROPOSAL_MIN_BYTES,
            min_kb = STEP8_PROPOSAL_MIN_BYTES / 1024,
            run_id = run_id,
            sop_name = sop_name,
            current_step = current_step,
            diagnosis = diagnosis,
        ))
    }

    /// Gate 3: scan all proposal.md files for residual template
    /// placeholders. Triggered on `sop_advance` at step 8+ for gated
    /// SOPs (regardless of `status` value — this gate is broader than
    /// the size gate because the residue bug is severity-independent).
    ///
    /// Returns `Some(cancel_message)` on detection, `None` if clean.
    fn check_placeholder_residue_gate(&self, name: &str, args: &Value) -> Option<String> {
        if name != STEP8_GATE_TOOL {
            return None;
        }
        // Detect at any sop_advance from step 8 onwards. Status doesn't
        // matter — placeholder residue is fatal at any point past assembly.
        let (run_id, sop_name, current_step) = {
            let engine = self.engine.lock().ok()?;
            engine
                .active_runs()
                .values()
                .find(|r| {
                    matches!(r.status, SopRunStatus::Running)
                        && r.current_step >= 8
                        && STEP8_GATED_SOPS.contains(&r.sop_name.as_str())
                })
                .map(|r| (r.run_id.clone(), r.sop_name.clone(), r.current_step))?
        };
        let _ = (args,); // status not used; gate is broader

        let glob_pattern_path = self.workspace_dir.join(STEP8_PROPOSAL_GLOB);
        let glob_pattern_str = glob_pattern_path.to_string_lossy();
        let proposals: Vec<_> = match glob::glob(&glob_pattern_str) {
            Ok(g) => g.flatten().collect(),
            Err(_) => return None,
        };
        if proposals.is_empty() {
            return None;
        }

        let mut findings: Vec<String> = Vec::new();
        for proposal in &proposals {
            let Ok(content) = std::fs::read_to_string(proposal) else {
                continue;
            };
            for (pattern, label) in PLACEHOLDER_RESIDUE_PATTERNS {
                let Ok(re) = regex::Regex::new(pattern) else {
                    continue;
                };
                let mut matches: Vec<String> = re
                    .find_iter(&content)
                    .map(|m| m.as_str().to_string())
                    .collect();
                if !matches.is_empty() {
                    matches.sort();
                    matches.dedup();
                    let sample: Vec<String> = matches.iter().take(5).cloned().collect();
                    findings.push(format!(
                        "  • {path}: {label}\n    matches: [{matches}{ellipsis}]",
                        path = proposal.display(),
                        label = label,
                        matches = sample.join(", "),
                        ellipsis = if matches.len() > 5 { ", ..." } else { "" },
                    ));
                }
            }
        }
        if findings.is_empty() {
            return None;
        }

        Some(format!(
            "🚫 SOP step {current_step} placeholder residue gate: template placeholders \
             not substituted in proposal output.\n\n\
             Active run `{run_id}` (`{sop_name}`) is at step {current_step} and called \
             `sop_advance`, but residual template tokens (e.g. `[疾病方向]`, `<auto-filled>`, \
             `<TBD>`) were detected in proposal files. These are致命 bugs: PI 评审 v_gao_v5 \
             第 1 页因 `[疾病方向]` 和 `<auto-filled>` 残留被直接判 P0 阻断级 \"不会签字\".\n\n\
             Findings:\n{findings}\n\n\
             To proceed:\n\
             • file_read each affected file, locate the placeholder context.\n\
             • Substitute with the actual content from pi_intel.md / disease_scan.md / \
               hypotheses.md (the chapter files have the real values).\n\
             • Re-call `sop_advance` once all placeholders are gone.\n\
             • If a placeholder is intentional (e.g. `__PI_FILL__` in PI-only fields), \
               it does NOT trigger this gate — only `[CAPS_BRACKET]`, `<auto-filled>`, \
               and `<TBD>`-class markers do.",
            findings = findings.join("\n"),
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

        // Gate 3: placeholder residue scan on sop_advance at step 8+
        // (broader than Gate 1 — fires on any status, not just completed)
        if let Some(reason) = self.check_placeholder_residue_gate(&name, &args) {
            tracing::warn!(
                hook = "sop-enforcement",
                tool = %name,
                "rejecting sop_advance: placeholder residue in proposal"
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

    /// Test helper: construct hook with CWD-relative workspace (`.`).
    /// Step-8 size-gate tests use [`CwdGuard`] to chdir into a tempdir,
    /// so a relative workspace is correct here. Production code uses an
    /// absolute path resolved from `config.workspace_dir`.
    fn test_hook(
        config: SopEnforcementConfig,
        engine: Arc<Mutex<SopEngine>>,
    ) -> SopEnforcementHook {
        SopEnforcementHook::new(config, engine, PathBuf::from("."))
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
        let hook = test_hook(cfg(false, &["case_library/"]), empty_engine());
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
        let hook = test_hook(cfg(true, &["case_library/"]), empty_engine());
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
        let hook = test_hook(
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
        let hook = test_hook(
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
        let hook = test_hook(
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
        let hook = test_hook(
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
        let hook = test_hook(
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

        let hook = test_hook(
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

        let hook = test_hook(
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

        let hook = test_hook(
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

    /// 2026-05-09 v_zhang regression: Gate 1 reject must list each
    /// chapter file's MISSING / UNDERSIZED / OK status so the LLM can
    /// repair specifically instead of looping into dedup-abort. The
    /// previous reject message just said "size gate" → bot reissued
    /// the same sop_advance call → dedup detector aborted the turn.
    #[tokio::test]
    async fn step8_size_gate_reject_lists_missing_chapters() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // Active case dir has only pi_intel.md, all other chapters
        // missing — the exact 2026-05-09 v_zhang state.
        let case_dir = tmp.path().join("case_library/insight/zhang_xiaobo");
        std::fs::create_dir_all(&case_dir).unwrap();
        std::fs::write(case_dir.join("pi_intel.md"), "x".repeat(8 * 1024)).unwrap();
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = test_hook(
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
                // Must enumerate each missing chapter so the LLM
                // knows exactly what to file_write before retrying.
                assert!(
                    msg.contains("disease_scan.md"),
                    "missing disease_scan in msg: {msg}"
                );
                assert!(
                    msg.contains("hypotheses.md"),
                    "missing hypotheses in msg: {msg}"
                );
                assert!(
                    msg.contains("technical_route.md"),
                    "missing technical_route in msg: {msg}"
                );
                assert!(msg.contains("budget.md"), "missing budget in msg: {msg}");
                assert!(
                    msg.contains("output_matching.md"),
                    "missing output_matching in msg: {msg}"
                );
                // Must explicitly mark them as MISSING (not just listed)
                assert!(
                    msg.contains("MISSING"),
                    "expected MISSING marker in diagnosis: {msg}"
                );
                // Must show the OK chapter (pi_intel.md) too so the
                // LLM doesn't redundantly rewrite it.
                assert!(
                    msg.contains("pi_intel.md"),
                    "msg should list pi_intel: {msg}"
                );
                // Must include the case dir path so the LLM knows
                // where to write.
                assert!(
                    msg.contains("zhang_xiaobo"),
                    "expected case dir path in msg: {msg}"
                );
                // Must explicitly warn against retrying the same
                // sop_advance (dedup abort prevention).
                assert!(
                    msg.contains("dedup") || msg.contains("retry"),
                    "expected dedup/retry warning in msg: {msg}"
                );
            }
            HookResult::Continue(_) => panic!("expected Cancel when no proposal exists"),
        }
    }

    /// Same as above but for the undersized-proposal branch — when
    /// proposal.md exists but is < 20KB, the diagnosis must still
    /// list chapter status so the LLM knows whether to expand chapters
    /// or just re-merge.
    #[tokio::test]
    async fn step8_size_gate_undersized_proposal_lists_chapter_status() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal(tmp.path(), "zhang_xiaobo", 5); // 5KB < 20KB
                                                       // Add some chapters so diagnosis has a mix of OK/MISSING:
        let case_dir = tmp.path().join("case_library/insight/zhang_xiaobo");
        std::fs::write(case_dir.join("pi_intel.md"), "x".repeat(8 * 1024)).unwrap();
        std::fs::write(case_dir.join("disease_scan.md"), "x".repeat(10 * 1024)).unwrap();
        // hypotheses, technical_route, budget, output_matching, pm_review missing
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = test_hook(
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
                assert!(msg.contains("Chapter file status"), "msg: {msg}");
                // OK chapters must show ✅
                assert!(msg.contains("✅"), "expected ✅ for ok chapters: {msg}");
                // Missing chapters must show ❌
                assert!(
                    msg.contains("❌"),
                    "expected ❌ for missing chapters: {msg}"
                );
                assert!(msg.contains("hypotheses.md"), "msg: {msg}");
                assert!(msg.contains("budget.md"), "msg: {msg}");
            }
            HookResult::Continue(_) => panic!("expected Cancel for 5KB proposal"),
        }
    }

    #[tokio::test]
    async fn step8_size_gate_only_fires_at_step_8() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal(tmp.path(), "zhang_hong", 5);
        let _cwd = CwdGuard::enter(tmp.path());

        // Active run is at step 7, not step 8 → gate must not fire
        let hook = test_hook(
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
        let hook = test_hook(
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
        let hook = test_hook(
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

    // ── Gate 3 (placeholder residue) tests ─────────────────────────

    /// Helper: write a 22 KB proposal whose body contains the supplied
    /// residue text in addition to filler. Size > 20 KB so size gate
    /// passes and Gate 3 is the only thing that can fire.
    fn write_proposal_with_text(
        case_dir: &std::path::Path,
        pi_name: &str,
        residue: &str,
    ) -> PathBuf {
        let pi_dir = case_dir.join("case_library").join("insight").join(pi_name);
        std::fs::create_dir_all(&pi_dir).expect("create case dir");
        let proposal = pi_dir.join("2026-05-03_proposal.md");
        let mut payload = String::with_capacity(22 * 1024 + residue.len());
        payload.push_str(residue);
        payload.push('\n');
        payload.push_str(&"x".repeat(22 * 1024));
        std::fs::write(&proposal, payload).expect("write proposal");
        proposal
    }

    #[tokio::test]
    async fn placeholder_residue_gate_catches_caps_bracket() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal_with_text(
            tmp.path(),
            "zhang_hong",
            "本课题聚焦 [疾病方向] 中的 [关键靶点] 调控机制",
        );
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = test_hook(
            cfg(true, &["case_library/"]),
            engine_with_run("jl-insight-research-proposal", 8, SopRunStatus::Running),
        );
        // Even with status=in_progress (the LLM bypass that defeats Gate 1),
        // Gate 3 should still fire.
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "in_progress", "output": "step 8 progressing"}),
            )
            .await;
        match result {
            HookResult::Cancel(msg) => {
                assert!(
                    msg.contains("placeholder residue gate"),
                    "msg should mention placeholder residue gate: {msg}"
                );
                assert!(
                    msg.contains("[疾病方向]") || msg.contains("[关键靶点]"),
                    "msg should surface a concrete residue match: {msg}"
                );
            }
            HookResult::Continue(_) => panic!("expected Cancel for caps-bracket residue"),
        }
    }

    #[tokio::test]
    async fn placeholder_residue_gate_catches_auto_fill_marker() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal_with_text(
            tmp.path(),
            "zhang_hong",
            "SOP run_id: <auto-filled>\n调用数据库: <TBD>",
        );
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = test_hook(
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
                assert!(msg.contains("placeholder residue gate"), "msg: {msg}");
                assert!(
                    msg.contains("<auto-filled>") || msg.contains("<TBD>"),
                    "msg should surface auto-fill markers: {msg}"
                );
            }
            HookResult::Continue(_) => panic!("expected Cancel for auto-fill markers"),
        }
    }

    #[tokio::test]
    async fn placeholder_residue_gate_passes_clean_proposal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // 22 KB of clean content — no residue patterns, includes
        // legitimate `__PI_FILL__` (which is NOT a residue marker).
        write_proposal_with_text(
            tmp.path(),
            "zhang_hong",
            "PI 邮箱 __PI_FILL__: zhang.hong@example.cn (PI 自填)",
        );
        let _cwd = CwdGuard::enter(tmp.path());

        let hook = test_hook(
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
            "clean proposal with __PI_FILL__ should NOT trigger placeholder residue gate"
        );
    }

    #[tokio::test]
    async fn placeholder_residue_gate_only_fires_at_step_8_plus() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_proposal_with_text(tmp.path(), "zhang_hong", "[疾病方向] 占位符存在");
        let _cwd = CwdGuard::enter(tmp.path());

        // Active run is at step 5 — placeholders pre-step-8 are normal
        // (still being assembled). Gate 3 must NOT fire.
        let hook = test_hook(
            cfg(true, &["case_library/"]),
            engine_with_run("jl-insight-research-proposal", 5, SopRunStatus::Running),
        );
        let result = hook
            .before_tool_call(
                "sop_advance".into(),
                json!({"status": "completed", "output": "step 5 done"}),
            )
            .await;
        assert!(
            matches!(result, HookResult::Continue(_)),
            "placeholders pre-step-8 should not trigger Gate 3"
        );
    }
}
