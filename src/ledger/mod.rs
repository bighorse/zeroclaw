//! 任务证据台账（P1a 读取层）。
//!
//! 目的：让「每条结论都能挂到工具的真实原始返回」这句话成立。
//!
//! 设计取舍——**不改 agent 循环**：
//! 工具调用的参数与完整输出，`runtime_trace`（`runtime_trace_mode = "full"`）已经在
//! 逐条落盘（`tool_call_start` / `tool_call_result`，含 `turn_id`，凭据已脱敏）。
//! 与其把任务上下文一路穿进工具循环（那段跑在隔离 runtime 里，是历史死锁的高发区），
//! 不如在既有采集之上建一个**只读**账本。代价是：只能看到 trace 记下的东西，
//! 看不到的一律如实说「没有」，不猜。
//!
//! 诚实边界（务必保持）：
//! - trace 未开 full → 账本不可用，端点如实报原因，不返回空列表冒充「没有证据」。
//! - 只有 external 类工具的返回能当「依据」；本地文件与自产内容单列，不许冒充外部证据。
//! - 证据 id 不可猜（含内容哈希前缀），模型无法凭空编一个能解析的引用。
//! - 本模块不做防篡改：trace 文件的只追加与不可变（chattr +a）属运维措施，尚未实施，
//!   端点里如实标注 `tamper_evident: false`，不要在界面上把它说成"可验证不可改"。

use crate::observability::runtime_trace::{self, RuntimeTraceEvent};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;

/// 单条工具调用的来源分类。只有 External 能作为对外「依据」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceClass {
    /// 向外部世界取的数据（网络请求、检索、声明了数据源的脚本）
    External,
    /// 读本地文件——可能是几天前缓存下来的，不能冒充"当天查到的"
    LocalFile,
    /// 助手自己产生的（写文件、记忆、SOP 推进等）
    SelfProduced,
}

impl SourceClass {
    /// 按工具名分类。宁可保守：认不出的一律算自产，不许悄悄升级成「外部依据」。
    pub fn of_tool(tool: &str) -> Self {
        match tool {
            "http_request" | "web_fetch" | "web_search" | "browser" => Self::External,
            "file_read" => Self::LocalFile,
            // 其余一律自产——**shell 也在此列**：它什么都能干，可能真的调了外部 API，
            // 也可能只是 echo。无法确定就不许升级成「外部依据」，
            // 等技能登记机制上线（P2，脚本声明数据源 host）再按登记放开。
            _ => Self::SelfProduced,
        }
    }
}

/// 一条证据 = 一次工具调用的完整记录。
#[derive(Debug, Clone, Serialize)]
pub struct Evidence {
    /// 不可猜的证据编号，如 `E3-9f2a1c`。后缀取内容哈希，模型编不出能解析的引用。
    pub eid: String,
    pub n: u32,
    pub at: String,
    pub tool: String,
    pub source_class: SourceClass,
    /// 调用参数摘要（截断）
    pub args_excerpt: Option<String>,
    /// 输出摘要（截断；全文经 /api/tasks/{id}/evidence/{eid} 取）
    pub output_excerpt: String,
    /// 完整输出的字节数与 sha256——客户端可核对"给我看的就是存下来的"
    pub output_bytes: usize,
    pub sha256: String,
    pub success: bool,
    pub turn_id: String,
    /// 文件类工具（file_write / file_read …）操作的路径——前台据此列出「产物」
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// 网络类工具请求的地址（http_request 等）——「取自哪里」
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// 账本可用性。不可用时必须给出人能看懂的原因，不能静默返回空。
#[derive(Debug, Clone, Serialize)]
pub struct LedgerAvailability {
    pub available: bool,
    pub reason: Option<String>,
    /// 是否具备防篡改保证（只追加 + 不可变）。当前一律 false，如实标注。
    pub tamper_evident: bool,
}

const EXCERPT: usize = 400;
/// 单次扫描的 trace 条数上限：证据读取是交互路径，不能因为日志变大就把请求拖垮。
const SCAN_LIMIT: usize = 20_000;

fn excerpt(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…（已截断，全文见证据详情）")
}

fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

pub fn availability(trace_path: &Path, mode_is_full: bool) -> LedgerAvailability {
    if !mode_is_full {
        return LedgerAvailability {
            available: false,
            reason: Some(
                "证据台账需要 observability.runtime_trace_mode = \"full\"；当前未开启，因此本助手没有可供核对的工具调用记录。".into(),
            ),
            tamper_evident: false,
        };
    }
    if !trace_path.exists() {
        return LedgerAvailability {
            available: false,
            reason: Some("证据台账已开启，但尚未产生任何记录文件。".into()),
            tamper_evident: false,
        };
    }
    LedgerAvailability {
        available: true,
        reason: None,
        tamper_evident: false,
    }
}

fn payload_str(ev: &RuntimeTraceEvent, key: &str) -> Option<String> {
    ev.payload.get(key).and_then(|v| {
        v.as_str()
            .map(str::to_string)
            .or_else(|| Some(v.to_string()))
    })
}

/// 取某一轮（turn）里的全部工具调用，按时间顺序编号成证据。
pub fn evidence_for_turn(trace_path: &Path, turn_id: &str) -> Vec<Evidence> {
    evidence_with_output_for_turn(trace_path, turn_id)
        .into_iter()
        .map(|(e, _)| e)
        .collect()
}

/// 同 [`evidence_for_turn`]，但连同每次调用的**完整输出**一起给出（核对 PMID 之类要用全文，
/// 摘要是截断的）。一次扫描，不必为每条证据再读一遍 trace。
pub fn evidence_with_output_for_turn(trace_path: &Path, turn_id: &str) -> Vec<(Evidence, String)> {
    let Ok(events) = runtime_trace::load_events(trace_path, SCAN_LIMIT, None, None) else {
        return Vec::new();
    };
    // load_events 返回的是倒序（最新在前），这里按时间正序编号
    let mut chron: Vec<&RuntimeTraceEvent> = events
        .iter()
        .filter(|e| e.turn_id.as_deref() == Some(turn_id))
        .collect();
    chron.reverse();

    // tool_call_start 提供参数，tool_call_result 提供输出；按 (iteration, tool) 配对
    let mut args_by_key: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    // 完整参数（不截断），只用来抽取文件路径
    for e in &chron {
        if e.event_type == "tool_call_start" {
            let tool = payload_str(e, "tool").unwrap_or_default();
            let iter = payload_str(e, "iteration").unwrap_or_default();
            if let Some(a) = payload_str(e, "arguments") {
                args_by_key.insert(format!("{iter}:{tool}"), a);
            }
        }
    }

    let mut out = Vec::new();
    let mut n = 0u32;
    for e in &chron {
        if e.event_type != "tool_call_result" {
            continue;
        }
        let tool = payload_str(e, "tool").unwrap_or_default();
        let iter = payload_str(e, "iteration").unwrap_or_default();
        let output = payload_str(e, "output").unwrap_or_default();
        n += 1;
        let sha = sha256_hex(&output);
        let full_args = args_by_key.get(&format!("{iter}:{tool}"));
        let ev = Evidence {
            eid: format!("E{n}-{}", &sha[..6]),
            n,
            at: e.timestamp.clone(),
            source_class: SourceClass::of_tool(&tool),
            path: full_args.and_then(|a| path_arg(a)),
            url: full_args.and_then(|a| url_arg(a)),
            args_excerpt: full_args.map(|a| excerpt(a, 200)),
            output_excerpt: excerpt(&output, EXCERPT),
            output_bytes: output.len(),
            sha256: sha,
            success: e.success.unwrap_or(true),
            turn_id: turn_id.to_string(),
            tool,
        };
        out.push((ev, output));
    }
    out
}

/// 取一条证据的**全文**（供 `/api/tasks/{id}/evidence/{eid}`）。
pub fn evidence_full(trace_path: &Path, turn_id: &str, eid: &str) -> Option<(Evidence, String)> {
    let Ok(events) =
        runtime_trace::load_events(trace_path, SCAN_LIMIT, Some("tool_call_result"), None)
    else {
        return None;
    };
    let mut chron: Vec<&RuntimeTraceEvent> = events
        .iter()
        .filter(|e| e.turn_id.as_deref() == Some(turn_id))
        .collect();
    chron.reverse();

    let metas = evidence_for_turn(trace_path, turn_id);
    let meta = metas.iter().find(|m| m.eid == eid)?;
    let idx = usize::try_from(meta.n).ok()?.checked_sub(1)?;
    let full = payload_str(chron.get(idx)?, "output").unwrap_or_default();
    // 内容哈希必须对得上，否则说明 trace 被改过或编号错位——宁可不给，也不给错的
    if sha256_hex(&full) != meta.sha256 {
        return None;
    }
    Some((meta.clone(), full))
}

/// 从工具参数（JSON）里取文件路径。只认 `path` / `file_path` 键。
fn path_arg(args: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args).ok()?;
    ["path", "file_path"]
        .iter()
        .find_map(|k| v.get(*k).and_then(|p| p.as_str()))
        .map(str::to_string)
}

/// 从工具参数里取请求地址（`url` 键）。
fn url_arg(args: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args).ok()?;
    v.get("url").and_then(|u| u.as_str()).map(str::to_string)
}

/// 一单全部对话轮里的证据连同完整输出，编号与 [`evidence_for_run`] 一致。
pub fn evidence_with_output_for_run(trace_path: &Path, run_id: &str) -> Vec<(Evidence, String)> {
    let mut out: Vec<(Evidence, String)> = Vec::new();
    for t in turns_for_run(trace_path, run_id) {
        for (mut e, full) in evidence_with_output_for_turn(trace_path, &t) {
            let n = u32::try_from(out.len())
                .unwrap_or(u32::MAX)
                .saturating_add(1);
            e.eid = format!("E{n}-{}", &e.sha256[..6]);
            e.n = n;
            out.push((e, full));
        }
    }
    out
}

/// 碰过某个 run 的全部对话轮，按时间先后。
///
/// 一单不止一轮：派活那一轮做到门步停下，批准后的续跑是另一轮——正式报告就是在续跑轮里
/// 写出来的。只取一轮的话，「依据」和「产物」都会漏掉后半程。
pub fn turns_for_run(trace_path: &Path, run_id: &str) -> Vec<String> {
    let Ok(events) = runtime_trace::load_events(trace_path, SCAN_LIMIT, None, Some(run_id)) else {
        return Vec::new();
    };
    let mut turns: Vec<String> = Vec::new();
    // load_events 最新在前；倒过来按时间正序去重
    for e in events.iter().rev() {
        if let Some(t) = &e.turn_id {
            if !turns.contains(t) {
                turns.push(t.clone());
            }
        }
    }
    turns
}

/// 一单全部对话轮里的证据，编号跨轮连续（E1、E2… 后缀仍是内容哈希）。
pub fn evidence_for_run(trace_path: &Path, run_id: &str) -> Vec<Evidence> {
    let mut out: Vec<Evidence> = Vec::new();
    for t in turns_for_run(trace_path, run_id) {
        for mut e in evidence_for_turn(trace_path, &t) {
            let n = u32::try_from(out.len())
                .unwrap_or(u32::MAX)
                .saturating_add(1);
            e.eid = format!("E{n}-{}", &e.sha256[..6]);
            e.n = n;
            out.push(e);
        }
    }
    out
}

/// 按跨轮编号取一条证据的全文（哈希校验不变：对不上就不给）。
pub fn evidence_full_for_run(
    trace_path: &Path,
    run_id: &str,
    eid: &str,
) -> Option<(Evidence, String)> {
    let mut global = 0u32;
    for t in turns_for_run(trace_path, run_id) {
        for e in evidence_for_turn(trace_path, &t) {
            global = global.saturating_add(1);
            if format!("E{global}-{}", &e.sha256[..6]) == eid {
                let (mut meta, full) = evidence_full(trace_path, &t, &e.eid)?;
                meta.eid = eid.to_string();
                meta.n = global;
                return Some((meta, full));
            }
        }
    }
    None
}

/// 把某个 SOP run 关联到它所在的对话轮：
/// 在 trace 里找输出中提到该 run_id 的 `sop_execute` / `sop_advance` 调用，取其 turn_id。
pub fn turn_for_run(trace_path: &Path, run_id: &str) -> Option<String> {
    let events = runtime_trace::load_events(trace_path, SCAN_LIMIT, None, Some(run_id)).ok()?;
    events
        .iter()
        .find(|e| {
            e.turn_id.is_some()
                && payload_str(e, "tool")
                    .map(|t| t.starts_with("sop_"))
                    .unwrap_or(false)
        })
        .or_else(|| events.iter().find(|e| e.turn_id.is_some()))
        .and_then(|e| e.turn_id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_class_is_conservative() {
        assert_eq!(SourceClass::of_tool("web_search"), SourceClass::External);
        assert_eq!(SourceClass::of_tool("http_request"), SourceClass::External);
        assert_eq!(SourceClass::of_tool("file_read"), SourceClass::LocalFile);
        assert_eq!(
            SourceClass::of_tool("file_write"),
            SourceClass::SelfProduced
        );
        assert_eq!(
            SourceClass::of_tool("memory_store"),
            SourceClass::SelfProduced
        );
        // shell 无法确定是否真的取了外部数据 → 保守归为自产，绝不冒充外部依据
        assert_eq!(SourceClass::of_tool("shell"), SourceClass::SelfProduced);
        // 没见过的工具同样保守
        assert_eq!(
            SourceClass::of_tool("some_new_tool"),
            SourceClass::SelfProduced
        );
    }

    #[test]
    fn eid_is_not_guessable_from_index_alone() {
        // 证据编号带内容哈希后缀：模型没读过真实输出就编不出能解析的引用
        let a = sha256_hex("PubMed 返回 A");
        let b = sha256_hex("PubMed 返回 B");
        assert_ne!(&a[..6], &b[..6]);
    }

    #[test]
    fn availability_states_the_reason_instead_of_pretending_empty() {
        let p = std::path::Path::new("/nonexistent/trace.jsonl");
        let a = availability(p, false);
        assert!(!a.available);
        assert!(a.reason.unwrap().contains("runtime_trace_mode"));
        let b = availability(p, true);
        assert!(!b.available);
        assert!(b.reason.is_some());
        // 防篡改尚未实施，任何时候都不许自称具备
        assert!(!availability(p, true).tamper_evident);
    }

    #[test]
    fn path_arg_reads_path_keys_only() {
        assert_eq!(
            path_arg(r##"{"content":"# 报告","path":"case_library/a/report_final.md"}"##)
                .as_deref(),
            Some("case_library/a/report_final.md")
        );
        assert_eq!(path_arg(r#"{"file_path":"x.md"}"#).as_deref(), Some("x.md"));
        assert_eq!(path_arg(r#"{"command":"ls"}"#), None);
        assert_eq!(path_arg("not json"), None);
    }

    #[test]
    fn excerpt_marks_truncation() {
        let long = "x".repeat(1000);
        let e = excerpt(&long, 10);
        assert!(e.contains("已截断"));
        assert_eq!(excerpt("短", 10), "短");
    }
}
