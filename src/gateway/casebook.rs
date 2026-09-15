//! 结论的可核对性（前台「任务档案」的服务端一半）。
//!
//! 原则只有一条：**界面上每一个「已核」都必须由运行时算出来，不能出自模型之手。**
//! 模型写的解释再流畅也只是它的说法；可信的是可查的结构——
//!
//! - 引用核验：报告里的每个 PMID，看它是否出现在本单**运行时亲自执行并留存**的外部调用返回里
//!   （`http_request` 等 external 工具；主机受 `allowed_domains` 约束）。出现 = 已核，否则未核。
//!   模型自己写脚本经 shell 取回的内容不算——运行时无法确认脚本真的访问了哪里。
//! - 人工决定留痕：批准 / 驳回 / 停止都落盘到 `state/decisions.jsonl`，连同**决定时的核验状况**
//!   （已核几项、共几项）。对临床报告，这条记录比报告本身更重要。
//! - 结构化结论：规程产出 `conclusion.json`；这里逐条标注依据核验结果与「绑定」是否成立——
//!   标为指南 / 研究的建议必须至少挂一条已核依据，否则在界面上作为违规显示，而不是被悄悄接受。
//! - 快环：逐条「认可 / 不认同」与对「助手不确定的」问题的回答，落盘到 `feedback/verdicts.jsonl`，
//!   是评测样本与规程修订的原料。

use crate::ledger::{Evidence, SourceClass};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// 报告里引用的一篇文献的核验结果
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Citation {
    pub pmid: String,
    pub verified: bool,
    /// 核到它的那次调用
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieved_at: Option<String>,
    /// 取回内容里的标题（不是模型写的标题）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// 文本里的 PMID（按出现顺序去重）。只认明确写了「PMID」的，避免把日期、编号当成文献。
pub fn extract_pmids(text: &str) -> Vec<String> {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"PMID[:：\s]*((?:\d{6,9}(?:\s*[；;,，、/]\s*(?:PMID[:：\s]*)?)?)+)")
            .unwrap()
    });
    static NUM: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let num = NUM.get_or_init(|| regex::Regex::new(r"\d{6,9}").unwrap());
    let mut out: Vec<String> = Vec::new();
    for cap in re.captures_iter(text) {
        for m in num.find_iter(&cap[1]) {
            let p = m.as_str().to_string();
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// 输出里是否以「独立数字」出现了这个 PMID（避免 12345678 命中 123456789 的一部分）
fn mentions(output: &str, pmid: &str) -> bool {
    output.match_indices(pmid).any(|(i, _)| {
        let before = output[..i].chars().next_back();
        let after = output[i + pmid.len()..].chars().next();
        !before.is_some_and(|c| c.is_ascii_digit()) && !after.is_some_and(|c| c.is_ascii_digit())
    })
}

/// 从取回内容里找这篇的标题：esummary 的 JSON（result.<pmid>.title），或 efetch 的 MEDLINE 文本（TI  - ）
fn title_in(output: &str, pmid: &str) -> Option<String> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(output) {
        if let Some(t) = v
            .pointer(&format!("/result/{pmid}/title"))
            .and_then(|t| t.as_str())
        {
            return Some(t.trim().to_string());
        }
    }
    // MEDLINE 格式：同一条记录里 PMID- 在前，TI  - 在后
    let start = output.find(&format!("PMID- {pmid}"))?;
    let rest = &output[start..];
    let ti = rest.find("\nTI  - ")?;
    let mut title = String::new();
    for (i, line) in rest[ti + 1..].lines().enumerate() {
        if i == 0 {
            title.push_str(line.trim_start_matches("TI  - ").trim());
        } else if let Some(cont) = line.strip_prefix("      ") {
            title.push(' ');
            title.push_str(cont.trim());
        } else {
            break;
        }
    }
    (!title.is_empty()).then_some(title)
}

/// 逐个 PMID 核验：只有**成功的外部调用**的完整返回里出现了它，才算已核。
pub fn verify_citations(pmids: &[String], evidence: &[(Evidence, String)]) -> Vec<Citation> {
    pmids
        .iter()
        .map(|pmid| {
            let hit = evidence.iter().find(|(e, out)| {
                e.source_class == SourceClass::External && e.success && mentions(out, pmid)
            });
            match hit {
                Some((e, out)) => Citation {
                    pmid: pmid.clone(),
                    verified: true,
                    eid: Some(e.eid.clone()),
                    source_url: e.url.clone(),
                    retrieved_at: Some(e.at.clone()),
                    title: evidence
                        .iter()
                        .filter(|(e2, _)| e2.source_class == SourceClass::External && e2.success)
                        .find_map(|(_, o)| title_in(o, pmid))
                        .or_else(|| title_in(out, pmid)),
                },
                None => Citation {
                    pmid: pmid.clone(),
                    verified: false,
                    eid: None,
                    source_url: None,
                    retrieved_at: None,
                    title: None,
                },
            }
        })
        .collect()
}

/// 这一单的正式报告：优先 final，其次 draft，其余 .md 里最后写的那份
pub fn pick_report(paths: &[(String, PathBuf)]) -> Option<&(String, PathBuf)> {
    let md: Vec<&(String, PathBuf)> = paths
        .iter()
        .filter(|(rel, _)| rel.ends_with(".md") && rel.contains("report"))
        .collect();
    md.iter()
        .rev()
        .find(|(rel, _)| rel.contains("final"))
        .or_else(|| md.iter().rev().find(|(rel, _)| rel.contains("draft")))
        .or_else(|| md.last())
        .copied()
}

/// 结构化结论里建议的「绑定」判定。
/// 指南 / 研究：至少一条已核依据才成立；推断 / 假设：本就是助手的判断，允许没有依据（但界面要显眼标出）。
fn binding_of(kind: &str, statuses: &[(String, bool)]) -> &'static str {
    let any_verified = statuses.iter().any(|(_, v)| *v);
    match kind {
        "指南" | "研究" if any_verified => "ok",
        "指南" | "研究" if statuses.is_empty() => "no_evidence",
        "指南" | "研究" => "unverified",
        "推断" | "假设" => "judgement",
        _ => "unknown_kind",
    }
}

/// 给结构化结论逐条挂上核验结果（运行时算的，覆盖模型可能自带的任何「已核」字样）。
pub fn annotate_conclusion(
    mut c: serde_json::Value,
    evidence: &[(Evidence, String)],
) -> serde_json::Value {
    let annotate_list = |items: Option<&mut Vec<serde_json::Value>>, kind_key: Option<&str>| {
        let Some(items) = items else {
            return (0usize, 0usize);
        };
        let mut violations = 0usize;
        let mut total = 0usize;
        for item in items.iter_mut() {
            let pmids: Vec<String> = item
                .get("pmids")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|p| {
                            p.as_str().map(|s| {
                                s.trim()
                                    .trim_start_matches("PMID")
                                    .trim_start_matches([':', '：', ' '])
                                    .to_string()
                            })
                        })
                        .filter(|p| !p.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let checked = verify_citations(&pmids, evidence);
            let statuses: Vec<(String, bool)> = checked
                .iter()
                .map(|c| (c.pmid.clone(), c.verified))
                .collect();
            if let Some(obj) = item.as_object_mut() {
                obj.remove("verified"); // 模型自带的核验字样一律不作数
                obj.insert(
                    "evidence_status".into(),
                    serde_json::to_value(&checked).unwrap_or_default(),
                );
                if let Some(k) = kind_key {
                    let kind = obj
                        .get(k)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let b = binding_of(&kind, &statuses);
                    if b == "unverified" || b == "no_evidence" || b == "unknown_kind" {
                        violations += 1;
                    }
                    obj.insert("binding".into(), b.into());
                }
            }
            total += 1;
        }
        (violations, total)
    };
    let (rec_violations, rec_total) = annotate_list(
        c.get_mut("recommendations").and_then(|v| v.as_array_mut()),
        Some("kind"),
    );
    annotate_list(c.get_mut("rules").and_then(|v| v.as_array_mut()), None);
    if let Some(obj) = c.as_object_mut() {
        obj.insert(
            "checks".into(),
            serde_json::json!({ "recommendations": rec_total, "binding_violations": rec_violations }),
        );
    }
    c
}

// ── 人工决定留痕 ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct DecisionBasis {
    pub citations_total: usize,
    pub citations_verified: usize,
    pub external_calls: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_sha256: Option<String>,
}

pub fn decisions_path(workspace: &Path) -> PathBuf {
    workspace.join("state").join("decisions.jsonl")
}

pub fn verdicts_path(workspace: &Path) -> PathBuf {
    workspace.join("feedback").join("verdicts.jsonl")
}

/// 凭证的短指纹：能区分是哪台已配对设备做的决定，又不泄露凭证本身
pub fn device_fingerprint(bearer: &str) -> Option<String> {
    use sha2::Digest;
    let t = bearer.trim();
    if t.is_empty() {
        return None;
    }
    let mut h = sha2::Sha256::new();
    h.update(t.as_bytes());
    Some(format!("{:x}", h.finalize())[..8].to_string())
}

pub fn append_jsonl(path: &Path, value: &serde_json::Value) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    std::io::Write::write_all(&mut f, format!("{value}\n").as_bytes())
}

pub fn read_jsonl_for_run(path: &Path, run_id: &str) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .map(|raw| {
            raw.lines()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .filter(|v| v.get("run_id").and_then(|r| r.as_str()) == Some(run_id))
                .collect()
        })
        .unwrap_or_default()
}

/// 这一条（同一建议或同一问题）最新的一次记录，若与本次完全相同就返回它（调用方据此不再追加）。
/// 认可 / 不认同 / 撤回 共用「建议」这一个槽位；回答按问题各占一个槽位。
pub fn latest_verdict_if_same(
    existing: &[serde_json::Value],
    kind: &str,
    rec_id: Option<&str>,
    qid: Option<&str>,
    text: Option<&str>,
) -> Option<serde_json::Value> {
    let slot = |v: &serde_json::Value| -> Option<String> {
        match v.get("kind").and_then(|k| k.as_str())? {
            "answer" => v
                .get("qid")
                .and_then(|q| q.as_str())
                .map(|q| format!("q:{q}")),
            _ => v
                .get("rec_id")
                .and_then(|r| r.as_str())
                .map(|r| format!("r:{r}")),
        }
    };
    let want = match kind {
        "answer" => format!("q:{}", qid?),
        _ => format!("r:{}", rec_id?),
    };
    let last = existing
        .iter()
        .rev()
        .find(|v| slot(v).as_deref() == Some(&want))?;
    let same_kind = last.get("kind").and_then(|k| k.as_str()) == Some(kind);
    let same_text = last.get("text").and_then(|t| t.as_str()) == text;
    (same_kind && same_text).then(|| last.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(tool: &str, class: SourceClass, ok: bool, out: &str) -> (Evidence, String) {
        (
            Evidence {
                eid: format!("E1-{}", "0".repeat(6)),
                n: 1,
                at: "2026-09-15T02:00:00Z".into(),
                tool: tool.into(),
                source_class: class,
                args_excerpt: None,
                output_excerpt: String::new(),
                output_bytes: out.len(),
                sha256: "0".repeat(64),
                success: ok,
                turn_id: "t".into(),
                path: None,
                url: Some(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?id=32827126"
                        .into(),
                ),
            },
            out.to_string(),
        )
    }

    #[test]
    fn pmids_are_extracted_in_order_including_runs() {
        let t = "见 PMID: 32827126；28372962，另 PMID 32827126 与 PMID：35646667/36039947。日期 20260915 不算";
        assert_eq!(
            extract_pmids(t),
            vec!["32827126", "28372962", "35646667", "36039947"]
        );
    }

    #[test]
    fn only_successful_external_returns_verify_a_citation() {
        let esummary = r#"{"result":{"uids":["32827126"],"32827126":{"uid":"32827126","title":"2020 Chinese guidelines for ultrasound malignancy risk stratification of thyroid nodules: the C-TIRADS."}}}"#;
        let evidence = vec![
            ev(
                "shell",
                SourceClass::SelfProduced,
                true,
                "PMID 28372962 found by my script",
            ),
            ev("http_request", SourceClass::External, false, "28372962"),
            ev("http_request", SourceClass::External, true, esummary),
        ];
        let c = verify_citations(
            &["32827126".into(), "28372962".into(), "3282712".into()],
            &evidence,
        );
        assert!(c[0].verified);
        assert!(c[0].title.as_deref().unwrap().contains("C-TIRADS"));
        assert!(!c[1].verified, "shell 输出与失败的调用都不能核验");
        assert!(!c[2].verified, "数字的一部分不能算命中");
    }

    #[test]
    fn medline_title_is_read_from_efetch_text() {
        let medline = "PMID- 28372962\nOWN - NLM\nTI  - ACR Thyroid Imaging, Reporting and Data System (TI-RADS): White\n      Paper of the ACR TI-RADS Committee.\nPG  - 587-595\n";
        assert_eq!(
            title_in(medline, "28372962").as_deref(),
            Some("ACR Thyroid Imaging, Reporting and Data System (TI-RADS): White Paper of the ACR TI-RADS Committee.")
        );
    }

    #[test]
    fn guideline_claims_without_verified_evidence_are_flagged_and_model_badges_ignored() {
        let esummary = r#"{"result":{"32827126":{"title":"C-TIRADS"}}}"#;
        let evidence = vec![ev("http_request", SourceClass::External, true, esummary)];
        let c = serde_json::json!({
            "recommendations": [
                {"id":"R1","kind":"指南","pmids":["32827126"],"verified":true},
                {"id":"R2","kind":"研究","pmids":["26462967"],"verified":true},
                {"id":"R3","kind":"研究","pmids":[]},
                {"id":"R4","kind":"推断","pmids":[]}
            ],
            "rules": [{"system":"C-TIRADS","pmids":["PMID: 32827126"]}]
        });
        let a = annotate_conclusion(c, &evidence);
        let recs = a["recommendations"].as_array().unwrap();
        assert_eq!(recs[0]["binding"], "ok");
        assert_eq!(recs[1]["binding"], "unverified");
        assert_eq!(recs[2]["binding"], "no_evidence");
        assert_eq!(recs[3]["binding"], "judgement");
        assert!(recs[1].get("verified").is_none(), "模型自称的核验不作数");
        assert_eq!(a["checks"]["binding_violations"], 2);
        assert_eq!(a["rules"][0]["evidence_status"][0]["verified"], true);
    }

    #[test]
    fn repeated_clicks_do_not_pile_up_but_changes_are_kept() {
        let v = |kind: &str, rec: Option<&str>, q: Option<&str>, text: Option<&str>| serde_json::json!({"kind": kind, "rec_id": rec, "qid": q, "text": text});
        let log = vec![
            v("agree", Some("R1"), None, None),
            v("answer", None, Some("Q1"), Some("C-TIRADS")),
        ];
        assert!(latest_verdict_if_same(&log, "agree", Some("R1"), None, None).is_some());
        assert!(
            latest_verdict_if_same(&log, "answer", None, Some("Q1"), Some("C-TIRADS")).is_some()
        );
        // 改了主意：照常记
        assert!(
            latest_verdict_if_same(&log, "answer", None, Some("Q1"), Some("ACR TI-RADS")).is_none()
        );
        assert!(latest_verdict_if_same(&log, "disagree", Some("R1"), None, Some("不对")).is_none());
        // 别的建议、别的问题：互不相干
        assert!(latest_verdict_if_same(&log, "agree", Some("R2"), None, None).is_none());
        let mut log2 = log.clone();
        log2.push(v("retract", Some("R1"), None, None));
        assert!(
            latest_verdict_if_same(&log2, "agree", Some("R1"), None, None).is_none(),
            "撤回后再认可要记"
        );
    }

    #[test]
    fn report_prefers_final_then_draft() {
        let p = |s: &str| (s.to_string(), PathBuf::from(s));
        let list = vec![
            p("c/case_summary.md"),
            p("c/report_draft.md"),
            p("c/report_final.md"),
        ];
        assert_eq!(pick_report(&list).unwrap().0, "c/report_final.md");
        let list = vec![p("c/report_draft.md"), p("c/literature_notes.md")];
        assert_eq!(pick_report(&list).unwrap().0, "c/report_draft.md");
    }
}
