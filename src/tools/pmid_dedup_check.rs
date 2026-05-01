//! PMID dedup checker tool.
//!
//! Scans markdown files for PubMed citation reuse patterns flagged by
//! medical reviewers as "real PMID supporting two distinct claims" — a
//! more insidious variant of fabricated citation that LLMs naturally
//! produce when they retrieve one paper and stretch it across multiple
//! hypotheses.
//!
//! ## What it detects
//!
//! 1. The same PMID appearing under ≥2 distinct headings (where heading
//!    is the nearest preceding `##` / `###` line). Cross-section reuse
//!    is the strongest signal that a single paper is being made to
//!    bear weight for unrelated claims.
//! 2. The same PMID appearing ≥3 times in a document — even within the
//!    same section, heavy reuse suggests over-leaning on a single
//!    source.
//!
//! ## What it does NOT detect
//!
//! - Whether the PMID actually supports the cited claim (the
//!   "claim-PMID semantic match" check that reviewers also flagged).
//!   That requires fetching the abstract and running an LLM-as-judge,
//!   which is out of scope for this deterministic tool. The SOP
//!   step 7 prompt asks the LLM to do claim self-attribution
//!   ("write the conclusion of this paper and how it supports the
//!   claim") which catches ~80% of mismatch cases. This tool catches
//!   the structural reuse that prompt self-attribution misses.

use std::collections::HashMap;
use std::fmt::Write;
use std::path::PathBuf;

use async_trait::async_trait;
use regex::Regex;
use serde_json::json;
use std::sync::OnceLock;

use super::traits::{Tool, ToolResult};

/// Matches `PMID: 12345678`, `PMID:12345678`, `PMID 12345678`, also
/// `pubmed/12345678` and bare `(12345678)` are excluded — too many
/// false positives. We only match the explicit `PMID` prefix.
fn pmid_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)PMID[:\s]\s*(\d{7,9})").expect("static regex compiles")
    })
}

/// Matches markdown headings (`#` through `######`).
fn heading_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(#{1,6})\s+(.+?)\s*$").expect("static regex compiles"))
}

/// One occurrence of a PMID in a document.
#[derive(Debug, Clone)]
struct PmidOccurrence {
    line_no: usize,
    section: String,
    /// Up to ~140 chars of surrounding context, for the LLM to judge
    /// whether the reuse is legitimate.
    context: String,
}

pub struct PmidDedupCheckTool {
    workspace_dir: PathBuf,
}

impl PmidDedupCheckTool {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }

    /// Resolve a user-supplied path against the workspace dir, rejecting
    /// any attempt to escape the workspace via `..` or absolute paths.
    fn resolve(&self, rel: &str) -> anyhow::Result<PathBuf> {
        if rel.contains("..") {
            anyhow::bail!("path traversal denied: {rel}");
        }
        let candidate = if rel.starts_with('/') {
            PathBuf::from(rel)
        } else {
            self.workspace_dir.join(rel)
        };
        let canonical = candidate
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("cannot canonicalize {rel}: {e}"))?;
        let workspace_canonical = self
            .workspace_dir
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("cannot canonicalize workspace: {e}"))?;
        if !canonical.starts_with(&workspace_canonical) {
            anyhow::bail!("path escapes workspace: {rel}");
        }
        Ok(canonical)
    }

    fn scan_text(text: &str) -> HashMap<String, Vec<PmidOccurrence>> {
        let mut current_section = String::from("(no heading)");
        let mut by_pmid: HashMap<String, Vec<PmidOccurrence>> = HashMap::new();

        let pmid_re = pmid_regex();
        let heading_re = heading_regex();

        for (idx, line) in text.lines().enumerate() {
            if let Some(caps) = heading_re.captures(line) {
                if let Some(title) = caps.get(2) {
                    current_section = title.as_str().trim().to_string();
                }
                continue;
            }
            for caps in pmid_re.captures_iter(line) {
                let Some(pmid_match) = caps.get(1) else {
                    continue;
                };
                let pmid = pmid_match.as_str().to_string();
                let trimmed = line.trim();
                let context = if trimmed.len() <= 140 {
                    trimmed.to_string()
                } else {
                    format!("{}…", &trimmed[..140])
                };
                by_pmid
                    .entry(pmid)
                    .or_default()
                    .push(PmidOccurrence {
                        line_no: idx + 1,
                        section: current_section.clone(),
                        context,
                    });
            }
        }

        by_pmid
    }

    fn render_report(file_label: &str, by_pmid: &HashMap<String, Vec<PmidOccurrence>>) -> String {
        let total_pmids = by_pmid.len();
        let total_occurrences: usize = by_pmid.values().map(|v| v.len()).sum();

        let mut cross_section: Vec<(&String, &Vec<PmidOccurrence>)> = Vec::new();
        let mut heavy_reuse: Vec<(&String, &Vec<PmidOccurrence>)> = Vec::new();

        for (pmid, occs) in by_pmid {
            let distinct_sections: std::collections::HashSet<&str> =
                occs.iter().map(|o| o.section.as_str()).collect();
            if distinct_sections.len() >= 2 {
                cross_section.push((pmid, occs));
            }
            if occs.len() >= 3 {
                heavy_reuse.push((pmid, occs));
            }
        }

        cross_section.sort_by_key(|(_, occs)| std::cmp::Reverse(occs.len()));
        heavy_reuse.sort_by_key(|(_, occs)| std::cmp::Reverse(occs.len()));

        let mut out = String::new();
        let _ = writeln!(out, "# PMID Dedup Check — {file_label}\n");
        let _ = writeln!(out, "- Distinct PMIDs: **{total_pmids}**");
        let _ = writeln!(out, "- Total occurrences: **{total_occurrences}**");
        let _ = writeln!(
            out,
            "- Cross-section reuse (PMID under ≥2 headings): **{}**",
            cross_section.len()
        );
        let _ = writeln!(
            out,
            "- Heavy reuse (PMID cited ≥3 times): **{}**\n",
            heavy_reuse.len()
        );

        if cross_section.is_empty() && heavy_reuse.is_empty() {
            let _ = writeln!(
                out,
                "✅ No structural PMID reuse patterns detected. \
                 (Note: this tool does NOT validate claim-PMID semantic match — \
                 SOP step 7 self-attribution prompt covers that.)"
            );
            return out;
        }

        if !cross_section.is_empty() {
            let _ = writeln!(
                out,
                "## ⚠️ Cross-section reuse (P0 — same PMID supporting claims under different headings)\n"
            );
            let _ = writeln!(
                out,
                "Each PMID below appears under **multiple distinct headings**. \
                 SOP step 7 PM 审核 must verify each occurrence independently \
                 passes claim-PMID self-attribution. If any one cannot, replace it.\n"
            );
            for (pmid, occs) in cross_section {
                let _ = writeln!(out, "### PMID:{pmid} ({} occurrences)\n", occs.len());
                for occ in occs {
                    let _ = writeln!(
                        out,
                        "- L{} · `{}` — {}",
                        occ.line_no, occ.section, occ.context
                    );
                }
                let _ = writeln!(out);
            }
        }

        if !heavy_reuse.is_empty() {
            let _ = writeln!(
                out,
                "## ⚠️ Heavy reuse (P1 — PMID cited ≥3 times)\n"
            );
            for (pmid, occs) in heavy_reuse {
                let sections: std::collections::HashSet<&str> =
                    occs.iter().map(|o| o.section.as_str()).collect();
                let _ = writeln!(
                    out,
                    "- PMID:{pmid} — {} occurrences across {} section(s)",
                    occs.len(),
                    sections.len()
                );
            }
        }

        out
    }
}

#[async_trait]
impl Tool for PmidDedupCheckTool {
    fn name(&self) -> &str {
        "pmid_dedup_check"
    }

    fn description(&self) -> &str {
        "Detect PMID reuse patterns in a markdown file: same PMID supporting claims under multiple headings (P0), or PMID cited ≥3 times (P1). \
         Algorithmic check, no LLM. Output is a markdown report listing each flagged PMID with line numbers, sections, and surrounding context. \
         SOP step 7 PM 审核 should call this on the assembled report and check 5.3 (PMID 复用检测). \
         Does NOT validate claim-PMID semantic match — that is covered by step 7 self-attribution prompt."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the markdown file to scan, relative to the workspace root (e.g., 'case_library/insight/zhang_doctor/2026-05-01_proposal.md')."
                }
            },
            "required": ["file_path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let file_path = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: file_path"))?;

        let resolved = match self.resolve(file_path) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("path resolution failed: {e}")),
                });
            }
        };

        let text = match std::fs::read_to_string(&resolved) {
            Ok(t) => t,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("cannot read {file_path}: {e}")),
                });
            }
        };

        let by_pmid = Self::scan_text(&text);
        let report = Self::render_report(file_path, &by_pmid);

        Ok(ToolResult {
            success: true,
            output: report,
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmid_regex_matches_canonical_forms() {
        let re = pmid_regex();
        assert!(re.is_match("PMID: 12345678"));
        assert!(re.is_match("PMID:12345678"));
        assert!(re.is_match("PMID 12345678"));
        assert!(re.is_match("pmid: 12345678"));
        // bare 8-digit number should NOT match (too many false positives)
        assert!(!re.is_match("see 12345678 in the table"));
        // PMC IDs should not match the same pattern
        assert!(!re.is_match("PMC12345678"));
    }

    #[test]
    fn heading_regex_extracts_title() {
        let re = heading_regex();
        let caps = re.captures("## 假说 1: TME 重塑").unwrap();
        assert_eq!(caps.get(2).unwrap().as_str(), "假说 1: TME 重塑");
    }

    #[test]
    fn scan_text_groups_by_pmid_and_tracks_section() {
        let text = "\
## Section A
First reference is PMID: 11111111 here.
And another mention of PMID:11111111 in same section.

## Section B
Now PMID 11111111 appears under a different heading.
Unique to B is PMID: 22222222.
";
        let by_pmid = PmidDedupCheckTool::scan_text(text);

        assert_eq!(by_pmid.len(), 2);
        let pmid_a = by_pmid.get("11111111").unwrap();
        assert_eq!(pmid_a.len(), 3);
        let sections: std::collections::HashSet<&str> =
            pmid_a.iter().map(|o| o.section.as_str()).collect();
        assert_eq!(sections.len(), 2);
        assert!(sections.contains("Section A"));
        assert!(sections.contains("Section B"));

        let pmid_b = by_pmid.get("22222222").unwrap();
        assert_eq!(pmid_b.len(), 1);
        assert_eq!(pmid_b[0].section, "Section B");
    }

    #[test]
    fn render_report_flags_cross_section_reuse() {
        let mut by_pmid = HashMap::new();
        by_pmid.insert(
            "11111111".to_string(),
            vec![
                PmidOccurrence {
                    line_no: 5,
                    section: "假说 1".into(),
                    context: "支持 TME 重塑 (PMID:11111111)".into(),
                },
                PmidOccurrence {
                    line_no: 30,
                    section: "假说 3".into(),
                    context: "也用于 FAP+ CAF (PMID:11111111)".into(),
                },
            ],
        );
        let report = PmidDedupCheckTool::render_report("test.md", &by_pmid);
        assert!(report.contains("Cross-section reuse"));
        assert!(report.contains("PMID:11111111"));
        assert!(report.contains("假说 1"));
        assert!(report.contains("假说 3"));
    }

    #[test]
    fn render_report_clean_when_no_reuse() {
        let mut by_pmid = HashMap::new();
        by_pmid.insert(
            "11111111".to_string(),
            vec![PmidOccurrence {
                line_no: 5,
                section: "假说 1".into(),
                context: "...".into(),
            }],
        );
        let report = PmidDedupCheckTool::render_report("test.md", &by_pmid);
        assert!(report.contains("✅ No structural PMID reuse"));
    }
}
