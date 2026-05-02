//! PI publication-attribution validator.
//!
//! Filters a JSON file of PubMed papers (typically the
//! `pi_recent_papers.json` produced by the PharmaClaw SOP step 1
//! pubmed-search SKILL script) so that only entries where the named
//! PI actually occupies a primary author position survive.
//!
//! ## What it catches
//!
//! Live observation on PharmaClaw v3 (`run-1777680842661-0001`):
//! the LLM listed "ICARUS-LUNG01 (Cancer Cell 2026)" as a张力 (Zhang Li)
//! representative paper on the strength of topic similarity. Medical
//! reviewer flagged it as **academic-integrity-grade** misattribution
//! — that trial (NCT04940325) is led by David Planchard at Gustave
//! Roussy with AstraZeneca / Daiichi Sankyo as sponsors, not by Zhang
//! Li. The LLM had pulled the entry into the JSON (it was a
//! co-authored commentary), then promoted it to "Zhang Li
//! representative trial" in the report.
//!
//! This tool is the deterministic gate between
//! "PMID is in pi_recent_papers.json" and
//! "PMID can appear in pi_intel.md as a representative paper".
//!
//! ## Acceptance criteria (configurable, default = strict)
//!
//! A paper is **accepted** if any of the following is true:
//!  - `is_first_author == true` (PI is sole or co-first author)
//!  - `is_corresponding == true` (PI is corresponding / national lead)
//!  - `last_author == <pi_name>` (PI in last-author / senior position)
//!
//! Anything else — including "PI in author list at position N" with N
//! between 2 and (last - 1) — is **rejected**, even if the paper is
//! topically perfect.
//!
//! ## What it does NOT catch
//!
//! Paper *type*. If a commentary written by the PI cites a trial led
//! by someone else, this tool will pass it (the PI really is the
//! commentary's senior author). The downstream prompt (SOP step 1
//! P0-5) is responsible for ensuring the LLM's narrative around the
//! citation does not falsely promote a commentary to "PI-led trial".
//! A future `publication_type` filter could check PubMed
//! `PublicationTypeList` for "Editorial" / "Comment" / "Review" /
//! "Letter" tags, but that's out of scope for this MVP.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::traits::{Tool, ToolResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaperRecord {
    #[serde(default)]
    pmid: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    journal: String,
    #[serde(default)]
    year: String,
    #[serde(default)]
    first_author: String,
    #[serde(default)]
    last_author: String,
    #[serde(default)]
    is_first_author: bool,
    #[serde(default)]
    is_corresponding: bool,
    #[serde(default)]
    total_authors: u32,
    /// Raw author list as returned by PubMed eFetch. Used as a fallback
    /// when the LLM-written pubmed-search script doesn't pre-compute
    /// `is_first_author` / `is_corresponding` / `last_author` flags
    /// (observed live on V6: emily Shao Zhimin's `pi_recent_papers.json`
    /// only had `authors[]` raw list, no per-paper position flags, so the
    /// validator originally fell through to "PI not in primary author
    /// position" — but worse, the LLM then read the raw `authors[]` and
    /// fabricated "Shao ZM (last author)" against PubMed's actual data).
    /// When this field is present and the structured flags above are all
    /// default (false / empty), the validator computes positions from
    /// this raw list itself.
    #[serde(default)]
    authors: Vec<RawAuthor>,
}

/// Raw author entry shape emitted by the pubmed-search SKILL when the
/// script doesn't normalise author position. Tolerant: `last` and
/// `full` are optional, and unknown extra fields are accepted.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RawAuthor {
    #[serde(default)]
    last: String,
    #[serde(default)]
    first: String,
    #[serde(default)]
    full: String,
    #[serde(default)]
    affiliation: String,
}

#[derive(Debug, Clone, Serialize)]
struct PaperVerdict {
    pmid: String,
    title: String,
    journal: String,
    year: String,
    /// Compact author-position label for the LLM to copy directly into
    /// `pi_intel.md` (e.g. "first-author", "co-first / national lead PI",
    /// "last-author / corresponding").
    author_position: String,
}

#[derive(Debug, Clone, Serialize)]
struct RejectedPaper {
    pmid: String,
    title: String,
    reason: String,
}

pub struct PiPublicationValidatorTool {
    workspace_dir: PathBuf,
}

impl PiPublicationValidatorTool {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }

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

    /// Heuristic: does `last_author` look like the same person as
    /// `pi_name`? PubMed author lists are rendered "Last F" (e.g.
    /// "Zhang Li") while PharmaClaw triggers may pass the family name
    /// in either order, English or Chinese. We accept any of:
    ///  - exact case-insensitive match
    ///  - `pi_name` is a single token contained in `last_author`
    ///  - first whitespace-separated token of each side matches
    ///    case-insensitively (catches "Zhang Li" vs "Zhang L")
    fn last_author_matches(last_author: &str, pi_name: &str) -> bool {
        if last_author.is_empty() || pi_name.is_empty() {
            return false;
        }
        let a = last_author.trim().to_lowercase();
        let b = pi_name.trim().to_lowercase();
        if a == b {
            return true;
        }
        if a.contains(&b) || b.contains(&a) {
            return true;
        }
        // Compare first tokens (typically family name in English)
        let a_first = a.split_whitespace().next().unwrap_or("");
        let b_first = b.split_whitespace().next().unwrap_or("");
        !a_first.is_empty() && !b_first.is_empty() && a_first == b_first
    }

    /// Match a raw PubMed author entry against `pi_name`. PubMed
    /// renders author objects with separate `last`, `first`, and a
    /// composed `full` field (e.g. `last="Zhang"`, `first="Li"`,
    /// `full="Zhang Li"`); we accept any of these matching the PI
    /// name token-wise. Affiliation is intentionally NOT used —
    /// V6's "Shao Zhiming" mis-attribution was partly because the
    /// `corresponding=true` heuristic keyed off affiliation text
    /// (which contained Shao's institution as one of many).
    fn raw_author_matches(author: &RawAuthor, pi_name: &str) -> bool {
        if pi_name.is_empty() {
            return false;
        }
        // Try `full` first, then `last + first`, then `last` alone.
        if !author.full.is_empty() && Self::last_author_matches(&author.full, pi_name) {
            return true;
        }
        if !author.last.is_empty() {
            if !author.first.is_empty() {
                let combined = format!("{} {}", author.last, author.first);
                if Self::last_author_matches(&combined, pi_name) {
                    return true;
                }
            }
            if Self::last_author_matches(&author.last, pi_name) {
                return true;
            }
        }
        false
    }

    /// True when *no* structured author-position flags are populated.
    /// In that case `classify` falls back to scanning `paper.authors`
    /// directly.
    fn structured_fields_empty(paper: &PaperRecord) -> bool {
        !paper.is_first_author
            && !paper.is_corresponding
            && paper.last_author.is_empty()
            && paper.first_author.is_empty()
    }

    fn classify(paper: &PaperRecord, pi_name: &str) -> Result<PaperVerdict, String> {
        let mut positions: Vec<&str> = Vec::new();

        // Primary path: structured fields populated by the pubmed-search
        // script (the V5 张力 case did this correctly).
        if paper.is_first_author {
            positions.push("first-author");
        }
        if paper.is_corresponding {
            positions.push("corresponding");
        }
        if Self::last_author_matches(&paper.last_author, pi_name) {
            positions.push("last-author");
        }
        // V7 fallback: when the script didn't pre-compute position flags
        // (V6 邵志敏 case shipped raw `authors[]` only), classify directly
        // from the raw list. This closes the path that let the LLM read
        // `authors[]` and fabricate a "last author" claim that
        // contradicted PubMed's actual ordering (DESTINY-Breast05
        // PMID:41370739: Shao Zhiming was 3rd of ~70, NOT last).
        if positions.is_empty()
            && Self::structured_fields_empty(paper)
            && !paper.authors.is_empty()
        {
            let n = paper.authors.len();
            if n >= 1 && Self::raw_author_matches(&paper.authors[0], pi_name) {
                positions.push("first-author (from raw authors[])");
            }
            if n >= 2 && Self::raw_author_matches(&paper.authors[n - 1], pi_name) {
                positions.push("last-author (from raw authors[])");
            }
            // Co-first detection: if PI is at position 0 and the next
            // author is also explicitly tagged co-first in PubMed, we'd
            // need an `is_co_first` flag — out of scope here. Position 0
            // alone covers 99% of "first-author" attribution.
        }

        if positions.is_empty() {
            // If we have a non-empty authors list but PI didn't match
            // first/last, surface the actual middle position so the LLM
            // can't claim "last author" — it will see exactly where the
            // PI sits.
            let position_hint = if paper.authors.is_empty() {
                format!(" Paper has {} total authors", paper.total_authors)
            } else {
                let n = paper.authors.len();
                paper
                    .authors
                    .iter()
                    .position(|a| Self::raw_author_matches(a, pi_name))
                    .map(|i| format!(" (PI is at position {} of {n})", i + 1))
                    .unwrap_or_else(|| format!(" (PI not found in {n}-author list)"))
            };
            Err(format!(
                "PI not in primary author position (first / corresponding / last).{position_hint} \
                 Citing this paper as 'PI representative work' would be academic-integrity-grade misattribution."
            ))
        } else {
            Ok(PaperVerdict {
                pmid: paper.pmid.clone(),
                title: paper.title.clone(),
                journal: paper.journal.clone(),
                year: paper.year.clone(),
                author_position: positions.join(" / "),
            })
        }
    }

    /// `papers_field` lets PharmaClaw point to a nested array (e.g.
    /// `"all_papers"` for the SKILL output). Defaults to the JSON root
    /// being an array.
    fn extract_papers(json: &Value, papers_field: Option<&str>) -> anyhow::Result<Vec<PaperRecord>> {
        let array = match papers_field {
            Some(field) => json
                .get(field)
                .ok_or_else(|| anyhow::anyhow!("papers_field '{field}' not found in JSON"))?,
            None => json,
        };
        let array = array
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("expected array of papers"))?;
        let mut out = Vec::with_capacity(array.len());
        for (idx, entry) in array.iter().enumerate() {
            match serde_json::from_value::<PaperRecord>(entry.clone()) {
                Ok(p) => out.push(p),
                Err(e) => {
                    anyhow::bail!("entry {idx} could not be parsed as PaperRecord: {e}");
                }
            }
        }
        Ok(out)
    }

    fn render_report(
        pi_name: &str,
        valid: &[PaperVerdict],
        rejected: &[RejectedPaper],
    ) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "# PI Publication Validator — {pi_name}\n\n\
             - Total papers in JSON: **{}**\n\
             - **Valid** (PI in primary author position): **{}**\n\
             - **Rejected** (PI not first / corresponding / last): **{}**\n",
            valid.len() + rejected.len(),
            valid.len(),
            rejected.len()
        );

        if !valid.is_empty() {
            let _ = writeln!(out, "\n## ✅ Valid representative papers (use these in `pi_intel.md`)\n");
            for p in valid {
                let _ = writeln!(
                    out,
                    "- **{}** ({}) — {}\n  - PMID:{} | author position: **{}**",
                    p.title.trim_end_matches('.'),
                    p.year,
                    p.journal,
                    p.pmid,
                    p.author_position
                );
            }
        }

        if !rejected.is_empty() {
            let _ = writeln!(
                out,
                "\n## ⚠️ Rejected (do NOT cite these as PI representative work)\n\n\
                 The following PMIDs were in the JSON because the PubMed search returned them \
                 (PI's name appears somewhere in the author list), but PI is not in a primary \
                 author position. Citing these in `pi_intel.md` as PI representative papers \
                 would be academic-integrity-grade misattribution. SOP step 7 PM 审核 5.2c \
                 will flag this.\n"
            );
            for r in rejected {
                let _ = writeln!(out, "- PMID:{} — {}\n  - reason: {}", r.pmid, r.title.trim_end_matches('.'), r.reason);
            }
        }

        out
    }
}

#[async_trait]
impl Tool for PiPublicationValidatorTool {
    fn name(&self) -> &str {
        "pi_publication_validator"
    }

    fn description(&self) -> &str {
        "Filter a JSON file of PubMed papers so that only entries where the named PI is in a primary author position survive (first author, corresponding, or last author). \
         Inputs: `json_file` (workspace-relative path to PubMed search output, e.g. `case_library/insight/<pi>/pi_recent_papers.json`), `pi_name`, optional `papers_field` (default `\"all_papers\"`). \
         Output: a markdown report with two sections — Valid (use these in pi_intel.md as representative papers) and Rejected (do NOT cite as PI representative work; doing so is academic-integrity-grade misattribution). \
         SOP step 1 P0-5 requires this tool to be called before populating the 'PI representative papers' list. \
         SOP step 7 PM 审核 5.2c re-validates the final pi_intel.md against this tool's output."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "json_file": {
                    "type": "string",
                    "description": "Workspace-relative path to the JSON file containing PubMed search results (typically `case_library/insight/<pi>/pi_recent_papers.json`)."
                },
                "pi_name": {
                    "type": "string",
                    "description": "PI's name as it appears in the PubMed author list (e.g. 'Zhang Li' or '张力')."
                },
                "papers_field": {
                    "type": "string",
                    "description": "Name of the JSON field containing the array of paper records. Default: `all_papers` (matches the pubmed-search SKILL output). Use empty string or omit if the JSON root itself is the array."
                }
            },
            "required": ["json_file", "pi_name"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let json_file = args
            .get("json_file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: json_file"))?;
        let pi_name = args
            .get("pi_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: pi_name"))?;
        let papers_field = args
            .get("papers_field")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("all_papers");

        let resolved = match self.resolve(json_file) {
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
                    error: Some(format!("cannot read {json_file}: {e}")),
                });
            }
        };

        let json: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("invalid JSON in {json_file}: {e}")),
                });
            }
        };

        let papers = match Self::extract_papers(&json, Some(papers_field)) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("cannot extract papers: {e}")),
                });
            }
        };

        let mut valid = Vec::new();
        let mut rejected = Vec::new();
        for paper in &papers {
            match Self::classify(paper, pi_name) {
                Ok(v) => valid.push(v),
                Err(reason) => rejected.push(RejectedPaper {
                    pmid: paper.pmid.clone(),
                    title: paper.title.clone(),
                    reason,
                }),
            }
        }

        let report = Self::render_report(pi_name, &valid, &rejected);

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

    fn paper(
        pmid: &str,
        first: &str,
        last: &str,
        is_first: bool,
        is_corr: bool,
    ) -> PaperRecord {
        PaperRecord {
            pmid: pmid.to_string(),
            title: format!("Paper {pmid}"),
            journal: "Test Journal".into(),
            year: "2026".into(),
            first_author: first.to_string(),
            last_author: last.to_string(),
            is_first_author: is_first,
            is_corresponding: is_corr,
            total_authors: 5,
            authors: vec![],
        }
    }

    /// Helper for raw-authors[]-only papers (v7 fallback path), where
    /// the pubmed-search script didn't pre-compute position flags.
    fn paper_raw_authors(pmid: &str, authors: Vec<&str>) -> PaperRecord {
        let n = u32::try_from(authors.len()).unwrap_or(u32::MAX);
        let raw = authors
            .into_iter()
            .map(|name| {
                let mut parts = name.splitn(2, ' ');
                let last = parts.next().unwrap_or("").to_string();
                let first = parts.next().unwrap_or("").to_string();
                RawAuthor {
                    full: name.to_string(),
                    last,
                    first,
                    affiliation: String::new(),
                }
            })
            .collect();
        PaperRecord {
            pmid: pmid.to_string(),
            title: format!("Paper {pmid}"),
            journal: "Test Journal".into(),
            year: "2026".into(),
            first_author: String::new(),
            last_author: String::new(),
            is_first_author: false,
            is_corresponding: false,
            total_authors: n,
            authors: raw,
        }
    }

    #[test]
    fn last_author_matches_handles_chinese_and_english_pi_names() {
        assert!(PiPublicationValidatorTool::last_author_matches(
            "Zhang Li", "Zhang Li"
        ));
        assert!(PiPublicationValidatorTool::last_author_matches(
            "zhang li", "Zhang Li"
        ));
        // PubMed often abbreviates given name to one letter
        assert!(PiPublicationValidatorTool::last_author_matches(
            "Zhang L", "Zhang Li"
        ));
        // family name match suffices when given names truncate
        assert!(PiPublicationValidatorTool::last_author_matches(
            "Zhang Li", "Zhang"
        ));
        assert!(!PiPublicationValidatorTool::last_author_matches(
            "Wang Y", "Zhang Li"
        ));
        assert!(!PiPublicationValidatorTool::last_author_matches("", "Zhang Li"));
    }

    #[test]
    fn classify_accepts_first_author() {
        let p = paper("12345", "Zhang Li", "Wang Y", true, false);
        let v = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap();
        assert_eq!(v.pmid, "12345");
        assert!(v.author_position.contains("first-author"));
    }

    #[test]
    fn classify_accepts_corresponding_only() {
        let p = paper("12345", "Wang Y", "Liu Z", false, true);
        let v = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap();
        assert!(v.author_position.contains("corresponding"));
    }

    #[test]
    fn classify_accepts_last_author() {
        let p = paper("12345", "Wang Y", "Zhang Li", false, false);
        let v = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap();
        assert!(v.author_position.contains("last-author"));
    }

    #[test]
    fn classify_rejects_middle_author() {
        // PI is in the author list but not at first/corresponding/last
        let p = paper("12345", "Wang Y", "Liu Z", false, false);
        let err = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap_err();
        assert!(err.contains("not in primary author position"));
    }

    #[test]
    fn classify_combines_multiple_positions() {
        // PI is both corresponding and last author (common for senior PIs)
        let p = paper("12345", "Wang Y", "Zhang Li", false, true);
        let v = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap();
        assert!(v.author_position.contains("corresponding"));
        assert!(v.author_position.contains("last-author"));
    }

    #[test]
    fn extract_papers_walks_named_field() {
        let json: Value = serde_json::json!({
            "all_papers": [
                {"pmid": "1", "first_author": "X", "last_author": "Y", "is_first_author": true, "is_corresponding": false, "total_authors": 5},
                {"pmid": "2", "first_author": "X", "last_author": "Y", "is_first_author": false, "is_corresponding": false, "total_authors": 5}
            ]
        });
        let papers = PiPublicationValidatorTool::extract_papers(&json, Some("all_papers")).unwrap();
        assert_eq!(papers.len(), 2);
        assert_eq!(papers[0].pmid, "1");
    }

    #[test]
    fn extract_papers_walks_root_array() {
        let json: Value = serde_json::json!([
            {"pmid": "1", "first_author": "X", "last_author": "Y", "is_first_author": true, "is_corresponding": false, "total_authors": 5}
        ]);
        let papers = PiPublicationValidatorTool::extract_papers(&json, None).unwrap();
        assert_eq!(papers.len(), 1);
    }

    #[test]
    fn extract_papers_errors_on_missing_field() {
        let json: Value = serde_json::json!({"some_other_key": []});
        let err = PiPublicationValidatorTool::extract_papers(&json, Some("all_papers")).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn render_report_emits_both_sections_when_mixed() {
        let valid = vec![PaperVerdict {
            pmid: "1".into(),
            title: "Good paper".into(),
            journal: "J".into(),
            year: "2026".into(),
            author_position: "last-author".into(),
        }];
        let rejected = vec![RejectedPaper {
            pmid: "2".into(),
            title: "Topic-similar but PI not primary".into(),
            reason: "PI not in primary author position".into(),
        }];
        let report = PiPublicationValidatorTool::render_report("Zhang Li", &valid, &rejected);
        assert!(report.contains("✅ Valid representative papers"));
        assert!(report.contains("⚠️ Rejected"));
        assert!(report.contains("PMID:1"));
        assert!(report.contains("PMID:2"));
        assert!(report.contains("misattribution"));
    }

    #[test]
    fn classify_v7_fallback_accepts_first_author_from_raw_list() {
        // Regression for V6 mis-attribution mode: pubmed-search script
        // shipped only `authors[]` raw list with no structured flags.
        // V7 fallback should accept PI as first-author when authors[0]
        // matches.
        let p = paper_raw_authors(
            "12345",
            vec!["Zhang Li", "Wang Y", "Liu Z"],
        );
        let v = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap();
        assert!(v.author_position.contains("first-author"));
        assert!(v.author_position.contains("from raw authors[]"));
    }

    #[test]
    fn classify_v7_fallback_accepts_last_author_from_raw_list() {
        let p = paper_raw_authors(
            "12345",
            vec!["Wang Y", "Liu Z", "Zhang Li"],
        );
        let v = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap();
        assert!(v.author_position.contains("last-author"));
    }

    #[test]
    fn classify_v7_fallback_rejects_middle_author_with_position_hint() {
        // The exact V6 DESTINY-Breast05 failure mode: PI is at a middle
        // position (3rd of ~70), but absent structured flags the LLM
        // read the raw list and fabricated "Shao ZM (last author)".
        // The validator must (a) reject and (b) tell the LLM exactly
        // where the PI sits so it cannot fabricate an alternative.
        let mut authors = vec!["Loibl S", "Park YH", "Shao Z"];
        for i in 0..67 {
            // Pad with placeholder authors so we have ~70 total
            let _ = i;
            authors.push("Placeholder X");
        }
        let p = paper_raw_authors("41370739", authors);
        let err = PiPublicationValidatorTool::classify(&p, "Shao Z").unwrap_err();
        assert!(
            err.contains("PI is at position 3 of 70"),
            "expected position hint with concrete index; got: {err}"
        );
        assert!(err.contains("academic-integrity"));
    }

    #[test]
    fn classify_v7_fallback_skipped_when_structured_flags_present() {
        // If the pubmed-search script DID compute position flags
        // (the v5 张力 case), the structured path takes precedence and
        // the raw `authors[]` list is ignored — even if it contradicts.
        // Belt-and-braces: trust the script, not raw text.
        let mut p = paper("12345", "Zhang Li", "Other Author", true, false);
        p.authors = vec![RawAuthor {
            full: "Other Author".into(),
            last: "Other".into(),
            first: "Author".into(),
            affiliation: String::new(),
        }];
        let v = PiPublicationValidatorTool::classify(&p, "Zhang Li").unwrap();
        // Structured path matched first-author flag — fallback never ran
        assert_eq!(v.author_position, "first-author");
    }

    #[test]
    fn raw_author_matches_handles_pubmed_full_field() {
        let a = RawAuthor {
            full: "Zhang Li".into(),
            last: "Zhang".into(),
            first: "Li".into(),
            affiliation: String::new(),
        };
        assert!(PiPublicationValidatorTool::raw_author_matches(&a, "Zhang Li"));
        assert!(PiPublicationValidatorTool::raw_author_matches(&a, "Zhang"));
        assert!(!PiPublicationValidatorTool::raw_author_matches(&a, "Wang Y"));
    }

    #[test]
    fn raw_author_matches_falls_back_to_last_only() {
        let a = RawAuthor {
            full: String::new(),
            last: "Zhang".into(),
            first: "L".into(),
            affiliation: String::new(),
        };
        assert!(PiPublicationValidatorTool::raw_author_matches(&a, "Zhang"));
    }

    #[test]
    fn render_report_clean_when_no_rejections() {
        let valid = vec![PaperVerdict {
            pmid: "1".into(),
            title: "Good paper".into(),
            journal: "J".into(),
            year: "2026".into(),
            author_position: "first-author".into(),
        }];
        let report = PiPublicationValidatorTool::render_report("Zhang Li", &valid, &[]);
        assert!(report.contains("✅ Valid"));
        assert!(!report.contains("⚠️ Rejected"));
    }
}
