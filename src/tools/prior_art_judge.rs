//! LLM-as-judge prior-art adjudication (Layer 4 of PHC-RFC-2026-003).
//!
//! Fetches PubMed abstracts for a candidate PMID list and emits a
//! structured prompt the calling agent fills in via its own LLM
//! context. The agent's judgment becomes the deterministic prior-art
//! overlap table that downstream SOP steps consume.
//!
//! ## Architectural choice: abstract fetcher, not provider-in-tool
//!
//! The RFC describes Layer 4 as "LLM-as-judge". Two ways to implement:
//!
//! 1. **Provider injected into tool**: the tool itself calls deepseek
//!    for each candidate. Cleaner separation but requires plumbing
//!    `dyn Provider` through `PriorArtJudgeTool::new`, which the
//!    existing zeroclaw tool factory pattern doesn't carry. Would
//!    also mean L4 runs in a different LLM context than the bot's
//!    main reasoning, losing access to PI background, SOP rules,
//!    and prior turn state.
//! 2. **Agent does L4 in main loop** (this implementation): the tool
//!    deterministically fetches abstracts and emits a strict
//!    judgment template; the bot fills it in using its own LLM
//!    context where it already has the claim, PI context, prior
//!    art table, and SOP P0 rules loaded.
//!
//! Option 2 is simpler, cheaper (no separate provider call), and
//! more grounded (judgment uses full bot context). The "LLM-as-judge"
//! semantics is preserved — the judgment is still LLM-generated;
//! the tool just doesn't own the provider call.
//!
//! ## What it does
//!
//! - Takes a `claim` string + `candidate_pmids` list (typically the
//!   union of `aggregated_pmids` and tier sample PMIDs from
//!   `prior_art_expand`).
//! - Caps candidates at `max_candidates` (default 50) to bound
//!   downstream LLM input tokens.
//! - efetches PubMed abstracts in retmode=text (one batch request).
//! - Parses each record into (pmid, title, journal, year, abstract).
//! - Emits a structured markdown report: each candidate with its
//!   abstract truncated to ~600 chars + a **JSON judgment block**
//!   the bot must fill in for every candidate.
//!
//! ## What the bot must do after this tool returns
//!
//! For each candidate in the report, write the JSON judgment block
//! with fields:
//!  - `is_relevant` (bool)
//!  - `neighbor_type` ("same_disease_same_target" | "same_disease_different_target" | "different_disease_same_target" | "same_pathway_different_molecule" | "pi_research_path_extension" | "methodological_transfer" | "noise")
//!  - `overlap_estimate` (0-100)
//!  - `critical_warning` (string | null) — emit when:
//!    - claim says "0 papers" but this paper proves the neighborhood is non-empty
//!    - this paper is a study_protocol (not results) being cited as supporting evidence
//!    - PI authorship is misattributed (this paper's first/last author isn't the PI but
//!      the bot's pi_intel.md cites it as PI representative work)
//!    - cross-disease pollution (claim is on disease X, paper is on disease Y)
//!  - `framing_revision` (string | null) — concrete rewrite suggestion if the
//!    paper changes the novelty framing
//!
//! After judgment, the bot writes the filled markdown back to
//! `disease_scan.md` `## Layer 4: Judged prior art` section. SOP
//! step 7 PM 自检 reads this section to verify P0 rules.

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::traits::{Tool, ToolResult};

const NCBI_EFETCH_BASE: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";

/// Cap on candidates per call. 50 candidates × ~600 chars abstract
/// ≈ 7-8K tokens of input — fits comfortably in any modern LLM
/// context window even after the bot's existing claim / PI / SOP
/// context. Higher counts diminish judgment quality (LLM gets
/// distracted) and inflate cost.
const DEFAULT_MAX_CANDIDATES: usize = 50;

/// Hard upper bound regardless of caller request. NIH efetch's URL
/// length limit (~2K chars) corresponds to ~200 PMIDs comma-joined;
/// at that scale the report becomes unscannable for the LLM.
const HARD_MAX_CANDIDATES: usize = 100;

/// Truncation length for each abstract in the report. ~600 chars
/// captures the conclusion sentence in most PubMed abstracts (which
/// is what the judgment hinges on).
const ABSTRACT_TRUNCATE_CHARS: usize = 600;

#[derive(Debug, Clone, Default)]
struct PaperRecord {
    pmid: String,
    title: String,
    journal: String,
    year: String,
    abstract_text: String,
    /// Best-effort lookup error per record. Most efetch failures are
    /// batched (single HTTP call), but we tolerate per-record parse
    /// failures and surface them in the report.
    parse_error: Option<String>,
}

pub struct PriorArtJudgeTool {
    http: reqwest::Client,
    api_key: Option<String>,
}

impl PriorArtJudgeTool {
    pub fn new(api_key: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("zeroclaw-pharmaclaw/prior-art-judge")
            .build()
            .expect("reqwest client builds with default config");
        Self { http, api_key }
    }

    /// efetch PubMed abstracts in batch. retmode=text is easier to
    /// parse than xml for this use case (we only need title /
    /// journal / year / abstract; full structured XML would carry
    /// author lists, MeSH headings, grant numbers, etc that just
    /// inflate the LLM input).
    async fn fetch_abstracts(&self, pmids: &[String]) -> anyhow::Result<String> {
        if pmids.is_empty() {
            return Ok(String::new());
        }
        let mut params: Vec<(&str, String)> = vec![
            ("db", "pubmed".into()),
            ("id", pmids.join(",")),
            ("rettype", "abstract".into()),
            ("retmode", "text".into()),
        ];
        if let Some(k) = &self.api_key {
            params.push(("api_key", k.clone()));
        }
        let mut backoff_ms = 1000u64;
        let mut last_err: Option<String> = None;
        for attempt in 0..3 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = backoff_ms.saturating_mul(2);
            }
            let resp = match self.http.get(NCBI_EFETCH_BASE).query(&params).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(format!("transport error: {e}"));
                    continue;
                }
            };
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.is_success() {
                return Ok(body);
            }
            if status.as_u16() != 429 {
                anyhow::bail!("efetch HTTP {status}: {body}");
            }
            last_err = Some(format!("HTTP 429 (attempt {}): {body}", attempt + 1));
        }
        anyhow::bail!(
            "efetch failed after 3 attempts: {}",
            last_err.unwrap_or_else(|| "unknown error".into())
        )
    }

    fn render_report(claim: &str, papers: &[PaperRecord]) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        let _ = writeln!(out, "# Prior Art LLM-as-Judge (Layer 4)\n");
        let _ = writeln!(out, "**Claim**: {claim}\n");
        let _ = writeln!(
            out,
            "**Candidate count**: {} (capped at {} per call)\n",
            papers.len(),
            DEFAULT_MAX_CANDIDATES
        );

        // Instruction block: this is the LLM's job-statement. The SOP
        // step 5 prompt already tells the bot to fill these in, but
        // duplicating the contract here makes the report self-
        // documenting if anyone reads it after the fact.
        let _ = writeln!(
            out,
            "## Judgment task (the calling agent fills in the JSON block under each candidate)\n"
        );
        let _ = writeln!(
            out,
            "For each candidate below, write a fenced JSON block with these fields:\n"
        );
        let _ = writeln!(
            out,
            "- `is_relevant` (bool) — does this paper bear on the claim at all?"
        );
        let _ = writeln!(
            out,
            "- `neighbor_type` (string) — one of: `same_disease_same_target`, \
             `same_disease_different_target`, `different_disease_same_target`, \
             `same_pathway_different_molecule`, `pi_research_path_extension`, \
             `methodological_transfer`, `noise`."
        );
        let _ = writeln!(
            out,
            "- `overlap_estimate` (0-100) — % overlap with the claim's novelty argument."
        );
        let _ = writeln!(
            out,
            "- `critical_warning` (string | null) — emit when this paper proves a \
             '0 papers / 完全空白' claim is wrong, when the paper is a \
             `study_protocol` being cited as results, when PI authorship is \
             misattributed, or when there's cross-disease pollution. Null otherwise."
        );
        let _ = writeln!(
            out,
            "- `framing_revision` (string | null) — concrete rewrite text for \
             disease_scan.md / hypotheses.md if this paper changes how novelty must be \
             framed. Null if no rewrite needed."
        );
        let _ = writeln!(out);

        // Per-candidate payload.
        for (i, paper) in papers.iter().enumerate() {
            let _ = writeln!(
                out,
                "---\n\n### Candidate {n}: PMID:{pmid}\n",
                n = i + 1,
                pmid = paper.pmid
            );
            if let Some(err) = &paper.parse_error {
                let _ = writeln!(out, "⚠️ **parse error**: {err}\n");
                continue;
            }
            if !paper.title.is_empty() {
                let _ = writeln!(out, "**Title**: {}\n", paper.title);
            }
            if !paper.journal.is_empty() || !paper.year.is_empty() {
                let _ = writeln!(
                    out,
                    "**Journal/Year**: {} ({})\n",
                    paper.journal, paper.year
                );
            }
            if paper.abstract_text.is_empty() {
                let _ = writeln!(out, "_(no abstract retrieved)_\n");
            } else {
                let truncated = if paper.abstract_text.chars().count() > ABSTRACT_TRUNCATE_CHARS {
                    let cut: String = paper
                        .abstract_text
                        .chars()
                        .take(ABSTRACT_TRUNCATE_CHARS)
                        .collect();
                    format!("{cut}…")
                } else {
                    paper.abstract_text.clone()
                };
                let _ = writeln!(
                    out,
                    "**Abstract** (truncated to {ABSTRACT_TRUNCATE_CHARS} chars):\n"
                );
                let _ = writeln!(out, "> {truncated}\n");
            }

            // Empty fenced block the bot fills in.
            let _ = writeln!(out, "```json");
            let _ = writeln!(out, "{{");
            let _ = writeln!(out, "  \"pmid\": \"{}\",", paper.pmid);
            let _ = writeln!(out, "  \"is_relevant\": null,");
            let _ = writeln!(out, "  \"neighbor_type\": null,");
            let _ = writeln!(out, "  \"overlap_estimate\": null,");
            let _ = writeln!(out, "  \"critical_warning\": null,");
            let _ = writeln!(out, "  \"framing_revision\": null");
            let _ = writeln!(out, "}}");
            let _ = writeln!(out, "```\n");
        }

        if papers.is_empty() {
            let _ = writeln!(
                out,
                "_No candidate PMIDs supplied. The agent should pass the union of \
                 prior_art_expand tier PMIDs and pi_path aggregated PMIDs._"
            );
        }

        // Aggregation guidance for the bot.
        let _ = writeln!(out, "---\n");
        let _ = writeln!(
            out,
            "## After-judgment aggregation (the agent does this in disease_scan.md)\n"
        );
        let _ = writeln!(
            out,
            "Once every candidate has its JSON judgment filled in, the agent must:\n"
        );
        let _ = writeln!(
            out,
            "1. **Group by `neighbor_type`** — produce a markdown table of all \
             non-`noise` candidates organized by relationship to the claim."
        );
        let _ = writeln!(
            out,
            "2. **Surface every `critical_warning`** as a P0 issue at the top of the \
             section. SOP step 7 PM 自检 will block sop_advance if any \
             critical_warning is left unaddressed in the proposal narrative."
        );
        let _ = writeln!(
            out,
            "3. **Apply each `framing_revision`** to the §novelty / §prior art / \
             §differentiation narrative where the affected claim was made."
        );
        let _ = writeln!(
            out,
            "4. **Compute weighted overlap**: average `overlap_estimate` across \
             non-`noise` candidates. If average ≥ 50%, the claim's novelty argument \
             must be downscaled (no '高' creativity rating; pick '中' or '低')."
        );

        out
    }
}

#[async_trait]
impl Tool for PriorArtJudgeTool {
    fn name(&self) -> &str {
        "prior_art_judge"
    }

    fn description(&self) -> &str {
        "Layer 4 of PHC-RFC-2026-003. Fetches PubMed abstracts for a candidate PMID \
         list and emits a structured markdown report with one fenced JSON judgment \
         block per candidate. The calling agent fills in the JSON blocks using its \
         own LLM context (the agent already has claim / PI / SOP loaded — running \
         the judgment in-context is more grounded than running it in a separate \
         provider call). Inputs: `claim` (string), `candidate_pmids` (array of \
         PMID strings, typically the union of prior_art_expand aggregated and tier \
         sample PMIDs), optional `max_candidates` (default 50, hard cap 100). \
         Output is a markdown report ready to drop into disease_scan.md `## Layer \
         4: Judged prior art` section. SOP step 7 PM 自检 reads the filled-in JSON \
         to verify (a) no critical_warning is unaddressed, (b) framing_revisions \
         have been applied, (c) average overlap is consistent with the proposal's \
         novelty rating."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The natural-language research claim being judged. Used in the report header so the LLM judgment stays anchored."
                },
                "candidate_pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "PMID strings to fetch and emit judgment templates for. Typically constructed by unioning prior_art_expand tier_breakdown PMIDs + pi_path aggregated_pmids and dedup'ing."
                },
                "max_candidates": {
                    "type": "integer",
                    "description": "Cap on candidates this call processes. Default 50, hard cap 100. Higher counts inflate LLM input tokens and dilute judgment quality."
                }
            },
            "required": ["claim", "candidate_pmids"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let claim = args
            .get("claim")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let pmids: Vec<String> = args
            .get("candidate_pmids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
                    .collect()
            })
            .unwrap_or_default();

        if pmids.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "candidate_pmids is empty or all entries failed validation \
                     (must be ASCII digit strings). The agent should pass the \
                     union of prior_art_expand tier PMIDs and pi_path aggregated \
                     PMIDs before calling prior_art_judge."
                        .into(),
                ),
            });
        }

        // Dedup while preserving order so the report is reproducible.
        let mut seen: HashSet<String> = HashSet::new();
        let mut deduped: Vec<String> = Vec::with_capacity(pmids.len());
        for p in pmids {
            if seen.insert(p.clone()) {
                deduped.push(p);
            }
        }

        let max = args
            .get("max_candidates")
            .and_then(|v| v.as_u64())
            .map(|n| usize::try_from(n).unwrap_or(HARD_MAX_CANDIDATES))
            .unwrap_or(DEFAULT_MAX_CANDIDATES)
            .min(HARD_MAX_CANDIDATES);
        deduped.truncate(max);

        let raw = match self.fetch_abstracts(&deduped).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to fetch abstracts: {e}")),
                });
            }
        };

        let mut papers = parse_pubmed_text_records(&raw);

        // efetch may return fewer records than requested when some
        // PMIDs don't exist or are suppressed. We pad with explicit
        // "not retrieved" entries so the bot's judgment loop doesn't
        // silently drop candidates.
        if papers.len() < deduped.len() {
            let returned: HashSet<String> = papers.iter().map(|p| p.pmid.clone()).collect();
            for p in &deduped {
                if !returned.contains(p) {
                    papers.push(PaperRecord {
                        pmid: p.clone(),
                        parse_error: Some(
                            "no record returned by efetch (PMID may be retracted, suppressed, or invalid)"
                                .into(),
                        ),
                        ..Default::default()
                    });
                }
            }
        }

        // Sort by input order.
        papers.sort_by_key(|p| {
            deduped
                .iter()
                .position(|q| q == &p.pmid)
                .unwrap_or(usize::MAX)
        });

        let report = Self::render_report(&claim, &papers);
        Ok(ToolResult {
            success: true,
            output: report,
            error: None,
        })
    }
}

/// Parse PubMed efetch text-mode output into structured records.
///
/// Format (one record):
/// ```text
/// 1. Cell. 2025 Jan;...
///
/// Title goes here.
///
/// Author A, Author B, Author C.
///
/// Abstract paragraph one. Abstract paragraph two...
///
/// PMID:12345678
/// ```
///
/// Records are separated by blank line + record-number prefix
/// (`N. <citation>`). We split on the record-number prefix and
/// extract title (first non-empty line after the citation), journal
/// (first token of citation), year (first 4-digit run in citation),
/// abstract (everything between the author line and the PMID line),
/// and PMID (the explicit `PMID:XXXX` line).
fn parse_pubmed_text_records(blob: &str) -> Vec<PaperRecord> {
    let mut out: Vec<PaperRecord> = Vec::new();
    let mut current: Option<RecordBuilder> = None;

    for line in blob.lines() {
        // Record-start detection: line matches `<digit>+. <text>`
        // where the digit run is at the start. PubMed uses 1-indexed
        // record numbers in batch responses.
        if is_record_start(line) {
            if let Some(b) = current.take() {
                out.push(b.finalize());
            }
            current = Some(RecordBuilder::new(line));
            continue;
        }
        if let Some(b) = current.as_mut() {
            b.push_line(line);
        }
    }
    if let Some(b) = current {
        out.push(b.finalize());
    }
    // Drop any record without a PMID (parse failure on that record).
    out.retain(|r| !r.pmid.is_empty());
    out
}

fn is_record_start(line: &str) -> bool {
    let trimmed = line.trim_start();
    let mut chars = trimmed.chars();
    let first_non_digit = match chars.find(|c| !c.is_ascii_digit()) {
        Some(c) => c,
        None => return false,
    };
    if first_non_digit != '.' {
        return false;
    }
    // Must have content after the dot (citation)
    chars.next().map(|c| c == ' ' || c == '\t').unwrap_or(false)
}

/// Stateful per-record builder. We accumulate lines as we go and
/// classify them at finalize time when we have full context (e.g.
/// the title is "first non-empty line after the citation"; we can
/// only know that when we've seen at least the citation line and
/// one blank-then-content sequence).
struct RecordBuilder {
    citation_line: String,
    body_lines: Vec<String>,
    pmid: String,
}

impl RecordBuilder {
    fn new(citation_line: &str) -> Self {
        Self {
            citation_line: citation_line.to_string(),
            body_lines: Vec::new(),
            pmid: String::new(),
        }
    }

    fn push_line(&mut self, line: &str) {
        // Capture PMID line at any position.
        if let Some(rest) = line.trim_start().strip_prefix("PMID:") {
            let candidate: String = rest
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if !candidate.is_empty() {
                self.pmid = candidate;
                return;
            }
        }
        self.body_lines.push(line.to_string());
    }

    fn finalize(self) -> PaperRecord {
        let mut record = PaperRecord {
            pmid: self.pmid,
            ..Default::default()
        };

        // Citation line shape: "1. Cell. 2025 Jan;180(3):412-426."
        // Split off the leading "<n>. " prefix.
        let citation = self
            .citation_line
            .split_once(". ")
            .map(|(_, rest)| rest)
            .unwrap_or(&self.citation_line)
            .to_string();
        // Journal: text before first period.
        if let Some(idx) = citation.find('.') {
            record.journal = citation[..idx].trim().to_string();
        }
        // Year: first 4-digit run that starts with 19 or 20.
        let mut chars = citation.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '1' || c == '2' {
                let mut buf = String::from(c);
                for _ in 0..3 {
                    if let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            buf.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                if buf.len() == 4 && (buf.starts_with("19") || buf.starts_with("20")) {
                    record.year = buf;
                    break;
                }
            }
        }

        // Body: title is first non-empty paragraph; abstract is
        // everything after the second blank-line-separated block
        // (citation = block 0, title = block 1, authors = block 2,
        // abstract = block 3+).
        let blocks = blocks_from_lines(&self.body_lines);
        if let Some(b) = blocks.first() {
            record.title = b.join(" ").trim().to_string();
        }
        if blocks.len() >= 3 {
            // Concatenate from block 2 onward (skip authors block at
            // index 1). Some records have no authors block, in which
            // case we'd be off by one — handled by the heuristic in
            // looks_like_author_line: if block 1 looks like an
            // abstract (long sentence vs author list), use it.
            let start = if blocks.len() >= 2 && looks_like_author_line(&blocks[1]) {
                2
            } else {
                1
            };
            let mut abs_parts: Vec<String> = Vec::new();
            for blk in blocks.iter().skip(start) {
                abs_parts.push(blk.join(" "));
            }
            record.abstract_text = abs_parts.join("\n\n").trim().to_string();
        } else if blocks.len() == 2 {
            // Title + (authors or abstract) only — no abstract for sure.
        }
        record
    }
}

/// Group consecutive non-blank lines into paragraphs.
fn blocks_from_lines(lines: &[String]) -> Vec<Vec<String>> {
    let mut blocks: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            if !current.is_empty() {
                blocks.push(std::mem::take(&mut current));
            }
        } else {
            current.push(line.trim().to_string());
        }
    }
    if !current.is_empty() {
        blocks.push(current);
    }
    blocks
}

/// Heuristic: an author line is a comma-separated list of "Name N"
/// tokens (last name + initials). An abstract line is a long
/// sentence with regular punctuation. We treat ≥3 commas + average
/// word length < 8 as "looks like authors". This is good enough to
/// disambiguate author lists from one-paragraph abstracts.
fn looks_like_author_line(block: &[String]) -> bool {
    let joined: String = block.join(" ");
    let comma_count = joined.matches(',').count();
    if comma_count < 3 {
        return false;
    }
    let words: Vec<&str> = joined.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }
    let avg_word_len: f64 = words.iter().map(|w| w.len() as f64).sum::<f64>() / words.len() as f64;
    avg_word_len < 8.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_record_start_recognizes_numbered_citations() {
        assert!(is_record_start("1. Cell. 2025 Jan;180(3):412-426."));
        assert!(is_record_start("42. NEJM. 2024;390:1234."));
        assert!(!is_record_start("Cell. 2025 Jan."));
        assert!(!is_record_start("PMID:12345"));
        assert!(!is_record_start(""));
    }

    #[test]
    fn parse_single_record_extracts_all_fields() {
        let blob = "1. Cell. 2025 Mar 15;180(6):1234-1247.\n\n\
                    Title of the paper goes here.\n\n\
                    Smith JA, Doe BC, Jones DE.\n\n\
                    Abstract paragraph one explains the background and aim. \
                    Abstract paragraph two reports the results and conclusions.\n\n\
                    PMID:12345678\n";
        let records = parse_pubmed_text_records(blob);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(r.pmid, "12345678");
        assert_eq!(r.title, "Title of the paper goes here.");
        assert_eq!(r.journal, "Cell");
        assert_eq!(r.year, "2025");
        assert!(r.abstract_text.contains("background and aim"));
        assert!(r.abstract_text.contains("results and conclusions"));
    }

    #[test]
    fn parse_multi_record_handles_record_boundaries() {
        let blob = "1. Cell. 2025;180:1.\n\n\
                    First paper title.\n\n\
                    Author A.\n\n\
                    Abstract one.\n\n\
                    PMID:11111\n\
                    \n\
                    2. NEJM. 2024;390:2.\n\n\
                    Second paper title.\n\n\
                    Author B.\n\n\
                    Abstract two.\n\n\
                    PMID:22222\n";
        let records = parse_pubmed_text_records(blob);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].pmid, "11111");
        assert_eq!(records[1].pmid, "22222");
        assert_eq!(records[0].title, "First paper title.");
        assert_eq!(records[1].title, "Second paper title.");
    }

    #[test]
    fn parse_drops_records_without_pmid() {
        // A malformed record (no PMID line) should be silently dropped
        // rather than emitted with empty PMID, which would confuse the
        // post-fetch padding logic in execute().
        let blob = "1. Cell. 2025;180:1.\n\n\
                    Garbage record with no PMID.\n";
        let records = parse_pubmed_text_records(blob);
        assert!(records.is_empty());
    }

    #[test]
    fn looks_like_author_line_distinguishes_authors_from_abstract() {
        let authors = vec!["Smith JA, Doe BC, Jones DE, Wang H.".to_string()];
        assert!(looks_like_author_line(&authors));
        let abstract_block = vec![
            "We investigated the role of ALOX15 in intracerebral hemorrhage and \
             found that knockout mice showed reduced lipid peroxidation in the \
             perihematomal cortex."
                .to_string(),
        ];
        assert!(!looks_like_author_line(&abstract_block));
    }

    #[test]
    fn render_report_includes_judgment_template_for_each_paper() {
        let papers = vec![
            PaperRecord {
                pmid: "11111".into(),
                title: "Paper one".into(),
                journal: "Cell".into(),
                year: "2025".into(),
                abstract_text: "Abstract one.".into(),
                parse_error: None,
            },
            PaperRecord {
                pmid: "22222".into(),
                title: "Paper two".into(),
                journal: "NEJM".into(),
                year: "2024".into(),
                abstract_text: "Abstract two.".into(),
                parse_error: None,
            },
        ];
        let r = PriorArtJudgeTool::render_report("test claim", &papers);
        assert!(r.contains("# Prior Art LLM-as-Judge (Layer 4)"));
        assert!(r.contains("**Claim**: test claim"));
        assert!(r.contains("PMID:11111"));
        assert!(r.contains("PMID:22222"));
        assert!(r.contains("\"is_relevant\": null"));
        assert!(r.contains("\"neighbor_type\": null"));
        assert!(r.contains("\"overlap_estimate\": null"));
        assert!(r.contains("\"critical_warning\": null"));
        assert!(r.contains("\"framing_revision\": null"));
        // Per-PMID JSON blocks should reference the actual PMID.
        assert!(r.contains("\"pmid\": \"11111\""));
        assert!(r.contains("\"pmid\": \"22222\""));
    }

    #[test]
    fn render_report_marks_parse_errors() {
        let papers = vec![PaperRecord {
            pmid: "99999".into(),
            parse_error: Some("not retrieved".into()),
            ..Default::default()
        }];
        let r = PriorArtJudgeTool::render_report("c", &papers);
        assert!(r.contains("⚠️ **parse error**"));
        assert!(r.contains("not retrieved"));
    }

    #[test]
    fn render_report_truncates_long_abstracts() {
        let long_text = "x".repeat(2000);
        let papers = vec![PaperRecord {
            pmid: "1".into(),
            title: "T".into(),
            journal: "J".into(),
            year: "2025".into(),
            abstract_text: long_text,
            parse_error: None,
        }];
        let r = PriorArtJudgeTool::render_report("c", &papers);
        // Abstract section should end with the truncation ellipsis
        // rather than the full 2000-char text.
        assert!(r.contains("…"));
        assert!(!r.contains(&"x".repeat(1500)));
    }

    #[test]
    fn render_report_includes_aggregation_guidance() {
        let r = PriorArtJudgeTool::render_report("c", &[]);
        assert!(r.contains("After-judgment aggregation"));
        assert!(r.contains("Group by"));
        assert!(r.contains("Surface every"));
        assert!(r.contains("framing_revision"));
        assert!(r.contains("weighted overlap"));
    }

    #[tokio::test]
    async fn execute_rejects_empty_pmid_list() {
        let tool = PriorArtJudgeTool::new(None);
        let result = tool
            .execute(json!({
                "claim": "test",
                "candidate_pmids": []
            }))
            .await
            .expect("tool runs");
        assert!(!result.success);
        let err = result.error.expect("error message present");
        assert!(err.contains("candidate_pmids is empty"));
    }

    #[tokio::test]
    async fn execute_rejects_non_numeric_pmids() {
        let tool = PriorArtJudgeTool::new(None);
        let result = tool
            .execute(json!({
                "claim": "test",
                "candidate_pmids": ["not-a-pmid", "also-bad"]
            }))
            .await
            .expect("tool runs");
        assert!(!result.success);
    }
}
