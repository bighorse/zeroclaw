//! Prior Art neighborhood expansion (Layer 1 MVP).
//!
//! Implements the multi-query expansion described in PHC-RFC-2026-003.
//! The tool takes a research claim already decomposed by the LLM into
//! 5-7 axes (molecule / disease / mechanism / cell_type / outcome /
//! optional model / optional intervention) and runs 8-15 independent
//! PubMed eSearch queries that intersect different subsets of those
//! axes. The dedup'd PMID set is returned with a `blank_claim_check`
//! verdict — the deterministic signal SOP step 5 needs to refuse "PI
//! claims this is 0 papers but neighborhood research exists" patterns.
//!
//! ## What it catches
//!
//! Live observation on PharmaClaw v_zhang (`run-1778245403320-0002`,
//! zhang_xiaobo / 常德市第一人民医院 ICH neurology):
//! the LLM wrote "PubMed 2021-2026: ICH+ferroptosis 210 篇 / ICH+ALOX15
//! **0 篇**" and used "0 篇" as the cornerstone of its novelty
//! argument. Manual web verification found ≥3 directly relevant papers
//! the LLM missed:
//!  - J Proteome Res 2025 — Pan-Cell Death Protein Signature for ICH
//!    (directly ICH + ALOX15)
//!  - PMID:35186185 — Cepharanthine + SAH + ALOX15 (subarachnoid
//!    hemorrhage neighbor)
//!  - PMID:35090880 — SSAT1/ALOX15 + cerebral infarction (stroke
//!    neighbor)
//!
//! Root cause: the LLM ran a single `(ICH) AND (ALOX15)` query.
//! Splitting the claim into independent axes and intersecting subsets
//! ((mol AND mech AND cell), (mol AND mech), (mol AND disease neighbor))
//! catches all three.
//!
//! This recurrence pattern shows up across PharmaClaw's 20+ versions:
//!  - v6 邵志敏: Chen W 2026 Cancer Discov overlap missed (single-query
//!    too narrow on cancer indication)
//!  - v_li 李娟梅: GZFL+dienogest already-published vs claimed-novel
//!    (no cross-check on combination keyword)
//!  - **v_zhang 张小波**: ICH+ALOX15 "0 篇" overstatement (this case)
//!
//! ## What it does NOT catch
//!
//! - **Whether the candidate PMIDs actually contradict the claim**.
//!   The semantic relevance call ("does this paper support, contradict
//!   or merely neighbor the claim?") is Layer 4's job. This MVP only
//!   enforces "if neighborhood papers exist, the LLM cannot truthfully
//!   say '0 papers / complete blank'".
//! - **Decomposition quality**. The LLM is responsible for splitting
//!   the claim into useful axes. If the LLM writes one synonym per
//!   axis where 4-5 exist, recall suffers. The SOP step 5 prompt
//!   provides templates for the seven canonical axes.
//! - **Non-PubMed databases**. Chinese-language CNKI / Wanfang,
//!   preprint servers (bioRxiv, medRxiv), patent literature are out
//!   of scope. PubMed covers the v_zhang / v_li / v6 cases observed
//!   in production; CNKI bridge can be added in a future Layer.
//!
//! ## Acceptance criteria for "0 papers / complete blank" claim
//!
//! `blank_claim_check.verdict == "ok_to_claim_blank"` requires ALL of:
//!  - T1 (5-axis intersection) returns 0 PMIDs, AND
//!  - T2 (4-axis, drop one) returns 0 PMIDs across all 5 drop-one
//!    permutations, AND
//!  - T3 (3-axis subsets) returns 0 PMIDs across both T3 queries
//!
//! Anything else — including "T1 zero but T2 non-empty" — is
//! `forbidden_neighborhood_exists` and SOP step 5 must rewrite the
//! claim from "0 papers / complete blank" to "X papers in neighborhood,
//! N specifically on the (mol, disease, mechanism) intersection;
//! novelty argument restricted to (cell_type, intervention) axes".

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::traits::{Tool, ToolResult};

const NCBI_ESEARCH_BASE: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
const NCBI_EFETCH_BASE: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";

/// Maximum PMIDs returned per individual query. NIH eutils default is
/// 20; we ask for 100 because Layer 1's signal of interest is
/// "neighborhood is non-empty", and the broader queries (T4, T5) need
/// some volume to produce a useful upper bound on overlap. Layer 4
/// will downsample before LLM-as-judge.
const RETMAX_PER_QUERY: u32 = 100;

/// LLM-decomposed claim axes. The LLM is required to populate the
/// five canonical axes; `model` and `intervention` are optional and
/// only used when the claim mentions them (a mechanism study without
/// a perturbation set is still complete with the five required axes).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClaimDimensions {
    /// Synonyms / aliases for the molecule, gene, drug, or compound
    /// (e.g. `ALOX15`, `ALOX-15`, `15-lipoxygenase`, `15-LOX-1`).
    /// Required.
    #[serde(default)]
    pub molecule: Vec<String>,
    /// Disease / pathology synonyms (e.g. `intracerebral hemorrhage`,
    /// `ICH`, `cerebral hemorrhage`). Required.
    #[serde(default)]
    pub disease: Vec<String>,
    /// Mechanism / pathway / cell-death modality synonyms (e.g.
    /// `ferroptosis`, `lipid peroxidation`, `iron-dependent cell
    /// death`). Required.
    #[serde(default)]
    pub mechanism: Vec<String>,
    /// Cell type / tissue context synonyms (e.g. `neuron`, `neuronal`,
    /// `cortical neuron`). Required.
    #[serde(default)]
    pub cell_type: Vec<String>,
    /// Outcome / readout / clinical endpoint synonyms (e.g. `cell
    /// death`, `neurodegeneration`, `neurological deficit`). Required.
    #[serde(default)]
    pub outcome: Vec<String>,
    /// Optional: experimental model synonyms (e.g. `mouse model`,
    /// `rat model`, `in vivo`).
    #[serde(default)]
    pub model: Vec<String>,
    /// Optional: intervention modality synonyms (e.g. `inhibitor`,
    /// `knockout`, `knockdown`).
    #[serde(default)]
    pub intervention: Vec<String>,
}

impl ClaimDimensions {
    /// Required axes are populated. (`model` and `intervention` may be
    /// empty without invalidating decomposition.)
    fn required_axes_present(&self) -> bool {
        !self.molecule.is_empty()
            && !self.disease.is_empty()
            && !self.mechanism.is_empty()
            && !self.cell_type.is_empty()
            && !self.outcome.is_empty()
    }

    fn missing_required(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.molecule.is_empty() {
            out.push("molecule");
        }
        if self.disease.is_empty() {
            out.push("disease");
        }
        if self.mechanism.is_empty() {
            out.push("mechanism");
        }
        if self.cell_type.is_empty() {
            out.push("cell_type");
        }
        if self.outcome.is_empty() {
            out.push("outcome");
        }
        out
    }
}

/// MeSH ontology expansion (Layer 2 of PHC-RFC-2026-003).
///
/// Wraps NCBI's MeSH database via eutils. For each input axis term
/// returns Entry Terms (synonyms) and immediate siblings via the
/// MeSH tree. Sibling lookup is the v_zhang case's path to catching
/// SAH / cerebral infarction when the axis term is `Intracerebral
/// Hemorrhage` — these are the cross-disease neighbors that L1's
/// drop-disease tier eventually catches but L2 catches up-front
/// at the term-mapping stage, before any PubMed query runs.
///
/// ## What MeSH gives us
///
/// `efetch?db=mesh&id=<UID>&retmode=xml` returns a structured text
/// blob (NCBI quirk: `retmode=xml` for db=mesh actually returns
/// labeled-section text, not XML — see efetch_mesh_record below).
/// The two sections we use:
///
/// - **Entry Terms**: synonyms of the same MeSH concept. For
///   "Cerebral Hemorrhage" (UID 68002543) these include
///   "Intracerebral Hemorrhage", "Brain Hemorrhage Cerebral",
///   "Cerebral Parenchymal Hemorrhage" — directly useful as
///   axis-term expansions because PubMed queries with `[tiab]`
///   restriction won't auto-expand to these.
/// - **Tree Number(s)**: hierarchical position
///   (e.g. `C14.907.253.573.200` for Cerebral Hemorrhage). The
///   parent prefix (`C14.907.253.573` = Intracranial Hemorrhages)
///   has other immediate children — the siblings — which we resolve
///   via a second esearch (`<parent>[Tree Number]`).
///
/// ## Failure mode
///
/// MeSH lookup degrades gracefully: any error (network / rate-limit
/// / term not in MeSH) emits an axis annotation in the report and
/// the tool falls back to L1-only expansion for that axis. The
/// blank_claim_check verdict is unaffected (L2 only widens recall;
/// L1 alone is sufficient for the v_zhang case empirically).
mod mesh {
    use std::time::Duration;

    use serde_json::Value;

    use super::{NCBI_EFETCH_BASE, NCBI_ESEARCH_BASE};

    /// Per-axis-term expansion: synonyms + siblings. The original
    /// term is always included so the caller can blindly replace
    /// the axis term list with the expansion result.
    #[derive(Debug, Clone, Default)]
    pub(super) struct MeshExpansion {
        pub original: String,
        pub synonyms: Vec<String>,
        pub siblings: Vec<String>,
        /// Best-effort lookup error, or `None` on success.
        pub error: Option<String>,
    }

    pub(super) struct MeshClient {
        http: reqwest::Client,
    }

    impl MeshClient {
        pub(super) fn new(http: reqwest::Client) -> Self {
            Self { http }
        }

        /// Look up a single axis term in MeSH. Returns the canonical
        /// MeSH descriptor name (which may differ from the input
        /// term, e.g. "ICH" → "Cerebral Hemorrhage") plus synonyms
        /// and siblings.
        pub(super) async fn expand_term(&self, term: &str) -> MeshExpansion {
            let mut out = MeshExpansion {
                original: term.to_string(),
                ..Default::default()
            };

            let uid = match self.lookup_uid(term).await {
                Ok(Some(uid)) => uid,
                Ok(None) => {
                    // Term doesn't map to MeSH — that's fine, many
                    // chemical / drug / acronym tokens won't. The
                    // axis still has its L1 synonyms.
                    return out;
                }
                Err(e) => {
                    out.error = Some(format!("MeSH lookup failed for '{term}': {e}"));
                    return out;
                }
            };

            // Polite delay between paired esearch + efetch for the
            // same term. NIH 3 req/s budget shared with L1.
            tokio::time::sleep(Duration::from_millis(350)).await;

            let record_text = match self.fetch_record(&uid).await {
                Ok(t) => t,
                Err(e) => {
                    out.error = Some(format!("MeSH efetch failed for UID {uid}: {e}"));
                    return out;
                }
            };

            let (canonical_name, entry_terms, tree_numbers) =
                parse_mesh_record(&record_text);

            // Sanity check: PubMed's esearch on db=mesh ranks by
            // relevance and can map polysemous terms to surprising
            // concepts. Live observation: "cell death" mapped to
            // "PD-L2 Ligand" (CD273) which has no relation to the
            // generic concept; "ferroptosis" mapped to "AIFM2
            // protein, mouse" (a regulator, not the concept). When
            // the canonical MeSH name shares no token with the
            // input term, we treat the lookup as a miss rather than
            // contaminate the axis with unrelated synonyms. The L1
            // query then runs on the LLM's original axis terms.
            if !canonical_name.is_empty() && !share_token(&canonical_name, term) {
                out.error = Some(format!(
                    "MeSH canonical name '{canonical_name}' does not share a token with input '{term}' \
                     — treating as polysemous miss, axis not expanded"
                ));
                return out;
            }

            // Canonical MeSH name + Entry Terms become axis synonyms.
            // Dedup against the original to avoid pathological
            // `(ALOX15 OR ALOX15)` clauses in L1 query building.
            let mut synonyms: Vec<String> = Vec::new();
            if !canonical_name.is_empty()
                && !canonical_name.eq_ignore_ascii_case(term)
            {
                synonyms.push(canonical_name);
            }
            for et in entry_terms {
                if !et.eq_ignore_ascii_case(term) && !synonyms.iter().any(|s| s.eq_ignore_ascii_case(&et)) {
                    synonyms.push(et);
                }
            }
            out.synonyms = synonyms;

            // Sibling expansion: take the first tree number, drop
            // the last segment, esearch for that prefix. Skip if
            // the term has no tree numbers (some chemicals have
            // none) or if the tree is at the root (no parent).
            if let Some(tree) = tree_numbers.first() {
                if let Some(parent) = parent_tree(tree) {
                    tokio::time::sleep(Duration::from_millis(350)).await;
                    match self.search_siblings(&parent).await {
                        Ok(siblings) => {
                            // Filter out the term itself (it's the
                            // tree's exact match) and any duplicate
                            // of canonical_name / synonyms.
                            out.siblings = siblings
                                .into_iter()
                                .filter(|s| {
                                    !s.eq_ignore_ascii_case(term)
                                        && !out.synonyms.iter()
                                            .any(|x| x.eq_ignore_ascii_case(s))
                                })
                                .collect();
                        }
                        Err(e) => {
                            out.error = Some(format!(
                                "sibling search failed for tree '{parent}': {e}"
                            ));
                        }
                    }
                }
            }

            out
        }

        /// esearch db=mesh by free-text term → returns the first UID
        /// (PubMed's term-mapping picks the most relevant match).
        async fn lookup_uid(&self, term: &str) -> anyhow::Result<Option<String>> {
            let resp = self
                .http
                .get(NCBI_ESEARCH_BASE)
                .query(&[
                    ("db", "mesh"),
                    ("term", term),
                    ("retmode", "json"),
                    ("retmax", "1"),
                ])
                .send()
                .await?;
            let status = resp.status();
            let body = resp.text().await?;
            if !status.is_success() {
                anyhow::bail!("HTTP {status}: {body}");
            }
            let parsed: Value = serde_json::from_str(&body)?;
            let uid = parsed
                .get("esearchresult")
                .and_then(|r| r.get("idlist"))
                .and_then(|l| l.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            Ok(uid)
        }

        /// efetch db=mesh by UID → returns the labeled-section text
        /// blob NCBI emits as the "xml" retmode for the mesh DB.
        async fn fetch_record(&self, uid: &str) -> anyhow::Result<String> {
            let resp = self
                .http
                .get(NCBI_EFETCH_BASE)
                .query(&[("db", "mesh"), ("id", uid), ("retmode", "xml")])
                .send()
                .await?;
            let status = resp.status();
            let body = resp.text().await?;
            if !status.is_success() {
                anyhow::bail!("HTTP {status}: {body}");
            }
            Ok(body)
        }

        /// Resolve siblings of a parent tree number. Returns the
        /// MeSH descriptor names of all immediate children of the
        /// parent (which includes the original term — caller filters).
        async fn search_siblings(&self, parent_tree: &str) -> anyhow::Result<Vec<String>> {
            // esearch returns UIDs; we then need efetch to get names.
            // To keep this cheap, we cap retmax at 12 — a node with
            // more than ~10 immediate siblings is usually too coarse
            // to be a useful neighborhood (e.g. drug classes).
            let resp = self
                .http
                .get(NCBI_ESEARCH_BASE)
                .query(&[
                    ("db", "mesh"),
                    ("term", &format!("{parent_tree}[Tree Number]")),
                    ("retmode", "json"),
                    ("retmax", "12"),
                ])
                .send()
                .await?;
            let status = resp.status();
            let body = resp.text().await?;
            if !status.is_success() {
                anyhow::bail!("HTTP {status}: {body}");
            }
            let parsed: Value = serde_json::from_str(&body)?;
            let uids: Vec<String> = parsed
                .get("esearchresult")
                .and_then(|r| r.get("idlist"))
                .and_then(|l| l.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            if uids.is_empty() {
                return Ok(Vec::new());
            }

            tokio::time::sleep(Duration::from_millis(350)).await;
            // Batch-efetch all sibling UIDs in one call (comma-joined).
            let resp = self
                .http
                .get(NCBI_EFETCH_BASE)
                .query(&[
                    ("db", "mesh"),
                    ("id", &uids.join(",")),
                    ("retmode", "xml"),
                ])
                .send()
                .await?;
            let status = resp.status();
            let body = resp.text().await?;
            if !status.is_success() {
                anyhow::bail!("HTTP {status}: {body}");
            }
            // Multi-record response: each record starts with a line
            // like "1: Cerebral Hemorrhage" / "2: Subarachnoid
            // Hemorrhage". We only want the names.
            Ok(parse_descriptor_names(&body))
        }
    }

    /// Compute the parent tree number by dropping the last
    /// dot-segment. `C14.907.253.573.200` → `Some("C14.907.253.573")`.
    /// Tree numbers with no dot (top-level categories) return None.
    fn parent_tree(tree: &str) -> Option<String> {
        tree.rfind('.').map(|i| tree[..i].to_string())
    }

    /// True when MeSH's canonical name and the input term share at
    /// least one alphanumeric token (case-insensitive, length ≥3).
    /// Stop-words ("of", "the", "in") and short tokens are ignored
    /// to avoid spurious matches. Used to detect polysemous
    /// mis-mappings ("cell death" → "PD-L2 Ligand" shares no token
    /// → flagged as miss).
    pub(super) fn share_token(canonical: &str, input: &str) -> bool {
        const STOPWORDS: &[&str] = &[
            "the", "and", "of", "in", "for", "with", "to", "a", "an", "or",
        ];
        fn tokenize(s: &str) -> Vec<String> {
            s.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|t| t.len() >= 3 && !STOPWORDS.contains(t))
                .map(|t| t.to_string())
                .collect()
        }
        let a = tokenize(canonical);
        let b = tokenize(input);
        if b.is_empty() {
            // Input is all stopwords / short tokens — refuse the
            // expansion entirely. This shouldn't happen for real
            // axis terms but we fail closed defensively.
            return false;
        }
        if a.is_empty() {
            // Canonical name has no real tokens (numeric-only
            // chemical names rare but possible); allow through
            // since we have no way to compare and downstream
            // filters can catch noise.
            return true;
        }
        a.iter().any(|t| b.contains(t))
    }

    /// Parse a MeSH efetch text blob. Returns
    /// (canonical_descriptor_name, entry_terms, tree_numbers).
    pub(super) fn parse_mesh_record(text: &str) -> (String, Vec<String>, Vec<String>) {
        let mut name = String::new();
        let mut entry_terms: Vec<String> = Vec::new();
        let mut tree_numbers: Vec<String> = Vec::new();
        let mut in_entry_terms = false;
        for (i, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim_end();
            // First line is `1: <Descriptor Name>` (1-indexed
            // record number for multi-record responses).
            if i == 0 {
                if let Some(rest) = line.split_once(':').map(|(_, r)| r.trim()) {
                    name = rest.to_string();
                }
                continue;
            }
            if line.starts_with("Tree Number(s):") {
                let body = line.trim_start_matches("Tree Number(s):").trim();
                tree_numbers = body
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                continue;
            }
            if line.trim() == "Entry Terms:" {
                in_entry_terms = true;
                continue;
            }
            if in_entry_terms {
                let trimmed = line.trim();
                // Entry Terms section ends at first blank line OR
                // at a line that doesn't start with whitespace
                // (the next labeled section starts flush-left).
                if trimmed.is_empty() {
                    in_entry_terms = false;
                    continue;
                }
                if !line.starts_with(char::is_whitespace) {
                    in_entry_terms = false;
                    // fall through to other parsers
                } else {
                    entry_terms.push(trimmed.to_string());
                    continue;
                }
            }
        }
        (name, entry_terms, tree_numbers)
    }

    /// Extract descriptor names from a multi-record efetch response.
    /// Each record's first line is `<n>: <Name>`; capture the name.
    pub(super) fn parse_descriptor_names(text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for line in text.lines() {
            // Match lines like "1: Cerebral Hemorrhage" or
            // "10: Subarachnoid Hemorrhage" — number colon space name.
            if let Some((prefix, rest)) = line.split_once(':') {
                if prefix.trim().chars().all(|c| c.is_ascii_digit()) {
                    let name = rest.trim();
                    if !name.is_empty() {
                        out.push(name.to_string());
                    }
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parent_tree_drops_last_segment() {
            assert_eq!(
                parent_tree("C14.907.253.573.200"),
                Some("C14.907.253.573".to_string())
            );
            assert_eq!(parent_tree("C14"), None);
            assert_eq!(parent_tree(""), None);
        }

        #[test]
        fn parse_mesh_record_extracts_entry_terms_and_tree() {
            // Real efetch response shape (truncated for tests).
            let blob = "1: Cerebral Hemorrhage\nBleeding into one or both CEREBRAL HEMISPHERES.\n\nTree Number(s): C10.228.140.300.535.200, C14.907.253.573.200\nEntry Terms:\n    Hemorrhage, Cerebral\n    Cerebral Hemorrhages\n    Intracerebral Hemorrhage\n    Cerebral Parenchymal Hemorrhage\n\n    All MeSH Categories\n";
            let (name, entry_terms, tree_numbers) = parse_mesh_record(blob);
            assert_eq!(name, "Cerebral Hemorrhage");
            assert!(entry_terms.contains(&"Hemorrhage, Cerebral".to_string()));
            assert!(entry_terms.contains(&"Intracerebral Hemorrhage".to_string()));
            assert!(entry_terms.contains(&"Cerebral Parenchymal Hemorrhage".to_string()));
            assert_eq!(tree_numbers.len(), 2);
            assert!(tree_numbers.contains(&"C14.907.253.573.200".to_string()));
        }

        #[test]
        fn parse_descriptor_names_handles_multi_record_response() {
            let blob = "1: Cerebral Hemorrhage\nBlah blah\n\n2: Subarachnoid Hemorrhage\nMore blah\n\n3: Intracranial Hemorrhages, Hypertensive\n";
            let names = parse_descriptor_names(blob);
            assert_eq!(names.len(), 3);
            assert_eq!(names[0], "Cerebral Hemorrhage");
            assert_eq!(names[1], "Subarachnoid Hemorrhage");
            assert_eq!(names[2], "Intracranial Hemorrhages, Hypertensive");
        }

        #[test]
        fn parse_mesh_record_handles_missing_entry_terms() {
            let blob = "1: ALOX15\nSome compound\n\nTree Number(s): D12.776.123\n";
            let (name, entry_terms, tree_numbers) = parse_mesh_record(blob);
            assert_eq!(name, "ALOX15");
            assert!(entry_terms.is_empty());
            assert_eq!(tree_numbers, vec!["D12.776.123".to_string()]);
        }

        #[test]
        fn share_token_catches_polysemous_mismaps() {
            // The exact live failure modes:
            assert!(
                !share_token("PD-L2 Ligand", "cell death"),
                "share_token should reject 'cell death' → 'PD-L2 Ligand'"
            );
            assert!(
                !share_token("AIFM2 protein, mouse", "ferroptosis"),
                "share_token should reject 'ferroptosis' → 'AIFM2 protein'"
            );
            // Legitimate matches:
            assert!(share_token("Cerebral Hemorrhage", "intracerebral hemorrhage"));
            assert!(share_token("Ferroptosis", "ferroptosis"));
            assert!(share_token("Neuron, Pacemaker", "neuron"));
        }

        #[test]
        fn share_token_ignores_short_words_and_stopwords() {
            // "the" and "of" alone don't trigger a match
            assert!(!share_token("Diseases of the Liver", "Of the"));
            // Short tokens (<3 chars) are ignored
            assert!(!share_token("PD L2", "ID 12"));
        }
    }
}

/// One tier in the multi-query expansion plan. Tiers are ordered
/// `T1` (most specific, 5-axis intersection) through `T5` (broadest,
/// single-axis sanity check). The blank_claim_check verdict reads
/// `T1`-`T3` results — anything caught at `T1` is a "claim already
/// has direct prior art", anything caught at `T2`/`T3` is "neighborhood
/// research exists, the '0 papers' framing is forbidden".
#[derive(Debug, Clone, Serialize)]
struct TieredQuery {
    /// Stable tier label (`T1` ... `T5`). Used as the key in the
    /// returned `tier_breakdown` map.
    label: String,
    /// Human-readable description of which axes intersected (e.g.
    /// "molecule × disease × mechanism").
    description: String,
    /// The PubMed `term=` parameter as it will be sent to eSearch.
    /// Logged in the report for transparency / audit.
    query: String,
}

/// Build the OR clause for a single axis: `(syn1[tiab] OR syn2[tiab])`.
/// `[tiab]` restricts to title + abstract — this trades some recall
/// for precision and avoids matching MeSH-only references where the
/// term is incidental rather than topical. (MeSH-tree expansion is
/// Layer 2's job; `[tiab]` is the right default for Layer 1.)
fn or_clause(axis_terms: &[String]) -> String {
    if axis_terms.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::with_capacity(axis_terms.len());
    for t in axis_terms {
        let trimmed = t.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Quote multi-word terms so PubMed treats them as phrases.
        // Single-token terms don't need quoting (and quoting them
        // would suppress PubMed's automatic term mapping for
        // single tokens like "ALOX15").
        if trimmed.contains(' ') {
            parts.push(format!("\"{trimmed}\"[tiab]"));
        } else {
            parts.push(format!("{trimmed}[tiab]"));
        }
    }
    if parts.is_empty() {
        String::new()
    } else if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        format!("({})", parts.join(" OR "))
    }
}

/// Build the AND clause for an intersection of axes. Axes whose terms
/// are all empty are skipped (so passing `model=[]` to a 4-axis
/// intersection that included `model` is equivalent to a 3-axis
/// intersection; this matters for Tier 2 drop-one-axis permutations
/// that include the optional axes).
fn and_clause(axes: &[&[String]]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(axes.len());
    for axis in axes {
        let clause = or_clause(axis);
        if !clause.is_empty() {
            parts.push(clause);
        }
    }
    parts.join(" AND ")
}

/// Generate the 11 standard tiered queries from a decomposed claim.
///
/// - T1 (1 query): AND of all 5 required axes.
/// - T2 (5 queries): AND of any 4 required axes (drop one at a time).
/// - T3 (2 queries): AND of (mol, disease, mech) and (mol, mech, cell).
/// - T4 (2 queries): AND of (mol, mech) and (mol, disease).
/// - T5 (1 query): molecule axis alone.
///
/// Optional `model` / `intervention` axes are used to refine T1 only
/// when populated (preserving recall on T2-T5 where they're absent).
fn build_queries(dim: &ClaimDimensions) -> Vec<TieredQuery> {
    let mut queries: Vec<TieredQuery> = Vec::new();

    // T1: 5-axis intersection (+ optional axes when present).
    let mut t1_axes: Vec<&[String]> = vec![
        &dim.molecule,
        &dim.disease,
        &dim.mechanism,
        &dim.cell_type,
        &dim.outcome,
    ];
    if !dim.model.is_empty() {
        t1_axes.push(&dim.model);
    }
    if !dim.intervention.is_empty() {
        t1_axes.push(&dim.intervention);
    }
    queries.push(TieredQuery {
        label: "T1".into(),
        description: "all required axes intersected (most specific)".into(),
        query: and_clause(&t1_axes),
    });

    // T2: drop-one-axis (5 queries; only the 5 required axes drop).
    type DropAxesFn = fn(&ClaimDimensions) -> Vec<&[String]>;
    let drop_targets: [(&str, DropAxesFn); 5] = [
        ("drop molecule", |d| {
            vec![&d.disease, &d.mechanism, &d.cell_type, &d.outcome]
        }),
        ("drop disease", |d| {
            vec![&d.molecule, &d.mechanism, &d.cell_type, &d.outcome]
        }),
        ("drop mechanism", |d| {
            vec![&d.molecule, &d.disease, &d.cell_type, &d.outcome]
        }),
        ("drop cell_type", |d| {
            vec![&d.molecule, &d.disease, &d.mechanism, &d.outcome]
        }),
        ("drop outcome", |d| {
            vec![&d.molecule, &d.disease, &d.mechanism, &d.cell_type]
        }),
    ];
    for (i, (descr, axes_fn)) in drop_targets.iter().enumerate() {
        queries.push(TieredQuery {
            label: format!("T2.{}", i + 1),
            description: (*descr).to_string(),
            query: and_clause(&axes_fn(dim)),
        });
    }

    // T3: 3-axis subsets that probe key conjunctions.
    queries.push(TieredQuery {
        label: "T3.1".into(),
        description: "molecule × disease × mechanism (core triangle)".into(),
        query: and_clause(&[&dim.molecule, &dim.disease, &dim.mechanism]),
    });
    queries.push(TieredQuery {
        label: "T3.2".into(),
        description: "molecule × mechanism × cell_type (mechanism-in-tissue)".into(),
        query: and_clause(&[&dim.molecule, &dim.mechanism, &dim.cell_type]),
    });

    // T4: 2-axis (broader, catches "molecule + mechanism in any
    // disease neighbor" and "molecule + disease without mechanism").
    queries.push(TieredQuery {
        label: "T4.1".into(),
        description: "molecule × mechanism (cross-disease neighbor)".into(),
        query: and_clause(&[&dim.molecule, &dim.mechanism]),
    });
    queries.push(TieredQuery {
        label: "T4.2".into(),
        description: "molecule × disease (any-mechanism neighbor)".into(),
        query: and_clause(&[&dim.molecule, &dim.disease]),
    });

    // T5: molecule axis alone (sanity-check upper bound).
    queries.push(TieredQuery {
        label: "T5".into(),
        description: "molecule alone (sanity-check upper bound)".into(),
        query: or_clause(&dim.molecule),
    });

    queries
}

/// Verdict on the LLM's "0 papers / complete blank" novelty claim.
#[derive(Debug, Clone, Serialize)]
struct BlankClaimVerdict {
    /// One of:
    ///  - `"ok_to_claim_blank"` — T1, T2 (all 5 drops), T3 (both
    ///    queries) all returned zero PMIDs. The LLM may write
    ///    "intersection unstudied".
    ///  - `"forbidden_neighborhood_exists"` — T1 zero but T2 or T3
    ///    non-empty. The LLM must NOT write "0 papers / complete
    ///    blank" and must reframe novelty around axes that are
    ///    actually unstudied.
    ///  - `"direct_prior_art"` — T1 itself non-empty. The LLM must
    ///    do the standard 5-dimension overlap analysis (this is the
    ///    pre-RFC behavior; the tool simply confirms the LLM is in
    ///    that branch).
    verdict: String,
    /// One-line rationale, suitable for inclusion in `disease_scan.md`
    /// as audit trail.
    rationale: String,
    /// PMIDs that triggered a `forbidden_neighborhood_exists` verdict
    /// (sample, capped at 10 per tier for report readability).
    sample_neighbor_pmids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TierResult {
    label: String,
    description: String,
    query: String,
    pmids: Vec<String>,
    error: Option<String>,
}

/// Per-axis record of what Layer 2 (MeSH) added to the LLM's
/// original axis term list. Surfaced in the report so the LLM
/// can audit "did MeSH widen recall on the disease axis to
/// include sibling diseases?" — and so failures are explicit
/// rather than silent.
#[derive(Debug, Clone, Default, Serialize)]
struct AxisMeshAudit {
    axis: String,
    input_term: String,
    added_synonyms: Vec<String>,
    added_siblings: Vec<String>,
    error: Option<String>,
    total_added: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
struct MeshAudit {
    enabled: bool,
    per_axis: Vec<AxisMeshAudit>,
}

impl MeshAudit {
    fn disabled() -> Self {
        Self {
            enabled: false,
            per_axis: Vec::new(),
        }
    }
}

pub struct PriorArtExpandTool {
    http: reqwest::Client,
    /// Optional NCBI eutils API key. With key, NIH allows 10 req/s
    /// instead of 3 req/s. Stored as `Option` because PharmaClaw
    /// deployments without an NCBI account still need the tool to
    /// work — they just run slower.
    api_key: Option<String>,
}

impl PriorArtExpandTool {
    pub fn new(api_key: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent("zeroclaw-pharmaclaw/prior-art-expand")
            .build()
            .expect("reqwest client builds with default config");
        Self { http, api_key }
    }

    /// Layer 2: expand each axis term through MeSH (synonyms +
    /// siblings). Returns the augmented dimensions and a per-axis
    /// audit trail recording what MeSH added for transparency.
    /// Errors per term are swallowed (logged into the audit) so a
    /// MeSH outage degrades gracefully to L1-only behavior.
    async fn expand_via_mesh(
        &self,
        original: &ClaimDimensions,
    ) -> (ClaimDimensions, MeshAudit) {
        let client = mesh::MeshClient::new(self.http.clone());
        let mut audit = MeshAudit::default();
        audit.enabled = true;

        let mut out = original.clone();
        // Expand each axis. Polite delay between terms is built
        // into MeshClient (350ms between paired esearch+efetch).
        for (axis_name, axis_terms) in [
            ("molecule", &mut out.molecule),
            ("disease", &mut out.disease),
            ("mechanism", &mut out.mechanism),
            ("cell_type", &mut out.cell_type),
            ("outcome", &mut out.outcome),
            ("model", &mut out.model),
            ("intervention", &mut out.intervention),
        ] {
            if axis_terms.is_empty() {
                continue;
            }
            // To bound API cost, expand only the first term per axis.
            // The remaining terms are kept verbatim. The first term
            // is canonically the most-specific synonym (the LLM is
            // instructed to put the canonical name first); MeSH's
            // sibling expansion off that anchor catches the
            // cross-disease neighbors that matter.
            let primary = axis_terms[0].clone();
            let expansion = client.expand_term(&primary).await;

            let mut added: Vec<String> = Vec::new();
            for syn in &expansion.synonyms {
                if !axis_terms.iter().any(|t| t.eq_ignore_ascii_case(syn)) {
                    axis_terms.push(syn.clone());
                    added.push(syn.clone());
                }
            }
            for sib in &expansion.siblings {
                if !axis_terms.iter().any(|t| t.eq_ignore_ascii_case(sib)) {
                    axis_terms.push(sib.clone());
                    added.push(sib.clone());
                }
            }

            audit.per_axis.push(AxisMeshAudit {
                axis: axis_name.into(),
                input_term: primary,
                added_synonyms: expansion.synonyms,
                added_siblings: expansion.siblings,
                error: expansion.error,
                total_added: added.len(),
            });
        }

        (out, audit)
    }

    /// Execute one PubMed eSearch with up to 2 retries on HTTP 429
    /// (NIH eutils rate limit). NIH allows 3 req/s without an API key
    /// and 10 req/s with one; the public limit is observed as a
    /// rolling 1-second window which an even 350ms cadence still
    /// occasionally trips on burst (eutils doesn't smooth bursts).
    /// On 429 we back off 1s then 2s, which empirically clears every
    /// observed limit in the v_zhang / v_li / v_gao_v5 regression
    /// suite. Other HTTP errors are returned immediately (no retry,
    /// since 4xx other than 429 won't fix themselves).
    async fn esearch(&self, term: &str) -> anyhow::Result<Vec<String>> {
        if term.trim().is_empty() {
            return Ok(Vec::new());
        }
        let mut params: Vec<(&str, String)> = vec![
            ("db", "pubmed".into()),
            ("term", term.to_string()),
            ("retmax", RETMAX_PER_QUERY.to_string()),
            ("retmode", "json".into()),
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
            let resp = match self.http.get(NCBI_ESEARCH_BASE).query(&params).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(format!("transport error: {e}"));
                    continue;
                }
            };
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.is_success() {
                let parsed: Value = serde_json::from_str(&body).map_err(|e| {
                    anyhow::anyhow!("eSearch returned non-JSON body: {e}; body: {body}")
                })?;
                let idlist = parsed
                    .get("esearchresult")
                    .and_then(|r| r.get("idlist"))
                    .and_then(|l| l.as_array())
                    .cloned()
                    .unwrap_or_default();
                return Ok(idlist
                    .into_iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect());
            }
            // 429 is the rate-limit retry target. Any other 4xx/5xx
            // is fail-fast — won't get better on retry, and chewing
            // up retry budget would just delay surfacing the real
            // error to SOP step 5.
            if status.as_u16() != 429 {
                anyhow::bail!("eSearch HTTP {status}: {body}");
            }
            last_err = Some(format!("HTTP 429 (attempt {}): {body}", attempt + 1));
        }
        anyhow::bail!(
            "eSearch failed after 3 attempts: {}",
            last_err.unwrap_or_else(|| "unknown error".into())
        )
    }

    /// Run all tiered queries serially. Serial (rather than parallel)
    /// keeps us within NIH's 3 req/s limit by default; with an api_key
    /// we could fan out, but the additional latency is bounded
    /// (~11 queries × ~300 ms = 3-4 s total) and easier to debug.
    async fn run_all_tiers(&self, queries: &[TieredQuery]) -> Vec<TierResult> {
        let mut results = Vec::with_capacity(queries.len());
        for q in queries {
            // Pause between requests to stay under NIH's 3 req/s
            // public limit. 500ms = 2 req/s gives margin against
            // burst trips observed at 350ms in the v_li / v_gao_v5
            // regression suite. Total per-claim overhead: 11×500ms
            // = 5.5s, dwarfed by the LLM thinking time elsewhere
            // in SOP step 5.
            if !results.is_empty() {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            let (pmids, err) = match self.esearch(&q.query).await {
                Ok(ids) => (ids, None),
                Err(e) => (Vec::new(), Some(e.to_string())),
            };
            results.push(TierResult {
                label: q.label.clone(),
                description: q.description.clone(),
                query: q.query.clone(),
                pmids,
                error: err,
            });
        }
        results
    }

    /// Decide whether the LLM may claim "0 papers / complete blank".
    /// See `BlankClaimVerdict` doc above for the rule.
    fn assess_blank_claim(tiers: &[TierResult]) -> BlankClaimVerdict {
        let by_label: BTreeMap<&str, &TierResult> =
            tiers.iter().map(|t| (t.label.as_str(), t)).collect();

        let t1_count = by_label.get("T1").map(|t| t.pmids.len()).unwrap_or(0);

        if t1_count > 0 {
            let sample = by_label
                .get("T1")
                .map(|t| t.pmids.iter().take(10).cloned().collect())
                .unwrap_or_default();
            return BlankClaimVerdict {
                verdict: "direct_prior_art".into(),
                rationale: format!(
                    "T1 (5-axis intersection) returned {} PMIDs — claim has direct prior art. \
                     Run standard M5.2 5-dimension overlap analysis instead of '0 papers' framing.",
                    t1_count
                ),
                sample_neighbor_pmids: sample,
            };
        }

        // T1 is zero. Check T2 (drop-one) and T3 (3-axis subsets).
        let mut neighborhood: BTreeSet<String> = BTreeSet::new();
        let mut neighborhood_tiers: Vec<&str> = Vec::new();
        for label_prefix in &["T2", "T3"] {
            for (lbl, tier) in &by_label {
                if lbl.starts_with(label_prefix) && !tier.pmids.is_empty() {
                    neighborhood_tiers.push(lbl);
                    for pmid in &tier.pmids {
                        neighborhood.insert(pmid.clone());
                    }
                }
            }
        }

        if neighborhood.is_empty() {
            return BlankClaimVerdict {
                verdict: "ok_to_claim_blank".into(),
                rationale: "T1 (5-axis), T2 (5×drop-one), T3 (2×3-axis) all returned zero PMIDs. \
                            Claiming 'intersection unstudied' is supported by neighborhood search. \
                            Note: T4 / T5 not part of this gate (they are intentionally broad and \
                            return cross-disease results that should not block a focused claim)."
                    .into(),
                sample_neighbor_pmids: Vec::new(),
            };
        }

        let mut sample: Vec<String> = neighborhood.iter().take(10).cloned().collect();
        sample.sort();

        BlankClaimVerdict {
            verdict: "forbidden_neighborhood_exists".into(),
            rationale: format!(
                "T1 (5-axis) returned 0 but neighborhood research exists: {} unique PMID(s) \
                 across tiers [{}]. The LLM is FORBIDDEN from writing '0 papers / complete \
                 blank / 完全空白'. Reframe novelty around the specific (cell_type / outcome / \
                 intervention) axes that are unstudied at the T2/T3 level. Sample neighborhood \
                 PMIDs (capped at 10): {}",
                neighborhood.len(),
                neighborhood_tiers.join(", "),
                sample.join(", ")
            ),
            sample_neighbor_pmids: sample,
        }
    }

    fn render_report(
        claim: &str,
        original_dim: &ClaimDimensions,
        expanded_dim: &ClaimDimensions,
        mesh_audit: &MeshAudit,
        tiers: &[TierResult],
        verdict: &BlankClaimVerdict,
    ) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        let _ = writeln!(
            out,
            "# Prior Art Neighborhood Expansion (Layer 1{})\n",
            if mesh_audit.enabled { " + 2" } else { "" }
        );
        let _ = writeln!(out, "**Claim**: {claim}\n");

        // Dimension echo so the LLM can audit its own decomposition.
        // Show the expanded version (what queries actually used) so the
        // tier breakdown below makes sense to the reader.
        let _ = writeln!(out, "## Decomposed axes (post-expansion)\n");
        let _ = writeln!(out, "| axis | terms |");
        let _ = writeln!(out, "|------|-------|");
        let _ = writeln!(out, "| molecule | {} |", expanded_dim.molecule.join(", "));
        let _ = writeln!(out, "| disease | {} |", expanded_dim.disease.join(", "));
        let _ = writeln!(out, "| mechanism | {} |", expanded_dim.mechanism.join(", "));
        let _ = writeln!(out, "| cell_type | {} |", expanded_dim.cell_type.join(", "));
        let _ = writeln!(out, "| outcome | {} |", expanded_dim.outcome.join(", "));
        if !expanded_dim.model.is_empty() {
            let _ = writeln!(out, "| model | {} |", expanded_dim.model.join(", "));
        }
        if !expanded_dim.intervention.is_empty() {
            let _ = writeln!(
                out,
                "| intervention | {} |",
                expanded_dim.intervention.join(", ")
            );
        }
        let _ = writeln!(out);

        // L2 audit: per-axis MeSH expansion record. Only emit when L2
        // ran (otherwise the section is noise).
        if mesh_audit.enabled {
            let total_added: usize =
                mesh_audit.per_axis.iter().map(|a| a.total_added).sum();
            let _ = writeln!(out, "## Layer 2: MeSH ontology expansion\n");
            let _ = writeln!(
                out,
                "MeSH expansion added **{total_added} term(s)** across {} axes (synonyms + tree siblings of the primary axis term).\n",
                mesh_audit.per_axis.len()
            );
            let _ = writeln!(out, "| axis | input | added synonyms | added siblings | error |");
            let _ = writeln!(out, "|------|-------|----------------|----------------|-------|");
            for a in &mesh_audit.per_axis {
                let syn = if a.added_synonyms.is_empty() {
                    "_(none)_".to_string()
                } else {
                    a.added_synonyms.join(", ")
                };
                let sib = if a.added_siblings.is_empty() {
                    "_(none)_".to_string()
                } else {
                    a.added_siblings.join(", ")
                };
                let err = a
                    .error
                    .as_deref()
                    .map(|e| format!("⚠️ {e}"))
                    .unwrap_or_default();
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} |",
                    a.axis, a.input_term, syn, sib, err
                );
            }
            let _ = writeln!(out);

            // Surface what the LLM ORIGINALLY decomposed, so reviewers
            // can confirm L2 only widened recall and didn't replace
            // the LLM's choices.
            let _ = writeln!(out, "<details><summary>Original LLM-decomposed axes (pre-expansion)</summary>\n");
            let _ = writeln!(out, "| axis | terms |");
            let _ = writeln!(out, "|------|-------|");
            let _ = writeln!(out, "| molecule | {} |", original_dim.molecule.join(", "));
            let _ = writeln!(out, "| disease | {} |", original_dim.disease.join(", "));
            let _ = writeln!(out, "| mechanism | {} |", original_dim.mechanism.join(", "));
            let _ = writeln!(out, "| cell_type | {} |", original_dim.cell_type.join(", "));
            let _ = writeln!(out, "| outcome | {} |", original_dim.outcome.join(", "));
            if !original_dim.model.is_empty() {
                let _ = writeln!(out, "| model | {} |", original_dim.model.join(", "));
            }
            if !original_dim.intervention.is_empty() {
                let _ = writeln!(
                    out,
                    "| intervention | {} |",
                    original_dim.intervention.join(", ")
                );
            }
            let _ = writeln!(out, "\n</details>\n");
        }

        // Verdict comes first — this is the deterministic gate the SOP
        // step 5 prompt acts on.
        let _ = writeln!(out, "## Blank-claim verdict\n");
        let _ = writeln!(out, "**Verdict**: `{}`\n", verdict.verdict);
        let _ = writeln!(out, "{}\n", verdict.rationale);

        // Per-tier detail.
        let _ = writeln!(out, "## Tier breakdown\n");
        let _ = writeln!(out, "| tier | description | hits | sample PMIDs |");
        let _ = writeln!(out, "|------|-------------|-----:|--------------|");
        for tier in tiers {
            let sample: Vec<String> = tier.pmids.iter().take(5).cloned().collect();
            let sample_str = if sample.is_empty() {
                if tier.error.is_some() {
                    "_(error)_".to_string()
                } else {
                    "_(none)_".to_string()
                }
            } else {
                sample.join(", ")
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} |",
                tier.label,
                tier.description,
                tier.pmids.len(),
                sample_str
            );
        }
        let _ = writeln!(out);

        // Surface query strings for audit / reproducibility.
        let _ = writeln!(out, "## Queries used (for audit)\n");
        for tier in tiers {
            let _ = writeln!(out, "- **{}**: `{}`", tier.label, tier.query);
            if let Some(err) = &tier.error {
                let _ = writeln!(out, "  - ⚠️ error: {err}");
            }
        }

        out
    }
}

#[async_trait]
impl Tool for PriorArtExpandTool {
    fn name(&self) -> &str {
        "prior_art_expand"
    }

    fn description(&self) -> &str {
        "Expand a research claim into 11 tiered PubMed eSearch queries (5-axis \
         intersection through molecule-only) and run them in parallel against NCBI \
         eutils. Returns: (a) the de-duplicated PMID set across all tiers, (b) a \
         per-tier breakdown for transparency, and (c) a deterministic \
         `blank_claim_check` verdict the SOP step 5 prompt MUST consult before \
         allowing the LLM to write '0 papers / complete blank / 完全空白'. \
         Inputs: `claim` (the natural-language research claim, used for the report \
         header), `dimensions` (an object with the LLM's decomposition into axes \
         molecule/disease/mechanism/cell_type/outcome [required] and model/intervention \
         [optional]; each axis is a list of synonym strings to OR together). \
         Verdict semantics: `direct_prior_art` (T1 hit, run standard overlap analysis), \
         `forbidden_neighborhood_exists` (T1 zero but T2/T3 hit, the '0 papers' \
         framing is forbidden — reframe novelty around unstudied sub-axes), \
         `ok_to_claim_blank` (T1/T2/T3 all zero, blank-intersection claim is supported). \
         Closes the v_zhang ICH+ALOX15 '0 papers' overstatement and the v_li \
         GZFL+dienogest already-published vs novel contradiction."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The natural-language research claim. Used only as the report header; the dimensions field carries the actual search payload."
                },
                "dimensions": {
                    "type": "object",
                    "description": "LLM-decomposed claim axes. Required: molecule, disease, mechanism, cell_type, outcome. Optional: model, intervention. Each axis is a list of synonym strings; the tool ORs them within an axis and ANDs across axes.",
                    "properties": {
                        "molecule":     {"type": "array", "items": {"type": "string"}},
                        "disease":      {"type": "array", "items": {"type": "string"}},
                        "mechanism":    {"type": "array", "items": {"type": "string"}},
                        "cell_type":    {"type": "array", "items": {"type": "string"}},
                        "outcome":      {"type": "array", "items": {"type": "string"}},
                        "model":        {"type": "array", "items": {"type": "string"}},
                        "intervention": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["molecule", "disease", "mechanism", "cell_type", "outcome"]
                }
            },
            "required": ["claim", "dimensions"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let claim = args
            .get("claim")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let dim_value = args
            .get("dimensions")
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: dimensions"))?;
        let original_dim: ClaimDimensions = serde_json::from_value(dim_value.clone())
            .map_err(|e| {
                anyhow::anyhow!("dimensions field could not be parsed as ClaimDimensions: {e}")
            })?;

        if !original_dim.required_axes_present() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "dimensions is missing required axes: {:?}. SOP step 5 must populate \
                     molecule, disease, mechanism, cell_type, outcome before calling \
                     prior_art_expand.",
                    original_dim.missing_required()
                )),
            });
        }

        // Layer 2: MeSH ontology expansion. Default ON because the
        // failure mode this fixes (recall on alternative phrasings)
        // is the v_zhang case's path to catching SAH / cerebral
        // infarction. Disable via `enable_mesh_expansion: false`
        // for tests that want to isolate L1 behavior.
        let enable_mesh = args
            .get("enable_mesh_expansion")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let (dim, mesh_audit) = if enable_mesh {
            self.expand_via_mesh(&original_dim).await
        } else {
            (original_dim.clone(), MeshAudit::disabled())
        };

        let queries = build_queries(&dim);
        let tiers = self.run_all_tiers(&queries).await;
        let verdict = Self::assess_blank_claim(&tiers);
        let report =
            Self::render_report(&claim, &original_dim, &dim, &mesh_audit, &tiers, &verdict);

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

    fn dim_full() -> ClaimDimensions {
        ClaimDimensions {
            molecule: vec!["ALOX15".into(), "15-LOX".into()],
            disease: vec!["intracerebral hemorrhage".into(), "ICH".into()],
            mechanism: vec!["ferroptosis".into()],
            cell_type: vec!["neuron".into(), "neuronal".into()],
            outcome: vec!["cell death".into()],
            model: vec![],
            intervention: vec![],
        }
    }

    #[test]
    fn or_clause_quotes_multiword_terms() {
        let s = or_clause(&["ALOX15".to_string(), "intracerebral hemorrhage".to_string()]);
        assert!(s.contains("ALOX15[tiab]"));
        assert!(s.contains("\"intracerebral hemorrhage\"[tiab]"));
        assert!(s.starts_with('('));
        assert!(s.contains(" OR "));
    }

    #[test]
    fn or_clause_handles_single_term() {
        let s = or_clause(&["ALOX15".to_string()]);
        // Single-term axis must not be wrapped in parentheses (the
        // outer AND in `and_clause` already groups across axes).
        assert_eq!(s, "ALOX15[tiab]");
    }

    #[test]
    fn or_clause_handles_empty() {
        let s = or_clause(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn and_clause_skips_empty_axes() {
        let mol = vec!["ALOX15".to_string()];
        let empty: Vec<String> = vec![];
        let dis = vec!["ICH".to_string()];
        let s = and_clause(&[&mol, &empty, &dis]);
        // Empty axis must not produce stray "AND" or empty parens.
        assert!(!s.contains(" AND  AND "));
        assert!(!s.contains("()"));
        assert!(s.contains("ALOX15[tiab]"));
        assert!(s.contains("ICH[tiab]"));
    }

    #[test]
    fn build_queries_emits_eleven_tiers_for_required_axes_only() {
        let dim = dim_full();
        let qs = build_queries(&dim);
        // 1 (T1) + 5 (T2 drop-one) + 2 (T3) + 2 (T4) + 1 (T5) = 11.
        assert_eq!(qs.len(), 11);
        assert_eq!(qs[0].label, "T1");
        assert!(qs.iter().any(|q| q.label == "T2.1"));
        assert!(qs.iter().any(|q| q.label == "T2.5"));
        assert!(qs.iter().any(|q| q.label == "T3.1"));
        assert!(qs.iter().any(|q| q.label == "T3.2"));
        assert!(qs.iter().any(|q| q.label == "T4.1"));
        assert!(qs.iter().any(|q| q.label == "T4.2"));
        assert!(qs.iter().any(|q| q.label == "T5"));
    }

    #[test]
    fn build_queries_t1_includes_optional_axes_when_populated() {
        let mut dim = dim_full();
        dim.model = vec!["mouse model".into()];
        dim.intervention = vec!["knockout".into()];
        let qs = build_queries(&dim);
        let t1 = qs.iter().find(|q| q.label == "T1").unwrap();
        // T1 should reference the optional axes.
        assert!(t1.query.contains("\"mouse model\"[tiab]"));
        assert!(t1.query.contains("knockout[tiab]"));
        // T2 drop-one queries only iterate over the 5 required axes,
        // so optional axes never appear there (recall preservation).
        for q in &qs {
            if q.label.starts_with("T2") {
                assert!(
                    !q.query.contains("\"mouse model\""),
                    "T2 must not include optional axes: {}",
                    q.query
                );
            }
        }
    }

    #[test]
    fn build_queries_t5_is_molecule_only() {
        let dim = dim_full();
        let qs = build_queries(&dim);
        let t5 = qs.iter().find(|q| q.label == "T5").unwrap();
        // T5 is just OR of molecule synonyms — no AND.
        assert!(!t5.query.contains(" AND "));
        assert!(t5.query.contains("ALOX15[tiab]"));
    }

    #[test]
    fn assess_blank_claim_t1_hit_means_direct_prior_art() {
        let tiers = vec![TierResult {
            label: "T1".into(),
            description: "5-axis".into(),
            query: "...".into(),
            pmids: vec!["12345".into(), "67890".into()],
            error: None,
        }];
        let v = PriorArtExpandTool::assess_blank_claim(&tiers);
        assert_eq!(v.verdict, "direct_prior_art");
        assert!(v.rationale.contains("2 PMIDs"));
        assert_eq!(v.sample_neighbor_pmids.len(), 2);
    }

    #[test]
    fn assess_blank_claim_t1_zero_t2_hit_forbids_blank_claim() {
        // The exact v_zhang ICH+ALOX15 failure mode: 5-axis query is
        // empty, but the drop-disease query catches stroke / SAH
        // neighbors. The LLM must NOT claim "0 papers".
        let tiers = vec![
            TierResult {
                label: "T1".into(),
                description: "5-axis".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
            TierResult {
                label: "T2.2".into(),
                description: "drop disease".into(),
                query: "...".into(),
                pmids: vec!["35186185".into(), "35090880".into()],
                error: None,
            },
            TierResult {
                label: "T3.1".into(),
                description: "mol×dis×mech".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
        ];
        let v = PriorArtExpandTool::assess_blank_claim(&tiers);
        assert_eq!(v.verdict, "forbidden_neighborhood_exists");
        assert!(v.rationale.contains("FORBIDDEN"));
        assert!(v.rationale.contains("35090880"));
        assert!(v.rationale.contains("35186185"));
        assert_eq!(v.sample_neighbor_pmids.len(), 2);
    }

    #[test]
    fn assess_blank_claim_all_zero_permits_blank_claim() {
        let tiers = vec![
            TierResult {
                label: "T1".into(),
                description: "5-axis".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
            TierResult {
                label: "T2.1".into(),
                description: "drop molecule".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
            TierResult {
                label: "T3.1".into(),
                description: "mol×dis×mech".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
        ];
        let v = PriorArtExpandTool::assess_blank_claim(&tiers);
        assert_eq!(v.verdict, "ok_to_claim_blank");
    }

    #[test]
    fn assess_blank_claim_ignores_t4_t5_when_t1_t3_are_clean() {
        // T4 / T5 are intentionally broad (catches molecule alone
        // across all diseases, e.g. ALOX15 in cancer / atherosclerosis
        // / asthma). They are the upper-bound sanity check, NOT a
        // gate for the blank-claim verdict — otherwise a focused
        // disease-specific claim would always be blocked by the
        // molecule's broader literature.
        let tiers = vec![
            TierResult {
                label: "T1".into(),
                description: "5-axis".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
            TierResult {
                label: "T2.1".into(),
                description: "drop molecule".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
            TierResult {
                label: "T3.1".into(),
                description: "mol×dis×mech".into(),
                query: "...".into(),
                pmids: vec![],
                error: None,
            },
            TierResult {
                label: "T4.1".into(),
                description: "mol×mech".into(),
                query: "...".into(),
                pmids: vec!["aaa".into(), "bbb".into()],
                error: None,
            },
            TierResult {
                label: "T5".into(),
                description: "molecule alone".into(),
                query: "...".into(),
                pmids: (0..50).map(|i| format!("p{i}")).collect(),
                error: None,
            },
        ];
        let v = PriorArtExpandTool::assess_blank_claim(&tiers);
        assert_eq!(
            v.verdict, "ok_to_claim_blank",
            "T4/T5 must not flip the verdict"
        );
    }

    #[test]
    fn missing_required_axis_returns_clear_error_message() {
        let mut dim = dim_full();
        dim.outcome = vec![];
        dim.cell_type = vec![];
        let missing = dim.missing_required();
        assert!(missing.contains(&"outcome"));
        assert!(missing.contains(&"cell_type"));
        assert!(!dim.required_axes_present());
    }

    #[test]
    fn render_report_contains_all_required_sections() {
        let dim = dim_full();
        let tiers = vec![TierResult {
            label: "T1".into(),
            description: "5-axis".into(),
            query: "ALOX15[tiab] AND ICH[tiab]".into(),
            pmids: vec!["12345".into()],
            error: None,
        }];
        let verdict = BlankClaimVerdict {
            verdict: "direct_prior_art".into(),
            rationale: "T1 returned 1".into(),
            sample_neighbor_pmids: vec!["12345".into()],
        };
        let mesh = MeshAudit::disabled();
        let r = PriorArtExpandTool::render_report(
            "ALOX15 in ICH ferroptosis",
            &dim,
            &dim,
            &mesh,
            &tiers,
            &verdict,
        );
        assert!(r.contains("# Prior Art Neighborhood Expansion (Layer 1)"));
        assert!(r.contains("**Claim**: ALOX15 in ICH ferroptosis"));
        assert!(r.contains("## Decomposed axes"));
        assert!(r.contains("## Blank-claim verdict"));
        assert!(r.contains("`direct_prior_art`"));
        assert!(r.contains("## Tier breakdown"));
        assert!(r.contains("## Queries used (for audit)"));
        assert!(r.contains("ALOX15[tiab] AND ICH[tiab]"));
        // Mesh disabled → no L2 section header.
        assert!(!r.contains("## Layer 2: MeSH ontology expansion"));
    }

    #[test]
    fn render_report_with_mesh_includes_layer_2_section() {
        let dim = dim_full();
        let tiers = vec![TierResult {
            label: "T1".into(),
            description: "5-axis".into(),
            query: "...".into(),
            pmids: vec![],
            error: None,
        }];
        let verdict = BlankClaimVerdict {
            verdict: "ok_to_claim_blank".into(),
            rationale: "all zero".into(),
            sample_neighbor_pmids: vec![],
        };
        let mesh = MeshAudit {
            enabled: true,
            per_axis: vec![AxisMeshAudit {
                axis: "disease".into(),
                input_term: "ICH".into(),
                added_synonyms: vec!["Cerebral Hemorrhage".into()],
                added_siblings: vec!["Subarachnoid Hemorrhage".into()],
                error: None,
                total_added: 2,
            }],
        };
        let r = PriorArtExpandTool::render_report("x", &dim, &dim, &mesh, &tiers, &verdict);
        assert!(r.contains("Layer 1 + 2"));
        assert!(r.contains("## Layer 2: MeSH ontology expansion"));
        assert!(r.contains("Cerebral Hemorrhage"));
        assert!(r.contains("Subarachnoid Hemorrhage"));
        assert!(r.contains("Original LLM-decomposed axes"));
    }

    #[test]
    fn render_report_marks_errored_tiers() {
        let dim = dim_full();
        let tiers = vec![TierResult {
            label: "T1".into(),
            description: "5-axis".into(),
            query: "...".into(),
            pmids: vec![],
            error: Some("HTTP 502".into()),
        }];
        let verdict = BlankClaimVerdict {
            verdict: "ok_to_claim_blank".into(),
            rationale: "all zero".into(),
            sample_neighbor_pmids: vec![],
        };
        let mesh = MeshAudit::disabled();
        let r =
            PriorArtExpandTool::render_report("x", &dim, &dim, &mesh, &tiers, &verdict);
        assert!(r.contains("_(error)_"));
        assert!(r.contains("HTTP 502"));
    }
}
