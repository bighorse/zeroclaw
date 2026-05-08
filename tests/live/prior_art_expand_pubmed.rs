//! Live regression test for `prior_art_expand` (Layer 1 PHC-RFC-2026-003).
//!
//! Hits the real NCBI eutils PubMed API. Skipped by default (requires
//! network); run with:
//!   cargo test --test test_live prior_art -- --ignored
//!
//! The three cases below correspond to the historical PharmaClaw
//! versions where the LLM made a "0 papers / already-published vs novel"
//! framing error that the multi-query expansion is designed to catch.
//! Each case sets up the LLM-decomposed dimensions exactly as the bot
//! would (or should) decompose them, then asserts the verdict.

use serde_json::json;
use zeroclaw::tools::traits::Tool;
use zeroclaw::tools::PriorArtExpandTool;

/// v_zhang case (2026-05-08, run-1778245403320-0002):
/// LLM wrote "PubMed 2021-2026: ICH+ALOX15 = 0 篇" and built the novelty
/// argument on that. Manual web verification found ≥3 directly
/// relevant papers (J Proteome Res 2025, PMID:35186185 SAH+ALOX15,
/// PMID:35090880 stroke+ALOX15). The verdict must be
/// `forbidden_neighborhood_exists` (not `ok_to_claim_blank`).
#[tokio::test]
#[ignore = "hits NCBI eutils — run with --ignored"]
async fn v_zhang_ich_alox15_must_flag_neighborhood() {
    let tool = PriorArtExpandTool::new(None);
    let args = json!({
        "claim": "ALOX-15 is the rate-limiting enzyme of neuronal ferroptosis after intracerebral hemorrhage",
        "dimensions": {
            "molecule":  ["ALOX15", "ALOX-15", "15-lipoxygenase", "15-LOX-1"],
            "disease":   ["intracerebral hemorrhage", "ICH", "cerebral hemorrhage"],
            "mechanism": ["ferroptosis", "lipid peroxidation"],
            "cell_type": ["neuron", "neuronal"],
            "outcome":   ["cell death", "neurodegeneration"]
        },
        "enable_mesh_expansion": true
    });
    let result = tool.execute(args).await.expect("tool runs");
    assert!(result.success, "tool failed: {:?}", result.error);
    let report = result.output;
    eprintln!("\n===== v_zhang report =====\n{report}\n=========================\n");

    // Hard requirement: the verdict must NOT be `ok_to_claim_blank`.
    // The "0 篇" claim was empirically wrong, and Layer 1 must surface
    // that. Verdict can be either `direct_prior_art` (T1 hit) or
    // `forbidden_neighborhood_exists` (T2/T3 hit) — both correctly
    // reject the LLM's framing.
    assert!(
        !report.contains("`ok_to_claim_blank`"),
        "verdict was ok_to_claim_blank but PMIDs 35186185 / 35090880 \
         are documented neighborhood papers; Layer 1 failed its primary \
         job. Report:\n{report}"
    );
    assert!(
        report.contains("`forbidden_neighborhood_exists`") || report.contains("`direct_prior_art`"),
        "expected forbidden_neighborhood_exists or direct_prior_art verdict; \
         report:\n{report}"
    );
}

/// v_li case (2026-05-08, run-1778224449218-0001):
/// LLM in §2.2 listed "Prior Art" with GZFL+dienogest meta-analysis
/// (PMID:39654212) at 32.5% overlap, but §3.3 联合用药方案 wrote "GZFL +
/// dienogest 临床前数据: 未发表协同 (本课题首次提供)" — directly
/// contradicting its own §2.2. Layer 1 must flag GZFL + dienogest +
/// endometriosis as a `direct_prior_art` neighborhood (T1 should hit
/// the meta-analysis directly).
#[tokio::test]
#[ignore = "hits NCBI eutils — run with --ignored"]
async fn v_li_gzfl_dienogest_must_flag_published_combination() {
    let tool = PriorArtExpandTool::new(None);
    let args = json!({
        "claim": "Guizhi Fuling combined with dienogest for endometriosis ferroptosis modulation (combination is novel)",
        "dimensions": {
            "molecule":  ["Guizhi Fuling", "GZFL", "Gui-Zhi-Fu-Ling", "dienogest"],
            "disease":   ["endometriosis", "endometriotic"],
            "mechanism": ["combination therapy", "synergy", "adjunct"],
            "cell_type": ["endometrium", "ectopic endometrium"],
            "outcome":   ["pain", "lesion", "recurrence"]
        }
    });
    let result = tool.execute(args).await.expect("tool runs");
    assert!(result.success, "tool failed: {:?}", result.error);
    let report = result.output;
    eprintln!("\n===== v_li report =====\n{report}\n=======================\n");

    // The combination has a published meta-analysis (PMID:39654212).
    // Verdict must NOT permit "first to combine" framing.
    assert!(
        !report.contains("`ok_to_claim_blank`"),
        "verdict was ok_to_claim_blank but PMID:39654212 (Qin Y 2024 \
         Medicine, GZFL+dienogest meta-analysis) is documented; Layer 1 \
         failed. Report:\n{report}"
    );
}

/// v_gao_v5 PMID:41179718 EASTERN protocol case:
/// LLM cited PMID:41179718 as supporting evidence for SGLT2i in
/// refractory ascites, but the paper is a *protocol* (no results yet).
/// This isn't strictly a Layer 1 failure mode (paper-type detection is
/// SOP-side), but the search should at least surface SGLT2i + ascites
/// neighborhood research so the LLM cannot claim "first to study"
/// without grappling with PMID:41179718. We test that T1/T2/T3 are
/// non-empty for this combination.
#[tokio::test]
#[ignore = "hits NCBI eutils — run with --ignored"]
async fn v_gao_v5_sglt2i_ascites_neighborhood_must_be_non_empty() {
    let tool = PriorArtExpandTool::new(None);
    let args = json!({
        "claim": "SGLT2 inhibitors as natriuretic adjunct for refractory ascites in cirrhosis",
        "dimensions": {
            "molecule":  ["SGLT2 inhibitor", "SGLT2i", "dapagliflozin", "empagliflozin"],
            "disease":   ["refractory ascites", "cirrhosis", "decompensated cirrhosis"],
            "mechanism": ["natriuresis", "sodium excretion", "diuretic"],
            "cell_type": ["proximal tubule", "kidney"],
            "outcome":   ["fluid overload", "ascites resolution"]
        }
    });
    let result = tool.execute(args).await.expect("tool runs");
    assert!(result.success, "tool failed: {:?}", result.error);
    let report = result.output;
    eprintln!("\n===== v_gao_v5 report =====\n{report}\n===========================\n");

    // SGLT2i + cirrhosis ascites is an actively published space
    // (PMID:39384712 Singh 2025 Dapa monotherapy in recurrent ascites,
    // PMID:41362018 Garcia-Pagan 2026 Zibotentan+Dapa in CTP-A,
    // PMID:41179718 EASTERN protocol). Verdict must NOT be blank.
    assert!(
        !report.contains("`ok_to_claim_blank`"),
        "verdict was ok_to_claim_blank but SGLT2i+ascites has multiple \
         published / in-flight studies; Layer 1 missed the neighborhood. \
         Report:\n{report}"
    );
}
