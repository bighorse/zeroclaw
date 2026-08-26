//! Tool-produced file artifacts: a structured contract for downloadable outputs.
//!
//! ## Why this exists
//!
//! Tools that materialise files in the workspace (`file_write`, `file_edit`,
//! `shell` running `pandoc`/`docx-js`/`soffice`/etc.) historically signalled
//! downloadable results by appending `Download: <signed-url>` lines to their
//! output string. Channel layers (Lark, Telegram, …) then regex-scanned the
//! text to surface a download button.
//!
//! That worked for `file_write` (single text file, one URL) but broke for
//! anything binary produced via `shell` — the `docx`/`pptx`/`xlsx` skills
//! invoke Node/Python through `shell` and never touch `file_write`, so no
//! `Download:` line was ever produced.
//!
//! ## The contract
//!
//! Tools opt-in by appending an `<!--zeroclaw-artifacts:[ARTIFACT_JSON]-->`
//! sentinel block to the tail of [`ToolResult::output`]. The block is
//! invisible in plain-text rendering and survives JSON round-tripping.
//!
//! Downstream consumers (the agent loop in `src/channels/mod.rs`, planned in
//! PR 2) call [`extract_artifacts`] to (a) recover the structured artifact
//! list and (b) get a cleaned text without the sentinel before forwarding
//! the message into LLM context or to the channel.
//!
//! ## Why a sentinel rather than a `ToolResult` field
//!
//! `ToolResult` is constructed at 346 call-sites across 40 tool files. Adding
//! a required field would force a mechanical change touching every site and
//! mix a contract refactor with unrelated edits — violating the "one concern
//! per PR" rule in `CLAUDE.md`. The sentinel keeps PR 1 to ~5 files and lets
//! PR 2 wire the consumer side without further touching tool code.
//!
//! ## Forward path
//!
//! - PR 2 strips sentinels from tool-result messages **before** they enter
//!   LLM context (so the model never sees or imitates them) and forwards the
//!   parsed `Vec<Artifact>` to channels via a new
//!   `Channel::send_with_artifacts` default method.
//! - PR 3 implements `Channel::send_with_artifacts` for Lark using the
//!   `im/v1/files` upload API so users see a native attachment card.
//! - Once stable, the sentinel can be promoted to a real `ToolResult.artifacts`
//!   field as a follow-up cleanup (separate PR, mechanical-only diff).

use serde::{Deserialize, Serialize};

/// A file produced by a tool that the user may want to download.
///
/// Fields are deliberately minimal: anything channel-layer specific
/// (Lark file_key, Telegram file_id, …) is computed *during* upload, not
/// stored here.
/// Media semantics of an [`Artifact`], independent of its concrete MIME type.
///
/// Channels render the same bytes differently depending on this: an image
/// belongs inline (Lark `msg_type=image`, Telegram `sendPhoto`), a document
/// belongs in a download card. Deciding this once, where the file is
/// produced, keeps every channel from re-deriving (and disagreeing about)
/// the classification.
///
/// ## Why all four variants ship together
///
/// The variant set is part of the sentinel wire format. Unknown *fields* are
/// ignored by serde, but an unknown *variant* is a hard parse error, and
/// [`extract_artifacts`] then drops **every** artifact in that block. Adding
/// `Audio`/`Video` later would be a breaking change for rollbacks and
/// mixed-version fleets; adding them now costs two match arms. Lark's own
/// upload classifier (`lark_file_type_for`) already speaks `opus` and `mp4`,
/// so they are not hypothetical - merely unreachable from
/// [`mime_for_extension`] today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactKind {
    /// Chat clients render this inline (png/jpeg/webp/gif/svg).
    Image,
    /// Anything the user downloads and opens elsewhere. The conservative
    /// default: it reproduces the pre-`kind` behaviour (a file attachment),
    /// so an unclassifiable artifact degrades to exactly today's UX.
    #[default]
    Document,
    /// Playable audio (voice notes, recordings).
    Audio,
    /// Playable video.
    Video,
}

impl ArtifactKind {
    /// Classify by MIME top-level type. Returns `None` for top-level types
    /// that carry no media semantics (`application/*`, `text/*`), leaving
    /// the decision to the caller's fallback.
    pub fn from_mime(mime: &str) -> Option<Self> {
        match mime.split('/').next()?.trim().to_ascii_lowercase().as_str() {
            "image" => Some(Self::Image),
            "audio" => Some(Self::Audio),
            "video" => Some(Self::Video),
            _ => None,
        }
    }

    /// Best-effort classification, in decreasing order of confidence: the
    /// explicit `mime`, then a MIME re-derived from the path extension, then
    /// [`ArtifactKind::Document`].
    ///
    /// The path fallback is not redundant: `mime` is
    /// `skip_serializing_if = "Option::is_none"`, so a sentinel can legitimately
    /// carry a path and no MIME at all.
    pub fn infer(mime: Option<&str>, path: &str) -> Self {
        if let Some(kind) = mime.and_then(Self::from_mime) {
            return kind;
        }
        let from_path = mime_for_extension(path);
        from_path
            .as_deref()
            .and_then(Self::from_mime)
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "ArtifactWire")]
pub struct Artifact {
    /// Workspace-relative path. Always uses `/` as separator.
    pub path: String,
    /// Display name (typically the file's basename).
    pub name: String,
    /// MIME type guessed from extension. Optional — some channels don't need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    /// File size in bytes at the time the tool produced it.
    pub size_bytes: u64,
    /// Pre-signed download URL if the gateway is configured. `None` when the
    /// agent runs without a public gateway URL — channels that support native
    /// upload (Lark file API, Telegram `sendDocument`) can still attach the
    /// file by reading it from `path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,
    /// How the artifact should be presented. Always serialized (unlike the
    /// `Option` fields) so a producer's explicit classification is never
    /// silently re-derived on the consumer side.
    ///
    /// Declared last on purpose: `serde_json` emits fields in declaration
    /// order, so every pre-existing key keeps its position in the sentinel.
    pub kind: ArtifactKind,
}

impl Artifact {
    /// Build an artifact for a workspace-relative path.
    ///
    /// `size_bytes` is read from the file system. Returns `None` if the file
    /// does not exist or cannot be stat'd — callers should silently skip
    /// rather than failing the tool call.
    pub fn from_workspace_path(
        workspace_dir: &std::path::Path,
        relative_path: &str,
        download_url: Option<String>,
    ) -> Option<Self> {
        let full = workspace_dir.join(relative_path);
        let meta = std::fs::metadata(&full).ok()?;
        if !meta.is_file() {
            return None;
        }
        let name = std::path::Path::new(relative_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| relative_path.to_string());
        let mime = mime_for_extension(relative_path);
        let kind = ArtifactKind::infer(mime.as_deref(), relative_path);
        Some(Self {
            path: relative_path.replace('\\', "/"),
            name,
            mime,
            size_bytes: meta.len(),
            download_url,
            kind,
        })
    }
}

/// On-the-wire mirror of [`Artifact`], used only for deserialization.
///
/// ## Why not `#[serde(default)]` on `Artifact::kind`
///
/// A field-level `default` supplies a constant - it cannot see sibling
/// fields - so every sentinel written before `kind` existed would decode to
/// `ArtifactKind::Document`, including images replayed from persisted
/// history. Routing through this struct makes "`kind` absent" mean "infer it
/// from `mime`/`path`", which is what the field means, while still meeting
/// the hard requirement that a missing `kind` must never fail the parse:
/// [`extract_artifacts`] discards the entire artifact list on any
/// deserialization error, so a non-tolerant field would silently delete
/// every attachment produced by an older binary.
///
/// Unknown extra fields are ignored (deliberately no `deny_unknown_fields`)
/// so a future field is likewise non-breaking. The exhaustive destructuring
/// in `From<ArtifactWire> for Artifact` is the drift guard: adding a field to
/// either struct without updating the other fails to compile.
#[derive(Deserialize)]
struct ArtifactWire {
    path: String,
    name: String,
    #[serde(default)]
    mime: Option<String>,
    size_bytes: u64,
    #[serde(default)]
    download_url: Option<String>,
    #[serde(default)]
    kind: Option<ArtifactKind>,
}

impl From<ArtifactWire> for Artifact {
    fn from(wire: ArtifactWire) -> Self {
        let ArtifactWire {
            path,
            name,
            mime,
            size_bytes,
            download_url,
            kind,
        } = wire;
        let kind = kind.unwrap_or_else(|| ArtifactKind::infer(mime.as_deref(), &path));
        Self {
            path,
            name,
            mime,
            size_bytes,
            download_url,
            kind,
        }
    }
}

/// File extensions considered **user-facing deliverables** — the final output
/// a human user wants to see, as opposed to intermediate files (scripts,
/// configs, schemas, logs, temp artifacts) generated during skill execution.
///
/// ## Why this list is narrow
///
/// Skills frequently `file_write` a Python/JS/shell script, then `shell` runs
/// it to produce the real deliverable. Before the narrowing done here, the
/// intermediate script was surfaced to the chat client alongside the real
/// deliverable — users saw two attachments (`generate.py` + `report.docx`)
/// where they only wanted one.
///
/// ## Rules of thumb for what belongs
///
/// - Something a non-technical recipient (doctor, manager) would expect to
///   open with Word/Excel/PDF viewer/image viewer.
/// - NOT something another program is likely to consume (json/xml/yaml).
/// - NOT something the chat client can already render inline (md/txt/html).
/// - Archive formats only when the skill explicitly packages a deliverable
///   (`zip` kept; `tar`/`gz` dropped — rarer and less portable).
///
/// False negatives here mean "no attachment appears" — users can always
/// retrieve the file through other means (the workspace is still mounted
/// on disk). False positives mean spammed chats and, historically, leaked
/// internal scripts. Bias strongly toward false negatives.
///
/// ## Contents (grouped by purpose)
///
/// - Office documents: `docx`, `doc`, `pptx`, `ppt`, `xlsx`, `xls`
/// - Reports: `pdf`
/// - Data exports: `csv` (often the final deliverable for data-pipeline skills)
/// - Images: `png`, `jpg`, `jpeg`, `webp`
/// - Archives: `zip` (for skills that package multi-file deliverables)
pub const USER_DELIVERABLE_EXTENSIONS: &[&str] = &[
    "docx", "doc", "pptx", "ppt", "xlsx", "xls", "pdf", "csv", "png", "jpg", "jpeg", "webp", "zip",
];

/// True if the path's lowercase extension is in [`USER_DELIVERABLE_EXTENSIONS`].
pub fn is_artifact_extension(path: &str) -> bool {
    path.rsplit('.')
        .next()
        .map(str::to_ascii_lowercase)
        .is_some_and(|ext| USER_DELIVERABLE_EXTENSIONS.contains(&ext.as_str()))
}

/// Best-effort MIME guess from extension. Mirrors `gateway::guess_content_type`
/// but returns `Option<String>` so artifacts without a confident match stay
/// `None` rather than defaulting to `application/octet-stream`.
pub fn mime_for_extension(path: &str) -> Option<String> {
    let ext = path.rsplit('.').next()?.to_ascii_lowercase();
    let mime = match ext.as_str() {
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc" => "application/msword",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt" => "application/vnd.ms-powerpoint",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls" => "application/vnd.ms-excel",
        "pdf" => "application/pdf",
        "csv" => "text/csv; charset=utf-8",
        "md" => "text/markdown; charset=utf-8",
        "txt" => "text/plain; charset=utf-8",
        "html" | "htm" => "text/html; charset=utf-8",
        "json" => "application/json",
        "xml" => "application/xml",
        "yaml" | "yml" => "application/yaml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "zip" => "application/zip",
        "tar" => "application/x-tar",
        "gz" => "application/gzip",
        "rtf" => "application/rtf",
        _ => return None,
    };
    Some(mime.to_string())
}

// ── Sentinel codec ─────────────────────────────────────────────────

/// Marker that opens an artifact block. Begins with two newlines so it never
/// fuses to preceding text and is easy to strip cleanly.
const SENTINEL_OPEN: &str = "\n\n<!--zeroclaw-artifacts:";
/// Marker that closes an artifact block.
const SENTINEL_CLOSE: &str = "-->";

/// Append an artifact list to the tail of a tool's `output` string.
///
/// No-op when `artifacts` is empty (keeps output clean for tools that
/// occasionally produce nothing downloadable).
///
/// The encoded form is `\n\n<!--zeroclaw-artifacts:[JSON]-->`. The leading
/// `\n\n` ensures the sentinel is on its own paragraph; HTML-comment syntax
/// keeps it invisible in Markdown renderers as a defence in depth in case
/// any consumer forgets to call [`extract_artifacts`].
pub fn append_artifacts(output: &mut String, artifacts: &[Artifact]) {
    if artifacts.is_empty() {
        return;
    }
    let json = match serde_json::to_string(artifacts) {
        Ok(j) => j,
        Err(_) => return, // serialization can't realistically fail for plain data; swallow rather than poison output
    };
    output.push_str(SENTINEL_OPEN);
    output.push_str(&json);
    output.push_str(SENTINEL_CLOSE);
}

/// Recover artifacts from a tool output string, returning `(cleaned_text, artifacts)`.
///
/// - Removes the sentinel block from the returned text.
/// - Returns an empty `Vec` and the unchanged text when no sentinel is present
///   or when the embedded JSON fails to parse (defensive: a malformed
///   sentinel must not cause data loss).
/// - Only the **last** sentinel block is recognised. Tools that call
///   [`append_artifacts`] more than once should batch their artifacts into
///   one call.
pub fn extract_artifacts(text: &str) -> (String, Vec<Artifact>) {
    let Some(open_at) = text.rfind(SENTINEL_OPEN) else {
        return (text.to_string(), Vec::new());
    };
    let json_start = open_at + SENTINEL_OPEN.len();
    let Some(close_rel) = text[json_start..].find(SENTINEL_CLOSE) else {
        return (text.to_string(), Vec::new());
    };
    let json = &text[json_start..json_start + close_rel];
    let Ok(artifacts) = serde_json::from_str::<Vec<Artifact>>(json) else {
        return (text.to_string(), Vec::new());
    };
    let mut cleaned = String::with_capacity(text.len());
    cleaned.push_str(&text[..open_at]);
    cleaned.push_str(&text[json_start + close_rel + SENTINEL_CLOSE.len()..]);
    (cleaned, artifacts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_artifact() -> Artifact {
        Artifact {
            path: "reports/q1.docx".into(),
            name: "q1.docx".into(),
            mime: Some(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(),
            ),
            size_bytes: 12_345,
            download_url: Some("https://gw/download/reports%2Fq1.docx?expires=1&sig=abc".into()),
            kind: ArtifactKind::Document,
        }
    }

    /// CORE BACK-COMPAT GUARD. A sentinel produced before `kind` existed has
    /// no `kind` key. It must still parse: `extract_artifacts` drops the
    /// whole artifact list on any parse error, so a field that is not
    /// tolerant of absence silently deletes every attachment produced by an
    /// older binary or replayed from persisted history.
    #[test]
    fn legacy_sentinel_without_kind_parses_and_infers_from_mime() {
        let json = serde_json::json!([{
            "path": "charts/chart.png",
            "name": "chart.png",
            "mime": "image/png",
            "size_bytes": 42
        }])
        .to_string();
        let text = format!("Wrote chart.png\n\n<!--zeroclaw-artifacts:{json}-->");

        let (cleaned, arts) = extract_artifacts(&text);
        assert_eq!(cleaned, "Wrote chart.png");
        assert_eq!(arts.len(), 1, "legacy sentinel must not be dropped");
        assert_eq!(arts[0].path, "charts/chart.png");
        assert_eq!(arts[0].size_bytes, 42);
        assert_eq!(arts[0].download_url, None);
        assert_eq!(
            arts[0].kind,
            ArtifactKind::Image,
            "kind must be inferred from mime, not defaulted to Document"
        );
    }
    /// Second legacy shape: `mime` is `skip_serializing_if = "Option::is_none"`,
    /// so artifacts built by literals with `mime: None` (the shape used across
    /// the channel tests) serialize without a `mime` key either. Inference
    /// must then fall back to the path extension.
    #[test]
    fn legacy_sentinel_without_mime_infers_kind_from_path() {
        let json = serde_json::json!([
            { "path": "a/b.webp", "name": "b.webp", "size_bytes": 7 },
            { "path": "a/b.docx", "name": "b.docx", "size_bytes": 8 }
        ])
        .to_string();
        let text = format!("x\n\n<!--zeroclaw-artifacts:{json}-->");

        let (cleaned, arts) = extract_artifacts(&text);
        assert_eq!(cleaned, "x");
        assert_eq!(arts.len(), 2);
        assert_eq!(arts[0].mime, None);
        assert_eq!(arts[0].kind, ArtifactKind::Image);
        assert_eq!(arts[1].kind, ArtifactKind::Document);
    }
    /// An explicit `kind` must win over what the MIME implies. This is the
    /// reason `kind` is a stored field rather than a method on `Artifact`:
    /// a producer that knows better than the extension can say so, and the
    /// consumer must honour it.
    #[test]
    fn explicit_kind_overrides_mime_inference() {
        let json = serde_json::json!([{
            "path": "a.pdf",
            "name": "a.pdf",
            "mime": "application/pdf",
            "size_bytes": 1,
            "kind": "image"
        }])
        .to_string();
        let text = format!("x\n\n<!--zeroclaw-artifacts:{json}-->");

        let (_, arts) = extract_artifacts(&text);
        assert_eq!(arts.len(), 1);
        assert_eq!(arts[0].kind, ArtifactKind::Image);
    }
    /// The sentinel must carry `kind` explicitly so a producer's
    /// classification is not silently re-derived (and possibly re-derived
    /// differently) on the consumer side.
    #[test]
    fn append_emits_kind_and_roundtrips() {
        let mut art = sample_artifact();
        art.path = "charts/q1.png".into();
        art.name = "q1.png".into();
        art.mime = Some("image/png".into());
        art.kind = ArtifactKind::Image;

        let mut out = "chart".to_string();
        append_artifacts(&mut out, std::slice::from_ref(&art));
        assert!(
            out.contains("\"kind\":\"image\""),
            "sentinel must serialize kind: {out}"
        );

        let (cleaned, arts) = extract_artifacts(&out);
        assert_eq!(cleaned, "chart");
        assert_eq!(arts, vec![art]);
    }
    #[test]
    fn infer_kind_table() {
        let cases: &[(Option<&str>, &str, ArtifactKind)] = &[
            (Some("image/png"), "a.png", ArtifactKind::Image),
            (Some("image/svg+xml"), "a.svg", ArtifactKind::Image),
            (Some("IMAGE/PNG"), "a.png", ArtifactKind::Image),
            (Some("audio/opus"), "a.opus", ArtifactKind::Audio),
            (Some("video/mp4"), "a.mp4", ArtifactKind::Video),
            (Some("application/pdf"), "a.pdf", ArtifactKind::Document),
            (Some("application/zip"), "a.zip", ArtifactKind::Document),
            (
                Some("text/csv; charset=utf-8"),
                "a.csv",
                ArtifactKind::Document,
            ),
            // mime absent (older sentinels omit it): fall back to the path.
            (None, "a.jpeg", ArtifactKind::Image),
            (None, "a.webp", ArtifactKind::Image),
            (None, "a.docx", ArtifactKind::Document),
            // Neither mime nor a known extension: conservative default.
            (None, "a.unknown_ext", ArtifactKind::Document),
            (None, "no_extension", ArtifactKind::Document),
        ];
        for &(mime, path, expected) in cases {
            assert_eq!(
                ArtifactKind::infer(mime, path),
                expected,
                "infer(mime={mime:?}, path={path})"
            );
        }
    }
    /// The producer path must classify, not just the deserializer: a PNG
    /// written into the workspace has to arrive at the channel as
    /// [`ArtifactKind::Image`] or PR-3c's inline rendering never triggers.
    #[test]
    fn from_workspace_path_classifies_image_and_document() {
        let dir = std::env::temp_dir().join("zeroclaw_test_artifact_kind");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("chart.png"), [0x89, b'P', b'N', b'G']).unwrap();
        std::fs::write(dir.join("report.docx"), "x").unwrap();

        let png = Artifact::from_workspace_path(&dir, "chart.png", None).expect("png artifact");
        assert_eq!(png.kind, ArtifactKind::Image);
        let docx = Artifact::from_workspace_path(&dir, "report.docx", None).expect("docx artifact");
        assert_eq!(docx.kind, ArtifactKind::Document);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn append_then_extract_roundtrips() {
        let mut out = "Wrote q1.docx".to_string();
        append_artifacts(&mut out, &[sample_artifact()]);
        let (cleaned, arts) = extract_artifacts(&out);
        assert_eq!(cleaned, "Wrote q1.docx");
        assert_eq!(arts, vec![sample_artifact()]);
    }

    #[test]
    fn append_empty_is_noop() {
        let mut out = "nothing to attach".to_string();
        append_artifacts(&mut out, &[]);
        assert_eq!(out, "nothing to attach");
    }

    #[test]
    fn extract_without_sentinel_returns_original() {
        let (cleaned, arts) = extract_artifacts("plain text");
        assert_eq!(cleaned, "plain text");
        assert!(arts.is_empty());
    }

    #[test]
    fn extract_with_malformed_json_is_safe() {
        let text = "hello\n\n<!--zeroclaw-artifacts:not json-->";
        let (cleaned, arts) = extract_artifacts(text);
        // Defensive: malformed sentinel must not lose user-visible text
        assert_eq!(cleaned, text);
        assert!(arts.is_empty());
    }

    #[test]
    fn extract_handles_only_last_sentinel() {
        let mut out = "first".to_string();
        append_artifacts(&mut out, &[sample_artifact()]);
        out.push_str(" middle ");
        let mut second = sample_artifact();
        second.path = "other.pdf".into();
        second.name = "other.pdf".into();
        append_artifacts(&mut out, &[second.clone()]);
        let (_cleaned, arts) = extract_artifacts(&out);
        // Only the last block is recognised; first remains embedded
        assert_eq!(arts, vec![second]);
    }

    #[test]
    fn extension_whitelist_matches_user_deliverables() {
        assert!(is_artifact_extension("/tmp/foo.docx"));
        assert!(is_artifact_extension("a/b/c.PPTX")); // case-insensitive
        assert!(is_artifact_extension("report.pdf"));
        assert!(is_artifact_extension("data.csv"));
        assert!(is_artifact_extension("chart.png"));
        assert!(is_artifact_extension("bundle.zip"));
        assert!(!is_artifact_extension("script.py"));
        assert!(!is_artifact_extension("Cargo.lock"));
        assert!(!is_artifact_extension("no_extension_at_all"));
    }

    /// Regression guard: extensions removed from the whitelist during the
    /// narrowing pass must STAY removed. If any of these come back, real
    /// skill workflows immediately start spamming users with intermediate
    /// files — that regression is both invisible (tests pass) and
    /// high-impact (loud chat clients).
    ///
    /// Categories covered:
    /// - source scripts: py, js, ts, mjs, cjs, sh, bash
    /// - chat-renderable text: md, txt, html, htm
    /// - config/data interchange: json, xml, yaml, yml
    /// - execution logs: log
    /// - uncommon office/image: rtf, svg, gif
    /// - rare archive formats: tar, gz
    /// - build/package manifests: lock, toml
    #[test]
    fn extension_whitelist_excludes_intermediate_formats() {
        for ext in [
            "py", "js", "ts", "mjs", "cjs", "sh", "bash", "md", "txt", "html", "htm", "json",
            "xml", "yaml", "yml", "log", "rtf", "svg", "gif", "tar", "gz", "lock", "toml",
        ] {
            let path = format!("intermediate.{ext}");
            assert!(
                !is_artifact_extension(&path),
                "`{ext}` must NOT be a user deliverable (regression: skill intermediates will leak)"
            );
        }
    }

    #[test]
    fn mime_for_extension_known_office_formats() {
        assert_eq!(
            mime_for_extension("a.docx").as_deref(),
            Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        );
        assert_eq!(
            mime_for_extension("a.pdf").as_deref(),
            Some("application/pdf")
        );
        assert_eq!(mime_for_extension("a.unknown_ext"), None);
        assert_eq!(mime_for_extension("no_extension"), None);
    }

    #[test]
    fn from_workspace_path_reads_size() {
        let dir = std::env::temp_dir().join("zeroclaw_test_artifact_from_workspace");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hello.md"), "abc").unwrap();

        let art = Artifact::from_workspace_path(&dir, "hello.md", None).expect("artifact");
        assert_eq!(art.path, "hello.md");
        assert_eq!(art.name, "hello.md");
        assert_eq!(art.size_bytes, 3);
        assert_eq!(art.mime.as_deref(), Some("text/markdown; charset=utf-8"));
        assert!(art.download_url.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_workspace_path_returns_none_for_missing() {
        let dir = std::env::temp_dir().join("zeroclaw_test_artifact_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(Artifact::from_workspace_path(&dir, "nonexistent.docx", None).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_workspace_path_returns_none_for_directory() {
        let dir = std::env::temp_dir().join("zeroclaw_test_artifact_dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("subdir")).unwrap();
        assert!(Artifact::from_workspace_path(&dir, "subdir", None).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
