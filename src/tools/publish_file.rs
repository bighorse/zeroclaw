//! Explicit-intent tool for surfacing a workspace file to the user.
//!
//! Complements the extension-whitelist auto-detection in `file_write` /
//! `file_edit` / `shell` by giving the LLM an explicit channel to say
//! *"this file is a deliverable"* regardless of extension.
//!
//! See the design plan: "add `publish_file` tool" (plan file saved with
//! the bumblebee slug).
//!
//! ## Why this exists alongside auto-detection
//!
//! Auto-detection biases toward not spamming — extensions like `.md`,
//! `.json`, `.yaml`, source files, and logs never auto-emit artifacts.
//! That keeps skill-internal scratch files out of chat but also blocks
//! legitimate deliverables in those formats (clinical notes, data
//! exports). `publish_file` lets the LLM override the whitelist for a
//! single explicit path.
//!
//! ## Why not replace auto-detection entirely
//!
//! Would require every existing skill to learn the new pattern and the
//! system prompt to nudge it, both of which are out of scope. Coexist
//! keeps zero-change-for-callers as the default; explicit-intent is
//! opt-in.
//!
//! ## Security posture
//!
//! Whitelist bypass expands blast radius. Defences layered in order of
//! rejection speed (cheapest first):
//!
//! 1. Empty / traversal / absolute-outside-workspace (`is_path_allowed`).
//! 2. Canonicalized parent lands in an allowed root
//!    (`is_resolved_path_allowed`).
//! 3. Leaf is not a symlink (`symlink_metadata(...).is_symlink()`),
//!    mirroring `file_write`'s pattern — a `canonicalize(full_path)`
//!    would follow the leaf and silently defeat this check.
//! 4. Leaf is a regular file (not a directory / socket / device).
//! 5. Hardcoded sensitive-file denylist (`.env*`, `*.pem`, `*.key`,
//!    `*.sqlite*`, `id_rsa*`, `.git/**`, ...) — last line of defence
//!    against prompt injection or a skill bug steering the LLM at a
//!    secret. Cannot be overridden via config by design.
//! 6. Rate limit (`record_action`) — not just spam control; the
//!    downstream Lark `im/v1/files` API has per-app quota.

use super::artifact::{append_artifacts, Artifact};
use super::file_write::DownloadUrlConfig;
use super::traits::{Tool, ToolResult};
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Basenames that must never be published, whole-name match (lowercase).
const SENSITIVE_BASENAMES: &[&str] = &[
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
    "authorized_keys",
    "credentials",
];

/// Basename prefixes that signal sensitive content regardless of suffix.
/// `"id_rsa."` also blocks `id_rsa.pub` — users who genuinely want to
/// share an SSH public key should rename it away from the private-key
/// family. That trade-off is intentional.
const SENSITIVE_BASENAME_PREFIXES: &[&str] = &[
    ".env",
    "id_rsa.",
    "id_dsa.",
    "id_ecdsa.",
    "id_ed25519.",
    "credentials.",
];

/// Basename suffixes (extensions and compound suffixes) that signal
/// sensitive content.
const SENSITIVE_BASENAME_SUFFIXES: &[&str] = &[
    ".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db", ".secret",
];

/// Basename substrings that signal sensitive content (e.g. `secrets.json`,
/// `api_secrets.yaml`).
const SENSITIVE_BASENAME_CONTAINS: &[&str] = &["secrets."];

/// Decide whether a workspace-relative path names a sensitive file that
/// must never be surfaced to a chat client, regardless of LLM intent.
///
/// - `rel_str` is the workspace-relative path with `/` separators.
/// - `basename` is the final path component.
///
/// All comparisons are case-insensitive.
fn is_sensitive_path(rel_str: &str, basename: &str) -> bool {
    // `.git/` as a path segment (not a substring) — so `digitalgit/x`
    // remains publishable, but `.git/config` or `reports/.git/HEAD`
    // get rejected.
    if rel_str.split('/').any(|c| c.eq_ignore_ascii_case(".git")) {
        return true;
    }

    let b = basename.to_ascii_lowercase();

    if SENSITIVE_BASENAMES.iter().any(|n| b == *n) {
        return true;
    }
    if SENSITIVE_BASENAME_PREFIXES.iter().any(|p| b.starts_with(p)) {
        return true;
    }
    if SENSITIVE_BASENAME_SUFFIXES.iter().any(|s| b.ends_with(s)) {
        return true;
    }
    if SENSITIVE_BASENAME_CONTAINS.iter().any(|s| b.contains(s)) {
        return true;
    }

    false
}

/// Publish an existing workspace file as a user-facing deliverable.
pub struct PublishFileTool {
    security: Arc<SecurityPolicy>,
    download: Option<DownloadUrlConfig>,
}

impl PublishFileTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self {
            security,
            download: None,
        }
    }

    /// Construct with signed-URL generation enabled. Without this the tool
    /// still works (Lark native upload path uses `path`/`size_bytes`, not
    /// the URL), but deployments with a public gateway URL should pass it
    /// through so text-only channels get a clickable link.
    pub fn with_download(security: Arc<SecurityPolicy>, download: DownloadUrlConfig) -> Self {
        Self {
            security,
            download: Some(download),
        }
    }
}

#[async_trait]
impl Tool for PublishFileTool {
    fn name(&self) -> &str {
        "publish_file"
    }

    fn description(&self) -> &str {
        "Surface an existing workspace file to the user as a downloadable \
         attachment. Use this ONLY for files that would not otherwise be \
         auto-published: `.md` notes, `.json` / `.yaml` data exports, \
         `.txt` summaries, `.log` output, source code the user asked to \
         receive, or any non-standard extension. Do NOT call this for \
         office documents (.docx/.xlsx/.pptx), PDFs, CSVs, images, or \
         zips — those are auto-published by `file_write` / `shell` and \
         calling `publish_file` would produce a duplicate attachment. \
         The file must already exist; create it first with `file_write` \
         or `shell`, then publish in a separate call. Single file per \
         call. Does not create or modify files."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to an existing file."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // ── 1. Extract & sanity-check the path arg ────────────────
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?;

        if path.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Path cannot be empty".into()),
            });
        }

        // ── 2. Cheap security prefilter (traversal, absolute, etc.) ─
        if !self.security.is_path_allowed(path) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Path not allowed by security policy: {path}")),
            });
        }

        // ── 3. Canonicalize workspace + parent ──────────────────────
        //
        // Canonicalize workspace and parent separately. Canonicalizing the
        // FULL path here would follow a leaf symlink and make the later
        // `is_symlink` check dead code. See `file_write.rs:115-125` for the
        // exact pattern this mirrors.
        let canonical_workspace = tokio::fs::canonicalize(&self.security.workspace_dir)
            .await
            .unwrap_or_else(|_| self.security.workspace_dir.clone());

        let full_path = self.security.workspace_dir.join(path);
        let Some(parent) = full_path.parent() else {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Invalid path: missing parent directory".into()),
            });
        };

        let resolved_parent = match tokio::fs::canonicalize(parent).await {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to resolve path {path}: {e}")),
                });
            }
        };

        if !self.security.is_resolved_path_allowed(&resolved_parent) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    self.security
                        .resolved_path_violation_message(&resolved_parent),
                ),
            });
        }

        // ── 4. Resolve leaf + symlink/filetype checks ───────────────
        let Some(file_name) = full_path.file_name() else {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Invalid path: missing file name".into()),
            });
        };
        let resolved_target = resolved_parent.join(file_name);

        let meta = match tokio::fs::symlink_metadata(&resolved_target).await {
            Ok(m) => m,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("File not found: {path} ({e})")),
                });
            }
        };

        if meta.file_type().is_symlink() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Refusing to publish through symlink: {}",
                    resolved_target.display()
                )),
            });
        }
        if !meta.is_file() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Not a regular file: {path}")),
            });
        }

        // ── 5. Derive the canonical workspace-relative path ─────────
        //
        // Deriving from `resolved_target.strip_prefix(canonical_workspace)`
        // (rather than trusting the user-supplied `path` string) normalises
        // `./foo` prefixes, collapses redundant separators, and keeps the
        // signed URL + artifact metadata consistent on macOS where
        // `/var/folders/...` canonicalizes to `/private/var/folders/...`.
        let rel_str = resolved_target
            .strip_prefix(&canonical_workspace)
            .unwrap_or(&resolved_target)
            .to_string_lossy()
            .replace('\\', "/");

        // ── 6. Sensitive-file denylist ─────────────────────────────
        let basename = file_name.to_string_lossy();
        if is_sensitive_path(&rel_str, &basename) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Refusing to publish sensitive file: {path} (matches built-in denylist)"
                )),
            });
        }

        // ── 7. Rate limit ──────────────────────────────────────────
        if !self.security.record_action() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded: action budget exhausted".into()),
            });
        }

        // ── 8. Build signed URL (if configured) + artifact ─────────
        //
        // No whitelist gate here — bypass is the entire point of the tool.
        let download_url = self.download.as_ref().map(|dl| {
            crate::gateway::signed_url::sign_download_url(
                &dl.base_url,
                &rel_str,
                &dl.secret,
                crate::gateway::signed_url::DEFAULT_TTL_SECS,
            )
        });

        let mut output = format!("Published {rel_str} ({} bytes)", meta.len());
        if let Some(url) = download_url.as_deref() {
            use std::fmt::Write;
            let _ = write!(output, "\nDownload: {url}");
        }

        // Always emit the sentinel, even when `download_url` is None —
        // Lark's native-upload channel uses `path` + `size_bytes` to read
        // the file directly, it does not need the URL. Suppressing here
        // would break attachments on deployments without a public gateway.
        if let Some(artifact) =
            Artifact::from_workspace_path(&canonical_workspace, &rel_str, download_url)
        {
            append_artifacts(&mut output, std::slice::from_ref(&artifact));
        }

        Ok(ToolResult {
            success: true,
            output,
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::AutonomyLevel;
    use serde_json::json;

    fn test_security(workspace: std::path::PathBuf) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: workspace,
            ..SecurityPolicy::default()
        })
    }

    fn test_security_with_budget(
        workspace: std::path::PathBuf,
        max_actions_per_hour: u32,
    ) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: workspace,
            max_actions_per_hour,
            ..SecurityPolicy::default()
        })
    }

    fn sample_download_config() -> DownloadUrlConfig {
        DownloadUrlConfig {
            base_url: "https://gw.example".into(),
            secret: b"test-secret".to_vec(),
        }
    }

    // ── Positive / deliverable path ─────────────────────────────────

    /// Regression anchor: `.md` is NOT in `USER_DELIVERABLE_EXTENSIONS`
    /// (`file_write` would not surface it), but `publish_file` must.
    #[tokio::test]
    async fn publish_file_happy_path_markdown() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_md");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("notes.md"), "clinical note body")
            .await
            .unwrap();

        let tool =
            PublishFileTool::with_download(test_security(dir.clone()), sample_download_config());
        let result = tool.execute(json!({"path": "notes.md"})).await.unwrap();
        assert!(result.success, "unexpected failure: {:?}", result.error);
        assert!(result.output.contains("Published notes.md"));
        assert!(result
            .output
            .contains("\nDownload: https://gw.example/download/notes.md?expires="));

        // Sanity: confirm whitelist was actually bypassed (`.md` isn't on it)
        assert!(
            !crate::tools::artifact::is_artifact_extension("notes.md"),
            "test premise wrong — .md must not be on the auto-detect whitelist"
        );

        let (cleaned, artifacts) = crate::tools::artifact::extract_artifacts(&result.output);
        assert!(!cleaned.contains("zeroclaw-artifacts"));
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].path, "notes.md");
        assert_eq!(artifacts[0].name, "notes.md");
        assert_eq!(artifacts[0].size_bytes, 18);

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Completely arbitrary extension — whitelist is fully bypassed.
    #[tokio::test]
    async fn publish_file_happy_path_unknown_extension() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_xyz");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("report.xyz"), b"data")
            .await
            .unwrap();

        let tool =
            PublishFileTool::with_download(test_security(dir.clone()), sample_download_config());
        let result = tool.execute(json!({"path": "report.xyz"})).await.unwrap();
        assert!(result.success);
        let (_, artifacts) = crate::tools::artifact::extract_artifacts(&result.output);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].path, "report.xyz");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Without `with_download` the sentinel is still emitted (Lark native
    /// upload reads from the workspace; it does not need the URL).
    /// `Download:` line is absent in that configuration.
    #[tokio::test]
    async fn publish_file_without_download_emits_sentinel_only() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_no_dl");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("log.txt"), b"ok").await.unwrap();

        let tool = PublishFileTool::new(test_security(dir.clone()));
        let result = tool.execute(json!({"path": "log.txt"})).await.unwrap();
        assert!(result.success);
        assert!(!result.output.contains("Download:"));

        let (_, artifacts) = crate::tools::artifact::extract_artifacts(&result.output);
        assert_eq!(
            artifacts.len(),
            1,
            "sentinel must be emitted even without URL"
        );
        assert!(
            artifacts[0].download_url.is_none(),
            "download_url must be None when no gateway is configured"
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    /// Integration: `file_write` a `.md` (no auto-publish because `.md` is
    /// off-whitelist) → `publish_file` surfaces it → exactly one artifact
    /// in the combined message stream, no double-append.
    #[tokio::test]
    async fn publish_file_integration_with_file_write() {
        use crate::tools::FileWriteTool;

        let dir = std::env::temp_dir().join("zeroclaw_test_publish_integration");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let security = test_security(dir.clone());
        let write_tool = FileWriteTool::with_download(security.clone(), sample_download_config());
        let pub_tool = PublishFileTool::with_download(security.clone(), sample_download_config());

        let r1 = write_tool
            .execute(json!({"path": "draft.md", "content": "body"}))
            .await
            .unwrap();
        assert!(r1.success);
        // file_write emits nothing artifact-y for .md (whitelist)
        let (_, artifacts1) = crate::tools::artifact::extract_artifacts(&r1.output);
        assert!(
            artifacts1.is_empty(),
            "file_write must not auto-publish .md (regression guard)"
        );

        let r2 = pub_tool.execute(json!({"path": "draft.md"})).await.unwrap();
        assert!(r2.success);
        let (_, artifacts2) = crate::tools::artifact::extract_artifacts(&r2.output);
        assert_eq!(
            artifacts2.len(),
            1,
            "publish_file must emit exactly one artifact (no double-append)"
        );
        assert_eq!(artifacts2[0].path, "draft.md");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    // ── Security rejections ─────────────────────────────────────────

    #[tokio::test]
    async fn publish_file_rejects_missing_file() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_missing");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let tool = PublishFileTool::new(test_security(dir.clone()));
        let result = tool
            .execute(json!({"path": "does_not_exist.md"}))
            .await
            .unwrap();
        assert!(!result.success);
        let err = result.error.as_deref().unwrap_or("");
        assert!(
            err.contains("not found") || err.contains("Failed to resolve"),
            "error must indicate the file is missing: {err}"
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn publish_file_rejects_directory() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_dir");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(dir.join("sub")).await.unwrap();

        let tool = PublishFileTool::new(test_security(dir.clone()));
        let result = tool.execute(json!({"path": "sub"})).await.unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Not a regular file"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn publish_file_rejects_empty_path() {
        let tool = PublishFileTool::new(test_security(std::env::temp_dir()));
        let result = tool.execute(json!({"path": ""})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("empty"));
    }

    #[tokio::test]
    async fn publish_file_rejects_path_traversal() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_traversal");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let tool = PublishFileTool::new(test_security(dir.clone()));
        let result = tool
            .execute(json!({"path": "../../etc/passwd"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("not allowed"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn publish_file_rejects_absolute_path_outside_workspace() {
        let tool = PublishFileTool::new(test_security(std::env::temp_dir()));
        let result = tool.execute(json!({"path": "/etc/passwd"})).await.unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("not allowed"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn publish_file_blocks_symlink_leaf_to_outside() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join("zeroclaw_test_publish_symlink_out");
        let workspace = root.join("workspace");
        let outside = root.join("outside");

        let _ = tokio::fs::remove_dir_all(&root).await;
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::create_dir_all(&outside).await.unwrap();
        tokio::fs::write(outside.join("secret.docx"), b"contents")
            .await
            .unwrap();
        symlink(outside.join("secret.docx"), workspace.join("leaked.docx")).unwrap();

        let tool = PublishFileTool::new(test_security(workspace.clone()));
        let result = tool.execute(json!({"path": "leaked.docx"})).await.unwrap();
        assert!(!result.success, "symlink to outside must be refused");
        assert!(
            result.error.as_deref().unwrap_or("").contains("symlink"),
            "error must mention symlink: {:?}",
            result.error
        );

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    /// Symlink to a file INSIDE the workspace is also refused — matches
    /// `file_write` behaviour and prevents dedup confusion (two paths
    /// would point at the same artifact).
    #[cfg(unix)]
    #[tokio::test]
    async fn publish_file_blocks_symlink_leaf_to_inside() {
        use std::os::unix::fs::symlink;

        let workspace = std::env::temp_dir().join("zeroclaw_test_publish_symlink_in");
        let _ = tokio::fs::remove_dir_all(&workspace).await;
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("real.md"), b"x")
            .await
            .unwrap();
        symlink(workspace.join("real.md"), workspace.join("alias.md")).unwrap();

        let tool = PublishFileTool::new(test_security(workspace.clone()));
        let result = tool.execute(json!({"path": "alias.md"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("symlink"));

        let _ = tokio::fs::remove_dir_all(&workspace).await;
    }

    // ── Sensitive-file denylist ─────────────────────────────────────

    /// Helper: create a file and assert publish_file rejects it with
    /// "sensitive file" in the error.
    ///
    /// Uses `tempfile::TempDir` so every invocation gets a unique workspace
    /// — earlier iterations of this helper hand-rolled the directory name
    /// from the filename and flaked under concurrent test execution when
    /// two tests produced colliding slugs (e.g. `credentials` and
    /// `credentials.json` both map to the same slug after `.` replacement).
    async fn assert_sensitive_rejected(filename: &str, content: &[u8]) {
        let tmp = tempfile::TempDir::new().expect("create temp workspace");
        let dir = tmp.path().to_path_buf();
        // Create nested dirs if filename contains '/'.
        if let Some(parent) = std::path::Path::new(filename).parent() {
            tokio::fs::create_dir_all(dir.join(parent)).await.unwrap();
        }
        tokio::fs::write(dir.join(filename), content).await.unwrap();

        let tool = PublishFileTool::new(test_security(dir.clone()));
        let result = tool.execute(json!({"path": filename})).await.unwrap();
        assert!(
            !result.success,
            "{filename} should be rejected by denylist (got success=true, output={})",
            result.output
        );
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("sensitive file"),
            "{filename} rejection must mention 'sensitive file': {:?}",
            result.error
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn publish_file_rejects_dotenv() {
        for name in [".env", ".env.local", ".env.production"] {
            assert_sensitive_rejected(name, b"SECRET=hunter2").await;
        }
    }

    #[tokio::test]
    async fn publish_file_rejects_private_key() {
        // `id_rsa.pub` is deliberately caught by prefix `id_rsa.` — see the
        // const doc comment. Users who want to share a public key should
        // rename it outside the `id_rsa*` family first.
        for name in ["id_rsa", "id_rsa.pub", "id_ed25519"] {
            assert_sensitive_rejected(name, b"-----BEGIN PRIVATE KEY-----").await;
        }
    }

    #[tokio::test]
    async fn publish_file_rejects_pem_files() {
        for name in ["cert.pem", "server.key", "bundle.p12"] {
            assert_sensitive_rejected(name, b"binary").await;
        }
    }

    #[tokio::test]
    async fn publish_file_rejects_sqlite() {
        for name in ["cache.sqlite", "notes.db", "cache.sqlite3"] {
            assert_sensitive_rejected(name, b"\x53\x51\x4c").await;
        }
    }

    #[tokio::test]
    async fn publish_file_rejects_git_internals() {
        assert_sensitive_rejected(".git/config", b"[core]\n").await;
        assert_sensitive_rejected("reports/.git/objects/x", b"blob").await;
    }

    #[tokio::test]
    async fn publish_file_rejects_credentials_and_secrets() {
        assert_sensitive_rejected("credentials", b"user=x").await;
        assert_sensitive_rejected("credentials.json", b"{}").await;
        assert_sensitive_rejected("secrets.yaml", b"x: 1").await;
        assert_sensitive_rejected("api.secret", b"x").await;
    }

    #[tokio::test]
    async fn publish_file_denylist_case_insensitive() {
        for name in ["CACHE.SQLITE", "ID_RSA", ".Env.Local", "Credentials.JSON"] {
            assert_sensitive_rejected(name, b"x").await;
        }
    }

    // ── Path resolution edge cases ──────────────────────────────────

    /// `./notes.md` must land the same artifact path as `notes.md`.
    #[tokio::test]
    async fn publish_file_normalises_dot_slash() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_dotslash");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("notes.md"), b"body")
            .await
            .unwrap();

        let tool =
            PublishFileTool::with_download(test_security(dir.clone()), sample_download_config());
        let result = tool.execute(json!({"path": "./notes.md"})).await.unwrap();
        assert!(result.success, "unexpected failure: {:?}", result.error);
        let (_, artifacts) = crate::tools::artifact::extract_artifacts(&result.output);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(
            artifacts[0].path, "notes.md",
            "artifact.path must be normalised without ./ prefix"
        );
        let url = artifacts[0].download_url.as_deref().unwrap();
        assert!(
            url.contains("/download/notes.md?"),
            "signed URL must not embed ./: {url}"
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn publish_file_respects_rate_limit() {
        let dir = std::env::temp_dir().join("zeroclaw_test_publish_rate");
        let _ = tokio::fs::remove_dir_all(&dir).await;
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("report.md"), b"x").await.unwrap();

        // Zero budget — any action-recording call must fail.
        let tool = PublishFileTool::new(test_security_with_budget(dir.clone(), 0));
        let result = tool.execute(json!({"path": "report.md"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("Rate limit"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    // ── Trait metadata ──────────────────────────────────────────────

    #[test]
    fn publish_file_tool_name_and_schema() {
        let tool = PublishFileTool::new(test_security(std::env::temp_dir()));
        assert_eq!(tool.name(), "publish_file");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["path"].is_object());
        assert_eq!(
            schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["path"]
        );
    }

    #[test]
    fn publish_file_description_warns_against_duplicate_usage() {
        let tool = PublishFileTool::new(test_security(std::env::temp_dir()));
        let desc = tool.description();
        // Behavioural anchor: the description MUST tell the LLM not to
        // publish already-auto-detected formats. Without that hint Claude
        // will double-attach docx files.
        assert!(desc.contains("Do NOT call this for office documents"));
        assert!(desc.contains("duplicate"));
    }
}
