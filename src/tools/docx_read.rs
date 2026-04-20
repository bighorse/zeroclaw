use super::traits::{Tool, ToolResult};
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Maximum DOCX file size (25 MB).
const MAX_DOCX_BYTES: u64 = 25 * 1024 * 1024;
/// Default character limit returned to the LLM.
const DEFAULT_MAX_CHARS: usize = 50_000;
/// Hard ceiling regardless of what the caller requests.
const MAX_OUTPUT_CHARS: usize = 200_000;

/// Extract plain text from a DOCX (Word) file in the workspace.
///
/// DOCX extraction requires the `rag-docx` feature flag:
///   cargo build --features rag-docx
///
/// Without the feature the tool is still registered so the LLM receives a
/// clear, actionable error rather than a missing-tool confusion.
pub struct DocxReadTool {
    security: Arc<SecurityPolicy>,
}

impl DocxReadTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self { security }
    }
}

#[async_trait]
impl Tool for DocxReadTool {
    fn name(&self) -> &str {
        "docx_read"
    }

    fn description(&self) -> &str {
        "Extract plain text from a DOCX (Word) file in the workspace. \
         Returns paragraph-separated text from the main document body. \
         Requires the 'rag-docx' build feature."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the DOCX file. Relative paths resolve from workspace; outside paths require policy allowlist."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 50000, max: 200000)",
                    "minimum": 1,
                    "maximum": 200_000
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?;

        let max_chars = args
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .map(|n| {
                usize::try_from(n)
                    .unwrap_or(MAX_OUTPUT_CHARS)
                    .min(MAX_OUTPUT_CHARS)
            })
            .unwrap_or(DEFAULT_MAX_CHARS);

        if self.security.is_rate_limited() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded: too many actions in the last hour".into()),
            });
        }

        if !self.security.is_path_allowed(path) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Path not allowed by security policy: {path}")),
            });
        }

        // Record action before canonicalization so path-probing still consumes budget.
        if !self.security.record_action() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded: action budget exhausted".into()),
            });
        }

        let full_path = self.security.workspace_dir.join(path);

        let resolved_path = match tokio::fs::canonicalize(&full_path).await {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to resolve file path: {e}")),
                });
            }
        };

        if !self.security.is_resolved_path_allowed(&resolved_path) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    self.security
                        .resolved_path_violation_message(&resolved_path),
                ),
            });
        }

        tracing::debug!("Reading DOCX: {}", resolved_path.display());

        match tokio::fs::metadata(&resolved_path).await {
            Ok(meta) => {
                if meta.len() > MAX_DOCX_BYTES {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!(
                            "DOCX too large: {} bytes (limit: {MAX_DOCX_BYTES} bytes)",
                            meta.len()
                        )),
                    });
                }
            }
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to read file metadata: {e}")),
                });
            }
        }

        let bytes = match tokio::fs::read(&resolved_path).await {
            Ok(b) => b,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to read DOCX file: {e}")),
                });
            }
        };

        #[cfg(feature = "rag-docx")]
        {
            // zip + quick-xml parsing is CPU-bound; keep it off the async reactor.
            let text = match tokio::task::spawn_blocking(move || extract_docx_text(&bytes)).await {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("DOCX extraction failed: {e}")),
                    });
                }
                Err(e) => {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("DOCX extraction task panicked: {e}")),
                    });
                }
            };

            if text.trim().is_empty() {
                return Ok(ToolResult {
                    success: true,
                    output: "DOCX contains no extractable text (may be empty or image-only)".into(),
                    error: None,
                });
            }

            let output = if text.chars().count() > max_chars {
                let mut truncated: String = text.chars().take(max_chars).collect();
                use std::fmt::Write as _;
                let _ = write!(truncated, "\n\n... [truncated at {max_chars} chars]");
                truncated
            } else {
                text
            };

            return Ok(ToolResult {
                success: true,
                output,
                error: None,
            });
        }

        #[cfg(not(feature = "rag-docx"))]
        {
            let _ = bytes;
            let _ = max_chars;
            Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "DOCX extraction is not enabled. \
                     Rebuild with: cargo build --features rag-docx"
                        .into(),
                ),
            })
        }
    }
}

/// Extract plain text from the main body of a DOCX document.
///
/// DOCX is a ZIP archive with an XML part at `word/document.xml`. Body text
/// lives inside `<w:t>` runs, grouped by `<w:p>` paragraphs. This walks the
/// XML in a single pass, concatenating text runs and emitting a newline at
/// each paragraph boundary.
///
/// Deliberate limitations (out of scope for the initial PR):
/// - Headers, footers, footnotes, comments are in separate parts and skipped.
/// - Tables are flattened to their cell text without layout.
/// - Inline shapes, drawings, embedded objects are omitted.
#[cfg(feature = "rag-docx")]
fn extract_docx_text(bytes: &[u8]) -> anyhow::Result<String> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;
    use std::io::Read;

    let reader = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| anyhow::anyhow!("not a valid DOCX (ZIP) file: {e}"))?;

    let mut xml = String::new();
    {
        let mut doc = archive
            .by_name("word/document.xml")
            .map_err(|e| anyhow::anyhow!("DOCX missing word/document.xml: {e}"))?;
        doc.read_to_string(&mut xml)
            .map_err(|e| anyhow::anyhow!("failed to read word/document.xml: {e}"))?;
    }

    let mut reader = Reader::from_str(&xml);
    reader.config_mut().trim_text(false);

    let mut out = String::with_capacity(xml.len() / 4);
    let mut in_text_run = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = e.name();
                let local = name.as_ref();
                if local == b"w:t" || ends_with_local(local, b":t") || local == b"t" {
                    in_text_run = true;
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name();
                let local = name.as_ref();
                if local == b"w:t" || ends_with_local(local, b":t") || local == b"t" {
                    in_text_run = false;
                } else if local == b"w:p"
                    || ends_with_local(local, b":p")
                    || local == b"p"
                    || local == b"w:br"
                    || ends_with_local(local, b":br")
                {
                    out.push('\n');
                }
            }
            Ok(Event::Text(t)) if in_text_run => {
                let decoded = t
                    .unescape()
                    .map_err(|e| anyhow::anyhow!("malformed XML text: {e}"))?;
                out.push_str(&decoded);
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!("XML parse error: {e}")),
            _ => {}
        }
        buf.clear();
    }

    Ok(out)
}

/// True when `name` ends with `suffix` (typically `":t"` or `":p"`), to
/// match namespaced element names without hard-coding the `w:` prefix.
#[cfg(feature = "rag-docx")]
fn ends_with_local(name: &[u8], suffix: &[u8]) -> bool {
    name.len() > suffix.len() && name.ends_with(suffix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{AutonomyLevel, SecurityPolicy};
    use tempfile::TempDir;

    fn test_security(workspace: std::path::PathBuf) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: workspace,
            ..SecurityPolicy::default()
        })
    }

    fn test_security_with_limit(
        workspace: std::path::PathBuf,
        max_actions: u32,
    ) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: workspace,
            max_actions_per_hour: max_actions,
            ..SecurityPolicy::default()
        })
    }

    #[test]
    fn name_is_docx_read() {
        let tool = DocxReadTool::new(test_security(std::env::temp_dir()));
        assert_eq!(tool.name(), "docx_read");
    }

    #[test]
    fn description_not_empty() {
        let tool = DocxReadTool::new(test_security(std::env::temp_dir()));
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn schema_has_path_required() {
        let tool = DocxReadTool::new(test_security(std::env::temp_dir()));
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["max_chars"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
    }

    #[tokio::test]
    async fn missing_path_param_returns_error() {
        let tool = DocxReadTool::new(test_security(std::env::temp_dir()));
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path"));
    }

    #[tokio::test]
    async fn absolute_path_is_blocked() {
        let tool = DocxReadTool::new(test_security(std::env::temp_dir()));
        let result = tool.execute(json!({"path": "/etc/passwd"})).await.unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("not allowed"));
    }

    #[tokio::test]
    async fn nonexistent_file_returns_error() {
        let tmp = TempDir::new().unwrap();
        let tool = DocxReadTool::new(test_security(tmp.path().to_path_buf()));
        let result = tool
            .execute(json!({"path": "does_not_exist.docx"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Failed to resolve"));
    }

    #[tokio::test]
    async fn rate_limit_blocks_request() {
        let tmp = TempDir::new().unwrap();
        let tool = DocxReadTool::new(test_security_with_limit(tmp.path().to_path_buf(), 0));
        let result = tool.execute(json!({"path": "any.docx"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("Rate limit"));
    }

    #[cfg(feature = "rag-docx")]
    mod extraction {
        use super::*;
        use std::io::Write;

        /// Build a minimal valid DOCX in memory containing the supplied paragraph texts.
        /// The ZIP layout only includes `word/document.xml` — enough for our extractor
        /// to walk; real DOCX files carry additional parts we deliberately ignore.
        fn make_docx_bytes(paragraphs: &[&str]) -> Vec<u8> {
            let mut doc = String::from(
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>"#,
            );
            for p in paragraphs {
                doc.push_str("<w:p><w:r><w:t>");
                // Escape only the XML specials that matter for the test fixtures.
                doc.push_str(&p.replace('&', "&amp;").replace('<', "&lt;"));
                doc.push_str("</w:t></w:r></w:p>");
            }
            doc.push_str("</w:body></w:document>");

            let buf = std::io::Cursor::new(Vec::new());
            let mut zipw = zip::ZipWriter::new(buf);
            let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zipw.start_file("word/document.xml", opts).unwrap();
            zipw.write_all(doc.as_bytes()).unwrap();
            zipw.finish().unwrap().into_inner()
        }

        #[tokio::test]
        async fn extracts_text_from_valid_docx() {
            let tmp = TempDir::new().unwrap();
            let docx_path = tmp.path().join("hello.docx");
            tokio::fs::write(&docx_path, make_docx_bytes(&["Hello DOCX", "Second line"]))
                .await
                .unwrap();

            let tool = DocxReadTool::new(test_security(tmp.path().to_path_buf()));
            let result = tool.execute(json!({"path": "hello.docx"})).await.unwrap();

            assert!(result.success, "error: {:?}", result.error);
            assert!(result.output.contains("Hello DOCX"));
            assert!(result.output.contains("Second line"));
            // Paragraphs separated by newline
            assert!(result.output.contains('\n'));
        }

        #[tokio::test]
        async fn max_chars_truncates_output() {
            let tmp = TempDir::new().unwrap();
            let long = "abcdefghij".repeat(20); // 200 chars
            tokio::fs::write(tmp.path().join("long.docx"), make_docx_bytes(&[&long]))
                .await
                .unwrap();

            let tool = DocxReadTool::new(test_security(tmp.path().to_path_buf()));
            let result = tool
                .execute(json!({"path": "long.docx", "max_chars": 20}))
                .await
                .unwrap();

            assert!(result.success);
            assert!(result.output.contains("truncated"));
        }

        #[tokio::test]
        async fn corrupt_docx_returns_extraction_error() {
            let tmp = TempDir::new().unwrap();
            tokio::fs::write(tmp.path().join("bad.docx"), b"not a zip archive")
                .await
                .unwrap();

            let tool = DocxReadTool::new(test_security(tmp.path().to_path_buf()));
            let result = tool.execute(json!({"path": "bad.docx"})).await.unwrap();

            assert!(!result.success);
            assert!(result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("extraction failed"));
        }
    }

    #[cfg(not(feature = "rag-docx"))]
    #[tokio::test]
    async fn without_feature_returns_clear_error() {
        let tmp = TempDir::new().unwrap();
        tokio::fs::write(tmp.path().join("doc.docx"), b"fake")
            .await
            .unwrap();

        let tool = DocxReadTool::new(test_security(tmp.path().to_path_buf()));
        let result = tool.execute(json!({"path": "doc.docx"})).await.unwrap();
        assert!(!result.success);
        assert!(
            result.error.as_deref().unwrap_or("").contains("rag-docx"),
            "expected feature hint in error, got: {:?}",
            result.error
        );
    }
}
