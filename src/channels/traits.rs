use async_trait::async_trait;

pub use crate::tools::Artifact;

/// A message received from or sent to a channel
#[derive(Debug, Clone)]
pub struct ChannelMessage {
    pub id: String,
    pub sender: String,
    pub reply_target: String,
    pub content: String,
    pub channel: String,
    pub timestamp: u64,
    /// Platform thread identifier (e.g. Slack `ts`, Discord thread ID).
    /// When set, replies should be posted as threaded responses.
    pub thread_ts: Option<String>,
}

/// Message to send through a channel
#[derive(Debug, Clone)]
pub struct SendMessage {
    pub content: String,
    pub recipient: String,
    pub subject: Option<String>,
    /// Platform thread identifier for threaded replies (e.g. Slack `thread_ts`).
    pub thread_ts: Option<String>,
}

impl SendMessage {
    /// Create a new message with content and recipient
    pub fn new(content: impl Into<String>, recipient: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            recipient: recipient.into(),
            subject: None,
            thread_ts: None,
        }
    }

    /// Create a new message with content, recipient, and subject
    pub fn with_subject(
        content: impl Into<String>,
        recipient: impl Into<String>,
        subject: impl Into<String>,
    ) -> Self {
        Self {
            content: content.into(),
            recipient: recipient.into(),
            subject: Some(subject.into()),
            thread_ts: None,
        }
    }

    /// Set the thread identifier for threaded replies.
    pub fn in_thread(mut self, thread_ts: Option<String>) -> Self {
        self.thread_ts = thread_ts;
        self
    }
}

/// Remove `Download: <url>` lines from `content` whose URL refers to any
/// workspace path in `uploaded_paths`. Other lines (including `Download:`
/// lines for artifacts that *failed* to upload) are preserved so they
/// fall back to the regex-based button rendering in Lark's
/// `extract_download_links`.
///
/// Matching is done against the percent-encoded path segment in the URL
/// path component, not the URL string itself — an LLM may normalise the
/// URL differently than our signed-URL generator did (reordering query
/// params, changing scheme, …), and path-based matching survives that.
///
/// Empty `uploaded_paths` is a no-op (returns `content` unchanged as
/// `String`), which keeps the caller's fast path trivial.
///
/// Lives here rather than in `lark.rs` because `lark` is behind
/// `#[cfg(feature = "channel-lark")]` and Telegram/Discord need the same
/// stripping to avoid handing the user a link *and* the file.
pub(crate) fn strip_download_lines_for_paths(content: &str, uploaded_paths: &[String]) -> String {
    if uploaded_paths.is_empty() {
        return content.to_string();
    }
    // Pre-compute percent-encoded forms once per call.
    let encoded: Vec<String> = uploaded_paths
        .iter()
        .map(|p| urlencoding::encode(p).into_owned())
        .collect();

    let mut out = String::with_capacity(content.len());
    let mut first = true;
    for line in content.lines() {
        let trimmed = line.trim_start();
        let is_matched_download = trimmed
            .strip_prefix("Download: ")
            .map(|url| {
                encoded
                    .iter()
                    .any(|p| url.contains(format!("/download/{p}").as_str()))
            })
            .unwrap_or(false);
        if is_matched_download {
            continue;
        }
        if !first {
            out.push('\n');
        }
        out.push_str(line);
        first = false;
    }
    // Preserve a trailing newline if the original had one, since we stripped
    // line terminators via `.lines()`.
    if content.ends_with('\n') && !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Plain-text notice for artifacts a channel could not attach natively.
///
/// Returns `""` when there is nothing to say — the common case, so callers
/// keep a trivial fast path.
///
/// # Why this exists
///
/// The [`Channel::send_with_artifacts`] default used to discard `artifacts`
/// silently. That is invisible for `file_write` / `file_edit` (which also
/// emit a legacy `Download:` line the agent loop re-appends), but
/// `shell`-produced artifacts carry **no** `Download:` line at all, and
/// `publish_file` emits artifacts even with no gateway configured. On those
/// paths the user's `.docx` simply vanished.
///
/// # Never double up
///
/// `channels::handle_message` appends `Download: <url>` for every signed URL
/// in the turn's tool history that is not already in the reply text. An
/// artifact's `download_url` is the *same* `String` as the one on that line
/// (both come from one `sign_download_url` call in the tool), so the
/// containment check below is exact rather than heuristic:
///
/// - URL already in `content` → say nothing;
/// - URL present but missing from `content` → re-surface it (the `shell` case);
/// - no URL at all → a placeholder naming the file (the no-gateway case).
///
/// The text is English on purpose: this default is channel-agnostic and every
/// non-Lark channel in this repo speaks English to users. Lark, the only
/// Chinese-facing channel, overrides `send_with_artifacts` and never reaches
/// this code.
pub(crate) fn artifact_fallback_notice(content: &str, artifacts: &[Artifact]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for artifact in artifacts {
        match artifact.download_url.as_deref() {
            Some(url) if content.contains(url) => {}
            Some(url) => lines.push(format!("Download: {url}")),
            None => {
                let location = if artifact.path == artifact.name {
                    String::new()
                } else {
                    format!(" (workspace path: {})", artifact.path)
                };
                lines.push(format!(
                    "[Attachment not sent: {} — this channel cannot upload files{location}]",
                    artifact.name
                ));
            }
        }
    }
    if lines.is_empty() {
        return String::new();
    }
    let joined = lines.join("\n");
    format!("\n{joined}")
}

/// Apply [`artifact_fallback_notice`] to a whole message.
///
/// Returns `None` when nothing needs to be added, so callers can send the
/// original message untouched instead of cloning it.
pub(crate) fn message_with_artifact_notice(
    message: &SendMessage,
    artifacts: &[Artifact],
) -> Option<SendMessage> {
    let notice = artifact_fallback_notice(&message.content, artifacts);
    if notice.is_empty() {
        return None;
    }
    let mut degraded = message.clone();
    degraded.content.push_str(&notice);
    Some(degraded)
}

/// Core channel trait — implement for any messaging platform
#[async_trait]
pub trait Channel: Send + Sync {
    /// Human-readable channel name
    fn name(&self) -> &str;

    /// Send a message through this channel
    async fn send(&self, message: &SendMessage) -> anyhow::Result<()>;

    /// Start listening for incoming messages (long-running)
    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()>;

    /// Check if channel is healthy
    async fn health_check(&self) -> bool {
        true
    }

    /// Send a message along with a list of tool-produced file artifacts.
    ///
    /// Channels that support native file attachments (Lark `im/v1/files`,
    /// Telegram `sendDocument`, Discord multipart uploads, …) override this
    /// and upload each artifact inline so the user sees a proper attachment
    /// rather than a bare link.
    ///
    /// The default implementation cannot upload anything. Rather than
    /// discarding `artifacts` silently — which made `shell`-produced files
    /// and gateway-less `publish_file` output disappear without a trace — it
    /// **degrades loudly**: [`artifact_fallback_notice`] appends a short
    /// plain-text notice and a `warn` is logged. Text delivery is never
    /// blocked by an attachment problem, and the notice never duplicates a
    /// `Download:` line the agent loop already added.
    async fn send_with_artifacts(
        &self,
        message: &SendMessage,
        artifacts: &[Artifact],
    ) -> anyhow::Result<()> {
        match message_with_artifact_notice(message, artifacts) {
            None => self.send(message).await,
            Some(degraded) => {
                tracing::warn!(
                    channel = self.name(),
                    artifact_count = artifacts.len(),
                    "channel has no native artifact upload; degrading attachments to text"
                );
                self.send(&degraded).await
            }
        }
    }

    /// Draft equivalent of [`send_with_artifacts`], with the same
    /// degrade-loudly default: the notice is appended to `text` before it is
    /// handed to [`finalize_draft`].
    async fn finalize_draft_with_artifacts(
        &self,
        recipient: &str,
        message_id: &str,
        text: &str,
        artifacts: &[Artifact],
    ) -> anyhow::Result<()> {
        let notice = artifact_fallback_notice(text, artifacts);
        if notice.is_empty() {
            return self.finalize_draft(recipient, message_id, text).await;
        }
        tracing::warn!(
            channel = self.name(),
            artifact_count = artifacts.len(),
            "channel has no native artifact upload on the draft path; degrading attachments to text"
        );
        self.finalize_draft(recipient, message_id, &format!("{text}{notice}"))
            .await
    }

    /// Signal that the bot is processing a response (e.g. "typing" indicator).
    /// Implementations should repeat the indicator as needed for their platform.
    async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Stop any active typing indicator.
    async fn stop_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Whether this channel supports progressive message updates via draft edits.
    fn supports_draft_updates(&self) -> bool {
        false
    }

    /// Send an initial draft message. Returns a platform-specific message ID for later edits.
    async fn send_draft(&self, _message: &SendMessage) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    /// Update a previously sent draft message with new accumulated content.
    async fn update_draft(
        &self,
        _recipient: &str,
        _message_id: &str,
        _text: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Finalize a draft with the complete response (e.g. apply Markdown formatting).
    async fn finalize_draft(
        &self,
        _recipient: &str,
        _message_id: &str,
        _text: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Cancel and remove a previously sent draft message if the channel supports it.
    async fn cancel_draft(&self, _recipient: &str, _message_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Add a reaction (emoji) to a message.
    ///
    /// `channel_id` is the platform channel/conversation identifier (e.g. Discord channel ID).
    /// `message_id` is the platform-scoped message identifier (e.g. `discord_<snowflake>`).
    /// `emoji` is the Unicode emoji to react with (e.g. "👀", "✅").
    async fn add_reaction(
        &self,
        _channel_id: &str,
        _message_id: &str,
        _emoji: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Remove a reaction (emoji) from a message previously added by this bot.
    async fn remove_reaction(
        &self,
        _channel_id: &str,
        _message_id: &str,
        _emoji: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Pin a message in the channel.
    async fn pin_message(&self, _channel_id: &str, _message_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Unpin a previously pinned message.
    async fn unpin_message(&self, _channel_id: &str, _message_id: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyChannel;

    #[async_trait]
    impl Channel for DummyChannel {
        fn name(&self) -> &str {
            "dummy"
        }

        async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn listen(
            &self,
            tx: tokio::sync::mpsc::Sender<ChannelMessage>,
        ) -> anyhow::Result<()> {
            tx.send(ChannelMessage {
                id: "1".into(),
                sender: "tester".into(),
                reply_target: "tester".into(),
                content: "hello".into(),
                channel: "dummy".into(),
                timestamp: 123,
                thread_ts: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
        }
    }

    #[test]
    fn channel_message_clone_preserves_fields() {
        let message = ChannelMessage {
            id: "42".into(),
            sender: "alice".into(),
            reply_target: "alice".into(),
            content: "ping".into(),
            channel: "dummy".into(),
            timestamp: 999,
            thread_ts: None,
        };

        let cloned = message.clone();
        assert_eq!(cloned.id, "42");
        assert_eq!(cloned.sender, "alice");
        assert_eq!(cloned.reply_target, "alice");
        assert_eq!(cloned.content, "ping");
        assert_eq!(cloned.channel, "dummy");
        assert_eq!(cloned.timestamp, 999);
    }

    /// Build an `Artifact` for the delivery-contract tests. `path` is
    /// workspace-relative and appears verbatim in the signed URL, matching
    /// what `from_workspace_path` produces — the strip logic matches on
    /// `/download/{path}`, so an absolute fixture path would never match.
    fn artifact(name: &str, download_url: Option<&str>) -> Artifact {
        Artifact {
            path: name.to_string(),
            name: name.to_string(),
            mime: crate::tools::mime_for_extension(name),
            size_bytes: 1,
            download_url: download_url.map(str::to_string),
            kind: crate::tools::ArtifactKind::infer(None, name.to_string().as_ref()),
        }
    }

    /// The two halves of the "never double up" contract compose. This is the
    /// exact scenario `TelegramChannel`/`DiscordChannel::send_with_artifacts`
    /// run: strip the `Download:` line for what we are about to attach, then
    /// ask for a notice about what we could not.
    #[test]
    fn strip_then_notice_never_leaves_a_duplicate_download_line() {
        let url = "https://gw.example/download/report.docx?expires=1&sig=ab";
        let art = artifact("report.docx", Some(url));
        let content = format!("here you go\nDownload: {url}");

        // Delivered natively: the line is stripped and nothing is added back.
        let stripped = strip_download_lines_for_paths(&content, std::slice::from_ref(&art.path));
        assert_eq!(stripped, "here you go");
        assert!(artifact_fallback_notice(&stripped, &[]).is_empty());
        assert_eq!(stripped.matches("Download: ").count(), 0);

        // Not delivered: the line is kept and the notice stays silent about it.
        let kept = strip_download_lines_for_paths(&content, &[]);
        assert_eq!(kept, content);
        assert!(
            artifact_fallback_notice(&kept, std::slice::from_ref(&art)).is_empty(),
            "url is still in the text: the notice must not repeat it"
        );
        assert_eq!(kept.matches("Download: ").count(), 1);
    }

    #[tokio::test]
    async fn default_trait_methods_return_success() {
        let channel = DummyChannel;

        assert!(channel.health_check().await);
        assert!(channel.start_typing("bob").await.is_ok());
        assert!(channel.stop_typing("bob").await.is_ok());
        assert!(channel
            .send(&SendMessage::new("hello", "bob"))
            .await
            .is_ok());
    }

    /// The default `send_with_artifacts` / `finalize_draft_with_artifacts`
    /// impls must forward to `send` / `finalize_draft` exactly once for every
    /// channel that has not overridden them. Since PR 3c the forwarded text
    /// may carry an appended fallback notice (see
    /// `default_send_with_artifacts_degrades_without_duplicating_download_lines`),
    /// but the call count — and therefore the delivery semantics — is
    /// unchanged.
    #[tokio::test]
    async fn default_artifact_methods_fall_back_to_send() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct SendCountingChannel {
            sends: Arc<AtomicUsize>,
            finalize_drafts: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl Channel for SendCountingChannel {
            fn name(&self) -> &str {
                "count"
            }
            async fn send(&self, _m: &SendMessage) -> anyhow::Result<()> {
                self.sends.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            async fn listen(
                &self,
                _tx: tokio::sync::mpsc::Sender<ChannelMessage>,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            async fn finalize_draft(&self, _r: &str, _id: &str, _t: &str) -> anyhow::Result<()> {
                self.finalize_drafts.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let sends = Arc::new(AtomicUsize::new(0));
        let finalizes = Arc::new(AtomicUsize::new(0));
        let ch = SendCountingChannel {
            sends: sends.clone(),
            finalize_drafts: finalizes.clone(),
        };

        // Any artifact payload — default impl must ignore it and call send.
        let art = Artifact {
            path: "x.docx".into(),
            name: "x.docx".into(),
            mime: None,
            size_bytes: 1,
            download_url: None,
            kind: crate::tools::ArtifactKind::infer(None, "x.docx"),
        };

        ch.send_with_artifacts(&SendMessage::new("hi", "bob"), std::slice::from_ref(&art))
            .await
            .unwrap();
        assert_eq!(sends.load(Ordering::SeqCst), 1);

        ch.finalize_draft_with_artifacts("bob", "m1", "text", &[art])
            .await
            .unwrap();
        assert_eq!(finalizes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn default_reaction_methods_return_success() {
        let channel = DummyChannel;

        assert!(channel
            .add_reaction("chan_1", "msg_1", "\u{1F440}")
            .await
            .is_ok());
        assert!(channel
            .remove_reaction("chan_1", "msg_1", "\u{1F440}")
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn default_draft_methods_return_success() {
        let channel = DummyChannel;

        assert!(!channel.supports_draft_updates());
        assert!(channel
            .send_draft(&SendMessage::new("draft", "bob"))
            .await
            .unwrap()
            .is_none());
        assert!(channel.update_draft("bob", "msg_1", "text").await.is_ok());
        assert!(channel
            .finalize_draft("bob", "msg_1", "final text")
            .await
            .is_ok());
        assert!(channel.cancel_draft("bob", "msg_1").await.is_ok());
    }

    #[tokio::test]
    async fn listen_sends_message_to_channel() {
        let channel = DummyChannel;
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

        channel.listen(tx).await.unwrap();

        let received = rx.recv().await.expect("message should be sent");
        assert_eq!(received.sender, "tester");
        assert_eq!(received.content, "hello");
        assert_eq!(received.channel, "dummy");
    }
}
