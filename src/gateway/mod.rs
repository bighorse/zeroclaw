//! Axum-based HTTP gateway with proper HTTP/1.1 compliance, body limits, and timeouts.
//!
//! This module replaces the raw TCP implementation with axum for:
//! - Proper HTTP/1.1 parsing and compliance
//! - Content-Length validation (handled by hyper)
//! - Request body size limits (64KB max)
//! - Request timeouts (30s) to prevent slow-loris attacks
//! - Header sanitization (handled by axum/hyper)

pub mod api;
pub mod signed_url;
pub mod sse;
pub mod static_files;
pub mod ws;

use crate::channels::{
    Channel, LinqChannel, NextcloudTalkChannel, SendMessage, WatiChannel, WhatsAppChannel,
};
use crate::config::Config;
use crate::cost::CostTracker;
use crate::memory::{self, Memory, MemoryCategory};
use crate::providers::{self, ChatMessage, Provider};
use crate::runtime;
use crate::security::pairing::{constant_time_eq, is_public_bind, PairingGuard};
use crate::security::SecurityPolicy;
use crate::tools;
use crate::tools::traits::ToolSpec;
use crate::util::truncate_with_ellipsis;
use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{ConnectInfo, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json},
    routing::{delete, get, post, put},
    Router,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use uuid::Uuid;

/// Maximum request body size (64KB) — prevents memory exhaustion
pub const MAX_BODY_SIZE: usize = 65_536;
/// Request timeout. 30s is the legacy slow-loris guard, but it kills any
/// LLM-driven path (/webhook, /api/chat, channel webhooks) in the middle
/// of an agent loop — multi-turn tool calls easily run 30-120s. Slow-loris
/// is normally handled at the reverse proxy (nginx client_body_timeout
/// and read_timeout); 300s here lets agent loops complete while still
/// bounding pathological hangs.
pub const REQUEST_TIMEOUT_SECS: u64 = 300;
/// Sliding window used by gateway rate limiting.
pub const RATE_LIMIT_WINDOW_SECS: u64 = 60;
/// Fallback max distinct client keys tracked in gateway rate limiter.
pub const RATE_LIMIT_MAX_KEYS_DEFAULT: usize = 10_000;
/// Fallback max distinct idempotency keys retained in gateway memory.
pub const IDEMPOTENCY_MAX_KEYS_DEFAULT: usize = 10_000;

fn webhook_memory_key() -> String {
    format!("webhook_msg_{}", Uuid::new_v4())
}

fn whatsapp_memory_key(msg: &crate::channels::traits::ChannelMessage) -> String {
    format!("whatsapp_{}_{}", msg.sender, msg.id)
}

fn linq_memory_key(msg: &crate::channels::traits::ChannelMessage) -> String {
    format!("linq_{}_{}", msg.sender, msg.id)
}

fn wati_memory_key(msg: &crate::channels::traits::ChannelMessage) -> String {
    format!("wati_{}_{}", msg.sender, msg.id)
}

fn nextcloud_talk_memory_key(msg: &crate::channels::traits::ChannelMessage) -> String {
    format!("nextcloud_talk_{}_{}", msg.sender, msg.id)
}

fn hash_webhook_secret(value: &str) -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(value.as_bytes());
    hex::encode(digest)
}

/// How often the rate limiter sweeps stale IP entries from its map.
const RATE_LIMITER_SWEEP_INTERVAL_SECS: u64 = 300; // 5 minutes

#[derive(Debug)]
struct SlidingWindowRateLimiter {
    limit_per_window: u32,
    window: Duration,
    max_keys: usize,
    requests: Mutex<(HashMap<String, Vec<Instant>>, Instant)>,
}

impl SlidingWindowRateLimiter {
    fn new(limit_per_window: u32, window: Duration, max_keys: usize) -> Self {
        Self {
            limit_per_window,
            window,
            max_keys: max_keys.max(1),
            requests: Mutex::new((HashMap::new(), Instant::now())),
        }
    }

    fn prune_stale(requests: &mut HashMap<String, Vec<Instant>>, cutoff: Instant) {
        requests.retain(|_, timestamps| {
            timestamps.retain(|t| *t > cutoff);
            !timestamps.is_empty()
        });
    }

    fn allow(&self, key: &str) -> bool {
        if self.limit_per_window == 0 {
            return true;
        }

        let now = Instant::now();
        let cutoff = now.checked_sub(self.window).unwrap_or_else(Instant::now);

        let mut guard = self.requests.lock();
        let (requests, last_sweep) = &mut *guard;

        // Periodic sweep: remove keys with no recent requests
        if last_sweep.elapsed() >= Duration::from_secs(RATE_LIMITER_SWEEP_INTERVAL_SECS) {
            Self::prune_stale(requests, cutoff);
            *last_sweep = now;
        }

        if !requests.contains_key(key) && requests.len() >= self.max_keys {
            // Opportunistic stale cleanup before eviction under cardinality pressure.
            Self::prune_stale(requests, cutoff);
            *last_sweep = now;

            if requests.len() >= self.max_keys {
                let evict_key = requests
                    .iter()
                    .min_by_key(|(_, timestamps)| timestamps.last().copied().unwrap_or(cutoff))
                    .map(|(k, _)| k.clone());
                if let Some(evict_key) = evict_key {
                    requests.remove(&evict_key);
                }
            }
        }

        let entry = requests.entry(key.to_owned()).or_default();
        entry.retain(|instant| *instant > cutoff);

        if entry.len() >= self.limit_per_window as usize {
            return false;
        }

        entry.push(now);
        true
    }
}

#[derive(Debug)]
pub struct GatewayRateLimiter {
    pair: SlidingWindowRateLimiter,
    webhook: SlidingWindowRateLimiter,
}

impl GatewayRateLimiter {
    fn new(pair_per_minute: u32, webhook_per_minute: u32, max_keys: usize) -> Self {
        let window = Duration::from_secs(RATE_LIMIT_WINDOW_SECS);
        Self {
            pair: SlidingWindowRateLimiter::new(pair_per_minute, window, max_keys),
            webhook: SlidingWindowRateLimiter::new(webhook_per_minute, window, max_keys),
        }
    }

    fn allow_pair(&self, key: &str) -> bool {
        self.pair.allow(key)
    }

    fn allow_webhook(&self, key: &str) -> bool {
        self.webhook.allow(key)
    }
}

#[derive(Debug)]
pub struct IdempotencyStore {
    ttl: Duration,
    max_keys: usize,
    keys: Mutex<HashMap<String, Instant>>,
}

impl IdempotencyStore {
    fn new(ttl: Duration, max_keys: usize) -> Self {
        Self {
            ttl,
            max_keys: max_keys.max(1),
            keys: Mutex::new(HashMap::new()),
        }
    }

    /// Returns true if this key is new and is now recorded.
    fn record_if_new(&self, key: &str) -> bool {
        let now = Instant::now();
        let mut keys = self.keys.lock();

        keys.retain(|_, seen_at| now.duration_since(*seen_at) < self.ttl);

        if keys.contains_key(key) {
            return false;
        }

        if keys.len() >= self.max_keys {
            let evict_key = keys
                .iter()
                .min_by_key(|(_, seen_at)| *seen_at)
                .map(|(k, _)| k.clone());
            if let Some(evict_key) = evict_key {
                keys.remove(&evict_key);
            }
        }

        keys.insert(key.to_owned(), now);
        true
    }
}

fn parse_client_ip(value: &str) -> Option<IpAddr> {
    let value = value.trim().trim_matches('"').trim();
    if value.is_empty() {
        return None;
    }

    if let Ok(ip) = value.parse::<IpAddr>() {
        return Some(ip);
    }

    if let Ok(addr) = value.parse::<SocketAddr>() {
        return Some(addr.ip());
    }

    let value = value.trim_matches(['[', ']']);
    value.parse::<IpAddr>().ok()
}

fn forwarded_client_ip(headers: &HeaderMap) -> Option<IpAddr> {
    if let Some(xff) = headers.get("X-Forwarded-For").and_then(|v| v.to_str().ok()) {
        for candidate in xff.split(',') {
            if let Some(ip) = parse_client_ip(candidate) {
                return Some(ip);
            }
        }
    }

    headers
        .get("X-Real-IP")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_client_ip)
}

fn client_key_from_request(
    peer_addr: Option<SocketAddr>,
    headers: &HeaderMap,
    trust_forwarded_headers: bool,
) -> String {
    if trust_forwarded_headers {
        if let Some(ip) = forwarded_client_ip(headers) {
            return ip.to_string();
        }
    }

    peer_addr
        .map(|addr| addr.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn normalize_max_keys(configured: usize, fallback: usize) -> usize {
    if configured == 0 {
        fallback.max(1)
    } else {
        configured
    }
}

/// Shared state for all axum handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Mutex<Config>>,
    pub provider: Arc<dyn Provider>,
    pub model: String,
    pub temperature: f64,
    pub mem: Arc<dyn Memory>,
    pub auto_save: bool,
    /// SHA-256 hash of `X-Webhook-Secret` (hex-encoded), never plaintext.
    pub webhook_secret_hash: Option<Arc<str>>,
    pub pairing: Arc<PairingGuard>,
    pub trust_forwarded_headers: bool,
    pub rate_limiter: Arc<GatewayRateLimiter>,
    pub idempotency_store: Arc<IdempotencyStore>,
    pub whatsapp: Option<Arc<WhatsAppChannel>>,
    /// `WhatsApp` app secret for webhook signature verification (`X-Hub-Signature-256`)
    pub whatsapp_app_secret: Option<Arc<str>>,
    pub linq: Option<Arc<LinqChannel>>,
    /// Linq webhook signing secret for signature verification
    pub linq_signing_secret: Option<Arc<str>>,
    pub nextcloud_talk: Option<Arc<NextcloudTalkChannel>>,
    /// Nextcloud Talk webhook secret for signature verification
    pub nextcloud_talk_webhook_secret: Option<Arc<str>>,
    pub wati: Option<Arc<WatiChannel>>,
    /// Observability backend for metrics scraping
    pub observer: Arc<dyn crate::observability::Observer>,
    /// Registered tool specs (for web dashboard tools page)
    pub tools_registry: Arc<Vec<ToolSpec>>,
    /// Cost tracker (optional, for web dashboard cost page)
    pub cost_tracker: Option<Arc<CostTracker>>,
    /// SSE broadcast channel for real-time events
    pub event_tx: tokio::sync::broadcast::Sender<serde_json::Value>,
    /// Shutdown signal sender for graceful shutdown
    pub shutdown_tx: tokio::sync::watch::Sender<bool>,
    /// Persistent conversation history for the `/api/chat` REST endpoint.
    /// One daemon serves at most one user (by design), so a single global
    /// history vector is sufficient — no session id needed.
    pub api_chat_history: Arc<Mutex<Vec<crate::providers::ChatMessage>>>,
    /// Recent `sop_result` events, replayed to SSE clients on (re)connect.
    ///
    /// A SOP's final report is broadcast exactly once, at the moment it finishes.
    /// Users routinely approve a step and then close the tab, lock the phone, or
    /// just refresh — with no listener attached at that instant the report is gone
    /// for good and the chat shows nothing at all. This keeps a short replay window.
    pub recent_sop_results: Arc<Mutex<std::collections::VecDeque<serde_json::Value>>>,
    /// Shared SOP engine. Used by `POST /sop/*` to dispatch events
    /// directly into the engine, and shared with the agent loop's tool
    /// list so runs are visible to in-agent `sop_status` / `sop_advance`
    /// without crossing process boundaries.
    pub sop_engine: Arc<std::sync::Mutex<crate::sop::SopEngine>>,
}

/// Run the HTTP gateway using axum with proper HTTP/1.1 compliance.
///
/// `external_sop_engine`: when `Some`, the gateway shares this engine
/// with the daemon's other components (notably channels' agent loop).
/// This is the daemon-singleton pattern that ensures runs started via
/// `POST /sop/*` are visible to in-agent `sop_status`/`sop_advance`
/// calls — and that `POST /sop/approve/{run_id}` can find runs created
/// by an in-agent `sop_execute`. Pass `None` only when running the
/// gateway in isolation (tests, standalone webhook server).
#[allow(clippy::too_many_lines)]
pub async fn run_gateway(
    host: &str,
    port: u16,
    config: Config,
    external_sop_engine: Option<Arc<std::sync::Mutex<crate::sop::SopEngine>>>,
) -> Result<()> {
    // ── Security: refuse public bind without tunnel or explicit opt-in ──
    if is_public_bind(host) && config.tunnel.provider == "none" && !config.gateway.allow_public_bind
    {
        anyhow::bail!(
            "🛑 Refusing to bind to {host} — gateway would be exposed to the internet.\n\
             Fix: use --host 127.0.0.1 (default), configure a tunnel, or set\n\
             [gateway] allow_public_bind = true in config.toml (NOT recommended)."
        );
    }
    let config_state = Arc::new(Mutex::new(config.clone()));

    // ── Hooks ──────────────────────────────────────────────────────
    let hooks: Option<std::sync::Arc<crate::hooks::HookRunner>> = if config.hooks.enabled {
        Some(std::sync::Arc::new(crate::hooks::HookRunner::new()))
    } else {
        None
    };

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let actual_port = listener.local_addr()?.port();
    let display_addr = format!("{host}:{actual_port}");

    let provider: Arc<dyn Provider> = Arc::from(providers::create_resilient_provider_with_options(
        config.default_provider.as_deref().unwrap_or("openrouter"),
        config.api_key.as_deref(),
        config.api_url.as_deref(),
        &config.reliability,
        &providers::ProviderRuntimeOptions {
            auth_profile_override: None,
            provider_api_url: config.api_url.clone(),
            zeroclaw_dir: config.config_path.parent().map(std::path::PathBuf::from),
            secrets_encrypt: config.secrets.encrypt,
            reasoning_enabled: config.runtime.reasoning_enabled,
            provider_timeout_secs: Some(config.provider_timeout_secs),
            // PR-H: derive per-fallback `[model_providers.X]` base_url overrides.
            fallback_provider_base_urls: config
                .reliability
                .fallback_providers
                .iter()
                .filter_map(|fb_name| {
                    config
                        .model_providers
                        .iter()
                        .find(|(name, _)| name.eq_ignore_ascii_case(fb_name))
                        .and_then(|(_, profile)| {
                            profile
                                .base_url
                                .as_ref()
                                .map(|url| (fb_name.clone(), url.clone()))
                        })
                })
                .collect(),
        },
    )?);
    let model = config
        .default_model
        .clone()
        .unwrap_or_else(|| "anthropic/claude-sonnet-4".into());
    let temperature = config.default_temperature;
    let mem: Arc<dyn Memory> = Arc::from(memory::create_memory_with_storage(
        &config.memory,
        Some(&config.storage.provider.config),
        &config.workspace_dir,
        config.api_key.as_deref(),
    )?);
    let runtime: Arc<dyn runtime::RuntimeAdapter> =
        Arc::from(runtime::create_runtime(&config.runtime)?);
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));

    let (composio_key, composio_entity_id) = if config.composio.enabled {
        (
            config.composio.api_key.as_deref(),
            Some(config.composio.entity_id.as_str()),
        )
    } else {
        (None, None)
    };

    // Shared SOP engine. Prefer the daemon-supplied engine so all
    // components (gateway + channels' agent loop) see the same
    // `active_runs`. Fall back to a fresh local engine in standalone
    // mode (tests, isolated gateway).
    let sop_engine = external_sop_engine.unwrap_or_else(|| {
        let mut engine = crate::sop::SopEngine::new(config.sop.clone());
        engine.reload(&config.workspace_dir);
        Arc::new(std::sync::Mutex::new(engine))
    });

    let tools_registry_raw = tools::all_tools_with_runtime(
        Arc::new(config.clone()),
        &security,
        runtime,
        Arc::clone(&mem),
        composio_key,
        composio_entity_id,
        &config.browser,
        &config.http_request,
        &config.web_fetch,
        &config.workspace_dir,
        &config.agents,
        config.api_key.as_deref(),
        &config,
        Some(Arc::clone(&sop_engine)),
    );
    let tools_registry: Arc<Vec<ToolSpec>> =
        Arc::new(tools_registry_raw.iter().map(|t| t.spec()).collect());

    // Cost tracker (optional). Uses `shared_tracker` so gateway, channels,
    // heartbeat, and agent::run all observe the same in-memory budget
    // aggregate when they run in the same daemon process.
    let cost_tracker = crate::cost::shared_tracker(&config.cost, &config.workspace_dir);

    // SSE broadcast channel for real-time events
    let (event_tx, _event_rx) = tokio::sync::broadcast::channel::<serde_json::Value>(256);
    // Extract webhook secret for authentication
    let webhook_secret_hash: Option<Arc<str>> =
        config.channels_config.webhook.as_ref().and_then(|webhook| {
            webhook.secret.as_ref().and_then(|raw_secret| {
                let trimmed_secret = raw_secret.trim();
                (!trimmed_secret.is_empty())
                    .then(|| Arc::<str>::from(hash_webhook_secret(trimmed_secret)))
            })
        });

    // WhatsApp channel (if configured)
    let whatsapp_channel: Option<Arc<WhatsAppChannel>> = config
        .channels_config
        .whatsapp
        .as_ref()
        .filter(|wa| wa.is_cloud_config())
        .map(|wa| {
            Arc::new(WhatsAppChannel::new(
                wa.access_token.clone().unwrap_or_default(),
                wa.phone_number_id.clone().unwrap_or_default(),
                wa.verify_token.clone().unwrap_or_default(),
                wa.allowed_numbers.clone(),
            ))
        });

    // WhatsApp app secret for webhook signature verification
    // Priority: environment variable > config file
    let whatsapp_app_secret: Option<Arc<str>> = std::env::var("ZEROCLAW_WHATSAPP_APP_SECRET")
        .ok()
        .and_then(|secret| {
            let secret = secret.trim();
            (!secret.is_empty()).then(|| secret.to_owned())
        })
        .or_else(|| {
            config.channels_config.whatsapp.as_ref().and_then(|wa| {
                wa.app_secret
                    .as_deref()
                    .map(str::trim)
                    .filter(|secret| !secret.is_empty())
                    .map(ToOwned::to_owned)
            })
        })
        .map(Arc::from);

    // Linq channel (if configured)
    let linq_channel: Option<Arc<LinqChannel>> = config.channels_config.linq.as_ref().map(|lq| {
        Arc::new(LinqChannel::new(
            lq.api_token.clone(),
            lq.from_phone.clone(),
            lq.allowed_senders.clone(),
        ))
    });

    // Linq signing secret for webhook signature verification
    // Priority: environment variable > config file
    let linq_signing_secret: Option<Arc<str>> = std::env::var("ZEROCLAW_LINQ_SIGNING_SECRET")
        .ok()
        .and_then(|secret| {
            let secret = secret.trim();
            (!secret.is_empty()).then(|| secret.to_owned())
        })
        .or_else(|| {
            config.channels_config.linq.as_ref().and_then(|lq| {
                lq.signing_secret
                    .as_deref()
                    .map(str::trim)
                    .filter(|secret| !secret.is_empty())
                    .map(ToOwned::to_owned)
            })
        })
        .map(Arc::from);

    // WATI channel (if configured)
    let wati_channel: Option<Arc<WatiChannel>> =
        config.channels_config.wati.as_ref().map(|wati_cfg| {
            Arc::new(WatiChannel::new(
                wati_cfg.api_token.clone(),
                wati_cfg.api_url.clone(),
                wati_cfg.tenant_id.clone(),
                wati_cfg.allowed_numbers.clone(),
            ))
        });

    // Nextcloud Talk channel (if configured)
    let nextcloud_talk_channel: Option<Arc<NextcloudTalkChannel>> =
        config.channels_config.nextcloud_talk.as_ref().map(|nc| {
            Arc::new(NextcloudTalkChannel::new(
                nc.base_url.clone(),
                nc.app_token.clone(),
                nc.allowed_users.clone(),
            ))
        });

    // Nextcloud Talk webhook secret for signature verification
    // Priority: environment variable > config file
    let nextcloud_talk_webhook_secret: Option<Arc<str>> =
        std::env::var("ZEROCLAW_NEXTCLOUD_TALK_WEBHOOK_SECRET")
            .ok()
            .and_then(|secret| {
                let secret = secret.trim();
                (!secret.is_empty()).then(|| secret.to_owned())
            })
            .or_else(|| {
                config
                    .channels_config
                    .nextcloud_talk
                    .as_ref()
                    .and_then(|nc| {
                        nc.webhook_secret
                            .as_deref()
                            .map(str::trim)
                            .filter(|secret| !secret.is_empty())
                            .map(ToOwned::to_owned)
                    })
            })
            .map(Arc::from);

    // ── Pairing guard ──────────────────────────────────────
    let pairing = Arc::new(PairingGuard::new(
        config.gateway.require_pairing,
        &config.gateway.paired_tokens,
    ));
    let rate_limit_max_keys = normalize_max_keys(
        config.gateway.rate_limit_max_keys,
        RATE_LIMIT_MAX_KEYS_DEFAULT,
    );
    let rate_limiter = Arc::new(GatewayRateLimiter::new(
        config.gateway.pair_rate_limit_per_minute,
        config.gateway.webhook_rate_limit_per_minute,
        rate_limit_max_keys,
    ));
    let idempotency_max_keys = normalize_max_keys(
        config.gateway.idempotency_max_keys,
        IDEMPOTENCY_MAX_KEYS_DEFAULT,
    );
    let idempotency_store = Arc::new(IdempotencyStore::new(
        Duration::from_secs(config.gateway.idempotency_ttl_secs.max(1)),
        idempotency_max_keys,
    ));

    // ── Tunnel ────────────────────────────────────────────────
    let tunnel = crate::tunnel::create_tunnel(&config.tunnel)?;
    let mut tunnel_url: Option<String> = None;

    if let Some(ref tun) = tunnel {
        println!("🔗 Starting {} tunnel...", tun.name());
        match tun.start(host, actual_port).await {
            Ok(url) => {
                println!("🌐 Tunnel active: {url}");
                tunnel_url = Some(url);
            }
            Err(e) => {
                println!("⚠️  Tunnel failed to start: {e}");
                println!("   Falling back to local-only mode.");
            }
        }
    }

    println!("🦀 ZeroClaw Gateway listening on http://{display_addr}");
    if let Some(ref url) = tunnel_url {
        println!("  🌐 Public URL: {url}");
    }
    println!("  🌐 Web Dashboard: http://{display_addr}/");
    println!("  POST /pair      — pair a new client (X-Pairing-Code header)");
    println!("  POST /webhook   — {{\"message\": \"your prompt\"}}");
    if whatsapp_channel.is_some() {
        println!("  GET  /whatsapp  — Meta webhook verification");
        println!("  POST /whatsapp  — WhatsApp message webhook");
    }
    if linq_channel.is_some() {
        println!("  POST /linq      — Linq message webhook (iMessage/RCS/SMS)");
    }
    if wati_channel.is_some() {
        println!("  GET  /wati      — WATI webhook verification");
        println!("  POST /wati      — WATI message webhook");
    }
    if nextcloud_talk_channel.is_some() {
        println!("  POST /nextcloud-talk — Nextcloud Talk bot webhook");
    }
    println!("  GET  /api/*     — REST API (bearer token required)");
    println!("  GET  /ws/chat   — WebSocket agent chat");
    println!("  GET  /health    — health check");
    println!("  GET  /metrics   — Prometheus metrics");
    if let Some(code) = pairing.pairing_code() {
        println!();
        println!("  🔐 PAIRING REQUIRED — use this one-time code:");
        println!("     ┌──────────────┐");
        println!("     │  {code}  │");
        println!("     └──────────────┘");
        println!("     Send: POST /pair with header X-Pairing-Code: {code}");
    } else if pairing.require_pairing() {
        println!("  🔒 Pairing: ACTIVE (bearer token required)");
    } else {
        println!("  ⚠️  Pairing: DISABLED (all requests accepted)");
    }
    println!("  Press Ctrl+C to stop.\n");

    crate::health::mark_component_ok("gateway");

    // Fire gateway start hook
    if let Some(ref hooks) = hooks {
        hooks.fire_gateway_start(host, actual_port).await;
    }

    // Wrap observer with broadcast capability for SSE
    let broadcast_observer: Arc<dyn crate::observability::Observer> =
        Arc::new(sse::BroadcastObserver::new(
            crate::observability::create_observer(&config.observability),
            event_tx.clone(),
        ));

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);

    let state = AppState {
        config: config_state,
        provider,
        model,
        temperature,
        mem,
        auto_save: config.memory.auto_save,
        webhook_secret_hash,
        pairing,
        trust_forwarded_headers: config.gateway.trust_forwarded_headers,
        rate_limiter,
        idempotency_store,
        whatsapp: whatsapp_channel,
        whatsapp_app_secret,
        linq: linq_channel,
        linq_signing_secret,
        nextcloud_talk: nextcloud_talk_channel,
        nextcloud_talk_webhook_secret,
        wati: wati_channel,
        observer: broadcast_observer,
        tools_registry,
        cost_tracker,
        event_tx,
        shutdown_tx,
        api_chat_history: Arc::new(Mutex::new(Vec::new())),
        recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        sop_engine: Arc::clone(&sop_engine),
    };

    // Config PUT needs larger body limit (1MB)
    let config_put_router = Router::new()
        .route("/api/config", put(api::handle_api_config_put))
        .layer(RequestBodyLimitLayer::new(1_048_576));

    // Build router with middleware
    let app = Router::new()
        // ── Admin routes (for CLI management) ──
        .route("/admin/shutdown", post(handle_admin_shutdown))
        .route("/admin/paircode", get(handle_admin_paircode))
        .route("/admin/paircode/new", post(handle_admin_paircode_new))
        // ── Existing routes ──
        .route("/download/{*filepath}", get(handle_workspace_download))
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_metrics))
        .route("/pair", post(handle_pair))
        .route("/webhook", post(handle_webhook))
        .route("/sop/approve/{run_id}", post(handle_sop_approve))
        .route("/sop/reject/{run_id}", post(handle_sop_reject))
        .route("/sop/{*rest}", post(handle_sop_webhook))
        .route("/api/chat", post(handle_api_chat))
        .route("/whatsapp", get(handle_whatsapp_verify))
        .route("/whatsapp", post(handle_whatsapp_message))
        .route("/linq", post(handle_linq_webhook))
        .route("/wati", get(handle_wati_verify))
        .route("/wati", post(handle_wati_webhook))
        .route("/nextcloud-talk", post(handle_nextcloud_talk_webhook))
        // ── Web Dashboard API routes ──
        .route("/api/status", get(api::handle_api_status))
        .route("/api/sop/runs", get(api::handle_api_sop_runs))
        .route("/api/config", get(api::handle_api_config_get))
        .route("/api/tools", get(api::handle_api_tools))
        .route("/api/cron", get(api::handle_api_cron_list))
        .route("/api/cron", post(api::handle_api_cron_add))
        .route("/api/cron/{id}", delete(api::handle_api_cron_delete))
        .route("/api/integrations", get(api::handle_api_integrations))
        .route(
            "/api/integrations/settings",
            get(api::handle_api_integrations_settings),
        )
        .route(
            "/api/doctor",
            get(api::handle_api_doctor).post(api::handle_api_doctor),
        )
        .route("/api/memory", get(api::handle_api_memory_list))
        .route("/api/memory", post(api::handle_api_memory_store))
        .route("/api/memory/{key}", delete(api::handle_api_memory_delete))
        .route("/api/cost", get(api::handle_api_cost))
        .route("/api/cli-tools", get(api::handle_api_cli_tools))
        .route("/api/health", get(api::handle_api_health))
        // ── SSE event stream ──
        .route("/api/events", get(sse::handle_sse_events))
        // ── WebSocket agent chat ──
        .route("/ws/chat", get(ws::handle_ws_chat))
        // ── Static assets (web dashboard) ──
        .route("/_app/{*path}", get(static_files::handle_static))
        // ── Config PUT with larger body limit ──
        .merge(config_put_router)
        .with_state(state)
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(REQUEST_TIMEOUT_SECS),
        ))
        // ── SPA fallback: non-API GET requests serve index.html ──
        .fallback(get(static_files::handle_spa_fallback));

    // Run the server with graceful shutdown
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let _ = shutdown_rx.changed().await;
        tracing::info!("🦀 ZeroClaw Gateway shutting down...");
    })
    .await?;

    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// AXUM HANDLERS
// ══════════════════════════════════════════════════════════════════════════════

/// GET /health — always public (no secrets leaked)
async fn handle_health(State(state): State<AppState>) -> impl IntoResponse {
    let body = serde_json::json!({
        "status": "ok",
        "paired": state.pairing.is_paired(),
        "require_pairing": state.pairing.require_pairing(),
        "runtime": crate::health::snapshot_json(),
    });
    Json(body)
}

/// Query parameters for signed download URLs.
#[derive(serde::Deserialize)]
struct DownloadQuery {
    expires: Option<u64>,
    sig: Option<String>,
}

/// GET /download/*filepath — download a workspace file with signed-URL verification.
async fn handle_workspace_download(
    State(state): State<AppState>,
    axum::extract::Path(filepath): axum::extract::Path<String>,
    Query(query): Query<DownloadQuery>,
) -> impl IntoResponse {
    let (workspace_dir, download_secret) = {
        let guard = state.config.lock();
        (guard.workspace_dir.clone(), guard.resolve_download_secret())
    };

    // Decode percent-encoded path (e.g. `reports%2Fsummary.md` → `reports/summary.md`)
    let decoded = urlencoding::decode(&filepath).unwrap_or(std::borrow::Cow::Borrowed(&filepath));
    let file_path = decoded.as_ref();

    // Path traversal prevention: reject `..` and `\`
    if file_path.contains("..") || file_path.contains('\\') {
        return (StatusCode::BAD_REQUEST, "Invalid path").into_response();
    }

    // Verify signed URL if signature parameters are present.
    // If neither expires nor sig is provided, reject (require signing).
    eprintln!(
        "[DOWNLOAD] path={file_path:?} expires={:?} sig_present={} sig_len={}",
        query.expires,
        query.sig.is_some(),
        query.sig.as_deref().map(|s| s.len()).unwrap_or(0)
    );
    tracing::info!(
        path = %file_path,
        expires = ?query.expires,
        sig_present = query.sig.is_some(),
        sig_len = query.sig.as_deref().map(|s| s.len()).unwrap_or(0),
        "download: checking signed URL"
    );
    match (query.expires, query.sig.as_deref()) {
        (Some(expires), Some(sig)) => {
            let ok = signed_url::verify_download_url(file_path, &download_secret, expires, sig);
            tracing::info!(path = %file_path, expires, sig_ok = ok, "download: signature verified");
            if !ok {
                return (StatusCode::FORBIDDEN, "Invalid or expired signature").into_response();
            }
        }
        _ => {
            tracing::warn!(path = %file_path, expires = ?query.expires, sig = ?query.sig, "download: missing expires or sig — returning 403");
            return (StatusCode::FORBIDDEN, "链接缺少签名参数，无法直接下载。请回到对话里让龙虾重新发一次下载链接（它会生成带签名的完整链接）。").into_response();
        }
    }

    // Resolve and verify the path stays within workspace
    let full_path = workspace_dir.join(file_path);
    let canonical = match tokio::fs::canonicalize(&full_path).await {
        Ok(p) => p,
        Err(_) => return (StatusCode::NOT_FOUND, "File not found").into_response(),
    };
    let canonical_workspace = match tokio::fs::canonicalize(&workspace_dir).await {
        Ok(p) => p,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Workspace error").into_response(),
    };
    if !canonical.starts_with(&canonical_workspace) {
        return (StatusCode::FORBIDDEN, "Path escapes workspace").into_response();
    }

    match tokio::fs::read(&canonical).await {
        Ok(bytes) => {
            let display_name = file_path.rsplit('/').next().unwrap_or(file_path);
            let encoded_name = urlencoding::encode(display_name);
            let disposition = format!("attachment; filename*=UTF-8''{encoded_name}");
            let content_type = guess_content_type(file_path);
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, content_type.to_string()),
                    (header::CONTENT_DISPOSITION, disposition),
                ],
                bytes,
            )
                .into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "File not found").into_response(),
    }
}

/// Guess a reasonable Content-Type from the file extension.
fn guess_content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().map(str::to_lowercase).as_deref() {
        Some("md") => "text/markdown; charset=utf-8",
        Some("txt") => "text/plain; charset=utf-8",
        Some("json") => "application/json",
        Some("csv") => "text/csv; charset=utf-8",
        Some("html" | "htm") => "text/html; charset=utf-8",
        Some("pdf") => "application/pdf",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

/// Prometheus content type for text exposition format.
const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// GET /metrics — Prometheus text exposition format
async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let body = if let Some(prom) = state
        .observer
        .as_ref()
        .as_any()
        .downcast_ref::<crate::observability::PrometheusObserver>()
    {
        prom.encode()
    } else {
        String::from("# Prometheus backend not enabled. Set [observability] backend = \"prometheus\" in config.\n")
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, PROMETHEUS_CONTENT_TYPE)],
        body,
    )
}

/// POST /pair — exchange one-time code for bearer token
#[axum::debug_handler]
async fn handle_pair(
    State(state): State<AppState>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let rate_key =
        client_key_from_request(Some(peer_addr), &headers, state.trust_forwarded_headers);
    if !state.rate_limiter.allow_pair(&rate_key) {
        tracing::warn!("/pair rate limit exceeded");
        let err = serde_json::json!({
            "error": "Too many pairing requests. Please retry later.",
            "retry_after": RATE_LIMIT_WINDOW_SECS,
        });
        return (StatusCode::TOO_MANY_REQUESTS, Json(err));
    }

    let code = headers
        .get("X-Pairing-Code")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    match state.pairing.try_pair(code, &rate_key).await {
        Ok(Some(token)) => {
            tracing::info!("🔐 New client paired successfully");
            if let Err(err) = persist_pairing_tokens(state.config.clone(), &state.pairing).await {
                tracing::error!("🔐 Pairing succeeded but token persistence failed: {err:#}");
                let body = serde_json::json!({
                    "paired": true,
                    "persisted": false,
                    "token": token,
                    "message": "Paired for this process, but failed to persist token to config.toml. Check config path and write permissions.",
                });
                return (StatusCode::OK, Json(body));
            }

            let body = serde_json::json!({
                "paired": true,
                "persisted": true,
                "token": token,
                "message": "Save this token — use it as Authorization: Bearer <token>"
            });
            (StatusCode::OK, Json(body))
        }
        Ok(None) => {
            tracing::warn!("🔐 Pairing attempt with invalid code");
            let err = serde_json::json!({"error": "Invalid pairing code"});
            (StatusCode::FORBIDDEN, Json(err))
        }
        Err(lockout_secs) => {
            tracing::warn!(
                "🔐 Pairing locked out — too many failed attempts ({lockout_secs}s remaining)"
            );
            let err = serde_json::json!({
                "error": format!("Too many failed attempts. Try again in {lockout_secs}s."),
                "retry_after": lockout_secs
            });
            (StatusCode::TOO_MANY_REQUESTS, Json(err))
        }
    }
}

async fn persist_pairing_tokens(config: Arc<Mutex<Config>>, pairing: &PairingGuard) -> Result<()> {
    let paired_tokens = pairing.tokens();
    // This is needed because parking_lot's guard is not Send so we clone the inner
    // this should be removed once async mutexes are used everywhere
    let mut updated_cfg = { config.lock().clone() };
    updated_cfg.gateway.paired_tokens = paired_tokens;
    updated_cfg
        .save()
        .await
        .context("Failed to persist paired tokens to config.toml")?;

    // Keep shared runtime config in sync with persisted tokens.
    *config.lock() = updated_cfg;
    Ok(())
}

/// Simple chat for webhook endpoint (no tools, for backward compatibility and testing).
async fn run_gateway_chat_simple(state: &AppState, message: &str) -> anyhow::Result<String> {
    let user_messages = vec![ChatMessage::user(message)];

    // Keep webhook/gateway prompts aligned with channel behavior by injecting
    // workspace-aware system context before model invocation.
    let system_prompt = {
        let config_guard = state.config.lock();
        crate::channels::build_system_prompt(
            &config_guard.workspace_dir,
            &state.model,
            &[], // tools - empty for simple chat
            &[], // skills
            Some(&config_guard.identity),
            None, // bootstrap_max_chars - use default
        )
    };

    let mut messages = Vec::with_capacity(1 + user_messages.len());
    messages.push(ChatMessage::system(system_prompt));
    messages.extend(user_messages);

    let multimodal_config = state.config.lock().multimodal.clone();
    let prepared =
        crate::multimodal::prepare_messages_for_provider(&messages, &multimodal_config).await?;

    state
        .provider
        .chat_with_history(&prepared.messages, &state.model, state.temperature)
        .await
}

/// Full-featured chat with tools for channel handlers (WhatsApp, Linq, Nextcloud Talk).
async fn run_gateway_chat_with_tools(
    state: &AppState,
    message: &str,
    session_id: Option<&str>,
) -> anyhow::Result<String> {
    // process_message future is large (~20KB stack); box it to keep the
    // gateway task stack from blowing up when this is awaited inside a
    // long async chain (clippy::large_futures).
    let config = state.config.lock().clone();
    Box::pin(crate::agent::process_message(config, message, session_id)).await
}

/// Webhook request body
#[derive(serde::Deserialize)]
pub struct WebhookBody {
    pub message: String,
}

/// POST /webhook — main webhook endpoint
async fn handle_webhook(
    State(state): State<AppState>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Json<WebhookBody>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let rate_key =
        client_key_from_request(Some(peer_addr), &headers, state.trust_forwarded_headers);
    if !state.rate_limiter.allow_webhook(&rate_key) {
        tracing::warn!("/webhook rate limit exceeded");
        let err = serde_json::json!({
            "error": "Too many webhook requests. Please retry later.",
            "retry_after": RATE_LIMIT_WINDOW_SECS,
        });
        return (StatusCode::TOO_MANY_REQUESTS, Json(err));
    }

    // ── Bearer token auth (pairing) ──
    if state.pairing.require_pairing() {
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth.strip_prefix("Bearer ").unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            tracing::warn!("Webhook: rejected — not paired / invalid bearer token");
            let err = serde_json::json!({
                "error": "Unauthorized — pair first via POST /pair, then send Authorization: Bearer <token>"
            });
            return (StatusCode::UNAUTHORIZED, Json(err));
        }
    }

    // ── Webhook secret auth (optional, additional layer) ──
    if let Some(ref secret_hash) = state.webhook_secret_hash {
        let header_hash = headers
            .get("X-Webhook-Secret")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(hash_webhook_secret);
        match header_hash {
            Some(val) if constant_time_eq(&val, secret_hash.as_ref()) => {}
            _ => {
                tracing::warn!("Webhook: rejected request — invalid or missing X-Webhook-Secret");
                let err = serde_json::json!({"error": "Unauthorized — invalid or missing X-Webhook-Secret header"});
                return (StatusCode::UNAUTHORIZED, Json(err));
            }
        }
    }

    // ── Parse body ──
    let Json(webhook_body) = match body {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("Webhook JSON parse error: {e}");
            let err = serde_json::json!({
                "error": "Invalid JSON body. Expected: {\"message\": \"...\"}"
            });
            return (StatusCode::BAD_REQUEST, Json(err));
        }
    };

    // ── Idempotency (optional) ──
    if let Some(idempotency_key) = headers
        .get("X-Idempotency-Key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if !state.idempotency_store.record_if_new(idempotency_key) {
            tracing::info!("Webhook duplicate ignored (idempotency key: {idempotency_key})");
            let body = serde_json::json!({
                "status": "duplicate",
                "idempotent": true,
                "message": "Request already processed for this idempotency key"
            });
            return (StatusCode::OK, Json(body));
        }
    }

    let message = &webhook_body.message;

    if state.auto_save {
        let key = webhook_memory_key();
        let _ = state
            .mem
            .store(
                &key,
                message,
                MemoryCategory::Conversation,
                Some(crate::memory::GATEWAY_WEBHOOK_SESSION_ID),
            )
            .await;
    }

    let provider_label = state
        .config
        .lock()
        .default_provider
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let model_label = state.model.clone();
    let started_at = Instant::now();

    state
        .observer
        .record_event(&crate::observability::ObserverEvent::AgentStart {
            provider: provider_label.clone(),
            model: model_label.clone(),
        });
    state
        .observer
        .record_event(&crate::observability::ObserverEvent::LlmRequest {
            provider: provider_label.clone(),
            model: model_label.clone(),
            messages_count: 1,
        });

    match run_gateway_chat_simple(&state, message).await {
        Ok(response) => {
            let duration = started_at.elapsed();
            state
                .observer
                .record_event(&crate::observability::ObserverEvent::LlmResponse {
                    provider: provider_label.clone(),
                    model: model_label.clone(),
                    duration,
                    success: true,
                    error_message: None,
                    input_tokens: None,
                    output_tokens: None,
                });
            state.observer.record_metric(
                &crate::observability::traits::ObserverMetric::RequestLatency(duration),
            );
            state
                .observer
                .record_event(&crate::observability::ObserverEvent::AgentEnd {
                    provider: provider_label,
                    model: model_label,
                    duration,
                    tokens_used: None,
                    cost_usd: None,
                });

            let body = serde_json::json!({"response": response, "model": state.model});
            (StatusCode::OK, Json(body))
        }
        Err(e) => {
            let duration = started_at.elapsed();
            let sanitized = providers::sanitize_api_error(&e.to_string());

            state
                .observer
                .record_event(&crate::observability::ObserverEvent::LlmResponse {
                    provider: provider_label.clone(),
                    model: model_label.clone(),
                    duration,
                    success: false,
                    error_message: Some(sanitized.clone()),
                    input_tokens: None,
                    output_tokens: None,
                });
            state.observer.record_metric(
                &crate::observability::traits::ObserverMetric::RequestLatency(duration),
            );
            state
                .observer
                .record_event(&crate::observability::ObserverEvent::Error {
                    component: "gateway".to_string(),
                    message: sanitized.clone(),
                });
            state
                .observer
                .record_event(&crate::observability::ObserverEvent::AgentEnd {
                    provider: provider_label,
                    model: model_label,
                    duration,
                    tokens_used: None,
                    cost_usd: None,
                });

            tracing::error!("Webhook provider error: {}", sanitized);
            let err = serde_json::json!({"error": "LLM request failed"});
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err))
        }
    }
}

/// POST /sop/approve/{run_id} — Channel-only SOP step approval.
///
/// Approves a run currently in `WaitingApproval` status and returns the
/// next action. Reachable only by callers that hold a paired bearer
/// token — typically a channel adapter (LarkChannel command parser
/// for `@bot 批准 run-xxx`) or an external operator with a curl/CLI.
///
/// Note: this endpoint exists because `SopApproveTool` is intentionally
/// NOT exposed to the LLM — the dual-sign quality gate would be
/// trivially defeated if the LLM could self-approve.
async fn handle_sop_approve(
    State(state): State<AppState>,
    axum::extract::Path(run_id): axum::extract::Path<String>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let rate_key =
        client_key_from_request(Some(peer_addr), &headers, state.trust_forwarded_headers);
    if !state.rate_limiter.allow_webhook(&rate_key) {
        let err = serde_json::json!({"error":"rate limit exceeded"});
        return (StatusCode::TOO_MANY_REQUESTS, Json(err));
    }

    if state.pairing.require_pairing() {
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth.strip_prefix("Bearer ").unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            let err = serde_json::json!({
                "error":"Unauthorized — pair first via POST /pair, then send Authorization: Bearer <token>"
            });
            return (StatusCode::UNAUTHORIZED, Json(err));
        }
    }

    let result = {
        let mut engine = match state.sop_engine.lock() {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("SOP approve: engine lock poisoned: {e}");
                let err = serde_json::json!({"error":"engine lock poisoned"});
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(err));
            }
        };
        engine.approve_step(&run_id)
    };

    match result {
        Ok(action) => {
            tracing::info!(run_id = %run_id, "SOP approve: run advanced");
            // 批准即续跑：唤醒 LLM 执行后续步骤（与 Lark 通道的 wake_msg 同构）。
            // 不阻塞本响应；进展经 /api/sop/runs 轮询可见，结果并入 /api/chat 会话历史。
            {
                let config = state.config.lock().clone();
                let history = Arc::clone(&state.api_chat_history);
                let engine = Arc::clone(&state.sop_engine);
                let event_tx = state.event_tx.clone();
                let recent = Arc::clone(&state.recent_sop_results);
                let rid = run_id.clone();
                tokio::spawn(async move {
                    let wake = format!(
                        "[系统] SOP run {rid} 的等待审批步骤【已经获得用户批准，审批已完成，不要再向用户请求任何批准或确认】。请立即用 sop_advance 推进并执行该流程的后续步骤，直到全部完成或到达下一个真正需要审批的新步骤。最后用面向用户的口吻简要汇报执行结果（不要提系统消息或内部指令）。"
                    );
                    let prior = { history.lock().clone() };
                    match crate::agent::process_message_with_history(
                        config,
                        &wake,
                        Some(prior),
                        None,
                        Some(engine),
                        Some(crate::memory::GATEWAY_API_CHAT_SESSION_ID),
                    )
                    .await
                    {
                        Ok((resp, new_hist)) => {
                            *history.lock() = new_hist;
                            // 续跑结果实时推给前台（对话里直接显示汇报），同时留一份供补发：
                            // 用户批准后往往就切走了，这一刻没有 SSE 监听者的话结果会凭空消失。
                            let ts = chrono::Utc::now().timestamp();
                            let ev = serde_json::json!({
                                "type": "sop_result",
                                "id": format!("{rid}:{ts}"),
                                "run_id": rid,
                                "response": resp,
                                "timestamp": ts,
                            });
                            push_recent_sop_result(&recent, ev.clone());
                            let _ = event_tx.send(ev);
                            tracing::info!(run_id = %rid, "SOP auto-resume finished");
                        }
                        Err(e) => {
                            tracing::warn!(run_id = %rid, "SOP auto-resume failed: {e:#}");
                        }
                    }
                });
            }
            let body = serde_json::json!({
                "status": "approved",
                "run_id": run_id,
                "next_action": format!("{action:?}"),
            });
            (StatusCode::OK, Json(body))
        }
        Err(e) => {
            tracing::warn!(run_id = %run_id, error = %e, "SOP approve failed");
            let msg = e.to_string();
            let status = if msg.contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::CONFLICT
            };
            (
                status,
                Json(serde_json::json!({"error": msg, "run_id": run_id})),
            )
        }
    }
}

/// Store one `sop_result` in the replay buffer: newest 20, nothing older than 12h.
///
/// Two caps on purpose — the count keeps memory bounded, the age keeps a user who
/// returns the next day from being flooded with stale reports.
pub(crate) fn push_recent_sop_result(
    buf: &Arc<Mutex<std::collections::VecDeque<serde_json::Value>>>,
    ev: serde_json::Value,
) {
    const MAX_ITEMS: usize = 20;
    const MAX_AGE_SECS: i64 = 12 * 3600;
    let cutoff = chrono::Utc::now().timestamp() - MAX_AGE_SECS;
    let mut q = buf.lock();
    q.push_back(ev);
    while q.len() > MAX_ITEMS {
        q.pop_front();
    }
    q.retain(|e| e.get("timestamp").and_then(|t| t.as_i64()).unwrap_or(0) >= cutoff);
}

/// POST /sop/reject/{run_id} — Channel-only SOP step rejection.
///
/// Counterpart to `/sop/approve/{run_id}`: terminates a run that is waiting
/// on a human decision, marking it `Cancelled` so it leaves `active_runs`.
/// Optional JSON body `{"reason": "..."}` is recorded on the finished run.
///
/// Like approve, this is intentionally NOT exposed to the LLM — the
/// dual-sign quality gate would be trivially defeated by self-service.
async fn handle_sop_reject(
    State(state): State<AppState>,
    axum::extract::Path(run_id): axum::extract::Path<String>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Option<Json<serde_json::Value>>,
) -> impl IntoResponse {
    let rate_key =
        client_key_from_request(Some(peer_addr), &headers, state.trust_forwarded_headers);
    if !state.rate_limiter.allow_webhook(&rate_key) {
        let err = serde_json::json!({"error":"rate limit exceeded"});
        return (StatusCode::TOO_MANY_REQUESTS, Json(err));
    }

    if state.pairing.require_pairing() {
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth.strip_prefix("Bearer ").unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            let err = serde_json::json!({"error":"Unauthorized"});
            return (StatusCode::UNAUTHORIZED, Json(err));
        }
    }

    let reason = body
        .and_then(|Json(v)| {
            v.get("reason")
                .and_then(|r| r.as_str())
                .map(|s| s.trim().chars().take(500).collect::<String>())
        })
        .filter(|s| !s.is_empty());

    let result = {
        let mut engine = match state.sop_engine.lock() {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("SOP reject: engine lock poisoned: {e}");
                let err = serde_json::json!({"error":"engine lock poisoned"});
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(err));
            }
        };
        engine.reject_step(&run_id, reason.clone())
    };

    match result {
        Ok(()) => {
            tracing::info!(run_id = %run_id, "SOP reject: run cancelled");
            let body = serde_json::json!({
                "status": "rejected",
                "run_id": run_id,
                "reason": reason.unwrap_or_else(|| "rejected by user".to_string()),
            });
            (StatusCode::OK, Json(body))
        }
        Err(e) => {
            tracing::warn!(run_id = %run_id, error = %e, "SOP reject failed");
            let msg = e.to_string();
            let status = if msg.contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::CONFLICT
            };
            (
                status,
                Json(serde_json::json!({"error": msg, "run_id": run_id})),
            )
        }
    }
}

/// POST /sop/{*rest} — SOP-only event endpoint.
///
/// Routes incoming events into the shared `SopEngine`. No LLM fallback —
/// returns 404 if no SOP trigger matches the request path.
///
/// Auth/rate-limit/idempotency contract mirrors `/webhook`:
/// - Bearer token via `Authorization: Bearer <token>` (when pairing required)
/// - Optional `X-Webhook-Secret` header
/// - Optional `X-Idempotency-Key` for client-side dedup
///
/// The path under `/sop/` is forwarded to the engine as the trigger
/// path. Body (if present, JSON) is forwarded as the event payload.
async fn handle_sop_webhook(
    State(state): State<AppState>,
    axum::extract::Path(rest): axum::extract::Path<String>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Option<Json<serde_json::Value>>,
) -> impl IntoResponse {
    let trigger_path = format!("/sop/{rest}");

    // ── Rate limit ──
    let rate_key =
        client_key_from_request(Some(peer_addr), &headers, state.trust_forwarded_headers);
    if !state.rate_limiter.allow_webhook(&rate_key) {
        tracing::warn!("/sop/* rate limit exceeded");
        let err = serde_json::json!({
            "error": "Too many SOP webhook requests. Please retry later.",
            "retry_after": RATE_LIMIT_WINDOW_SECS,
        });
        return (StatusCode::TOO_MANY_REQUESTS, Json(err));
    }

    // ── Bearer token auth (pairing) ──
    if state.pairing.require_pairing() {
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth.strip_prefix("Bearer ").unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            tracing::warn!("SOP webhook: rejected — not paired / invalid bearer token");
            let err = serde_json::json!({
                "error": "Unauthorized — pair first via POST /pair, then send Authorization: Bearer <token>"
            });
            return (StatusCode::UNAUTHORIZED, Json(err));
        }
    }

    // ── Optional X-Webhook-Secret ──
    if let Some(ref secret_hash) = state.webhook_secret_hash {
        let header_hash = headers
            .get("X-Webhook-Secret")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(hash_webhook_secret);
        match header_hash {
            Some(val) if constant_time_eq(&val, secret_hash.as_ref()) => {}
            _ => {
                tracing::warn!("SOP webhook: rejected — invalid or missing X-Webhook-Secret");
                let err = serde_json::json!({
                    "error": "Unauthorized — invalid or missing X-Webhook-Secret header"
                });
                return (StatusCode::UNAUTHORIZED, Json(err));
            }
        }
    }

    // ── Idempotency (optional) — namespaced by path so /sop/* keys
    //   do not collide with /webhook keys. ──
    if let Some(idempotency_key) = headers
        .get("X-Idempotency-Key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let namespaced = format!("sop:{idempotency_key}");
        if !state.idempotency_store.record_if_new(&namespaced) {
            tracing::info!("SOP webhook duplicate ignored (idempotency key: {idempotency_key})");
            let body = serde_json::json!({
                "status": "duplicate",
                "idempotent": true,
                "message": "Request already processed for this idempotency key",
                "path": trigger_path,
            });
            return (StatusCode::OK, Json(body));
        }
    }

    // ── Build SopEvent ──
    let payload_str = body.and_then(|Json(v)| serde_json::to_string(&v).ok());
    let event = crate::sop::SopEvent {
        source: crate::sop::SopTriggerSource::Webhook,
        topic: Some(trigger_path.clone()),
        payload: payload_str,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // ── Dispatch ──
    let audit = crate::sop::SopAuditLogger::new(Arc::clone(&state.mem));
    let results = crate::sop::dispatch::dispatch_sop_event(&state.sop_engine, &audit, event).await;

    // ── Map results → response ──
    use crate::sop::dispatch::DispatchResult;
    let started: Vec<&str> = results
        .iter()
        .filter_map(|r| match r {
            DispatchResult::Started { sop_name, .. } => Some(sop_name.as_str()),
            _ => None,
        })
        .collect();

    if started.is_empty() && results.iter().all(|r| matches!(r, DispatchResult::NoMatch)) {
        tracing::info!("SOP webhook: no SOP matched path {trigger_path}");
        let err = serde_json::json!({
            "error": "No SOP matched",
            "path": trigger_path,
        });
        return (StatusCode::NOT_FOUND, Json(err));
    }

    let skipped: Vec<serde_json::Value> = results
        .iter()
        .filter_map(|r| match r {
            DispatchResult::Skipped { sop_name, reason } => Some(serde_json::json!({
                "sop": sop_name,
                "reason": reason,
            })),
            _ => None,
        })
        .collect();

    let body = serde_json::json!({
        "status": "accepted",
        "matched_sops": started,
        "skipped": skipped,
        "source": "sop_webhook",
        "path": trigger_path,
    });
    (StatusCode::OK, Json(body))
}

/// Hard cap for messages retained in the `/api/chat` history between turns.
/// The agent loop appends an assistant tool-call message and a tool-result
/// message for every iteration, so a single agentic turn can add dozens of
/// entries; without a cap the vector grows unboundedly for the daemon's
/// lifetime. (The channel runtime has the same guard via MAX_CHANNEL_HISTORY.)
const MAX_API_CHAT_HISTORY_MESSAGES: usize = 60;

/// Keep this many most-recent plain user/assistant messages when recovering
/// from a context-window overflow. Mirrors the channel runtime's
/// CHANNEL_HISTORY_COMPACT_KEEP_MESSAGES.
const API_CHAT_HISTORY_COMPACT_KEEP_MESSAGES: usize = 12;

/// Per-message content cap (chars) applied during overflow recovery. Mirrors
/// the channel runtime's CHANNEL_HISTORY_COMPACT_CONTENT_CHARS.
const API_CHAT_HISTORY_COMPACT_CONTENT_CHARS: usize = 600;

/// Trim `/api/chat` history to the hard cap, preserving the leading system
/// message and cutting at a user-turn boundary. Cutting mid-turn could leave
/// an orphaned `role:"tool"` message at the front, which OpenAI-compatible
/// endpoints reject because it lacks the assistant tool_calls message that
/// must precede it.
fn trim_api_chat_history(history: &mut Vec<crate::providers::ChatMessage>) {
    if history.len() <= MAX_API_CHAT_HISTORY_MESSAGES {
        return;
    }
    let body_start = usize::from(history.first().is_some_and(|m| m.role == "system"));
    let mut cut = history.len() + body_start - MAX_API_CHAT_HISTORY_MESSAGES;
    while cut < history.len() && history[cut].role != "user" {
        cut += 1;
    }
    if cut >= history.len() {
        // No clean user boundary in range — a single heavy agentic turn can
        // append 60+ scaffolding messages after its user message. Fall back
        // to compaction, which keeps the most recent plain turns (including
        // the final answer) instead of wiping the conversation.
        compact_api_chat_history(history);
        return;
    }
    history.drain(body_start..cut);
}

/// True for messages that are tool-call scaffolding rather than plain
/// conversation: (a) assistant messages whose content is the JSON envelope
/// the agent loop stores for native tool calls ({"content":...,
/// "tool_calls":[...]}). Providers re-parse that JSON into a native
/// assistant message with `tool_calls` set on resend, so keeping one
/// without its `role:"tool"` responses produces an orphaned tool_calls
/// message that strict OpenAI-compatible endpoints reject with 400;
/// (b) the synthetic "[Tool results]" user messages of prompt-guided mode.
fn is_tool_call_scaffolding(msg: &crate::providers::ChatMessage) -> bool {
    if msg.role == "assistant" {
        return serde_json::from_str::<serde_json::Value>(&msg.content)
            .is_ok_and(|v| v.get("tool_calls").is_some());
    }
    msg.role == "user" && msg.content.starts_with("[Tool results]")
}

/// Compact `/api/chat` history after a context-window overflow: keep the
/// leading system message plus the most recent plain user/assistant turns
/// (content-capped), dropping tool-call scaffolding entirely. Returns true
/// if the history changed. Parity with the channel runtime's
/// `compact_sender_history`, which only ever stores plain turns.
fn compact_api_chat_history(history: &mut Vec<crate::providers::ChatMessage>) -> bool {
    if history.is_empty() {
        return false;
    }
    let original_len = history.len();
    let system = history.first().filter(|m| m.role == "system").cloned();
    let body_start = usize::from(system.is_some());

    let mut turns: Vec<crate::providers::ChatMessage> = history[body_start..]
        .iter()
        .filter(|m| {
            (m.role == "user" || m.role == "assistant")
                && !m.content.trim().is_empty()
                && !is_tool_call_scaffolding(m)
        })
        .cloned()
        .collect();
    let keep_from = turns
        .len()
        .saturating_sub(API_CHAT_HISTORY_COMPACT_KEEP_MESSAGES);
    turns.drain(..keep_from);

    let mut truncated_any = false;
    for turn in &mut turns {
        if turn.content.chars().count() > API_CHAT_HISTORY_COMPACT_CONTENT_CHARS {
            turn.content = crate::util::truncate_with_ellipsis(
                &turn.content,
                API_CHAT_HISTORY_COMPACT_CONTENT_CHARS,
            );
            truncated_any = true;
        }
    }

    let mut rebuilt = Vec::with_capacity(turns.len() + 1);
    if let Some(sys) = system {
        rebuilt.push(sys);
    }
    rebuilt.extend(turns);

    let changed = truncated_any || rebuilt.len() != original_len;
    *history = rebuilt;
    changed
}

/// Per-request observer wrapper that intercepts `ToolCallStart { tool: "sop_execute" }`
/// to (a) fire an event webhook and (b) signal the gateway to return early so the
/// user gets an immediate "task started" reply while the agent loop continues in the
/// background. Also fires a "done" webhook when `ChatTurnCompleted` is received.
///
/// Wraps the global observer — all other events pass through unchanged.
struct SopStartedSignal {
    inner: Arc<dyn crate::observability::Observer>,
    /// Oneshot sender: fires with the SOP name when `sop_execute` is first called.
    /// Allows the gateway to return early before the agent loop completes.
    sop_tx: parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<String>>>,
    /// Config for the outgoing event webhook (URL, secret, owner openid).
    webhook_url: Option<String>,
    webhook_secret: Option<String>,
    owner_openid: Option<String>,
    /// Tracks which SOP was started so the "done" webhook carries the same name.
    sop_name_started: parking_lot::Mutex<Option<String>>,
    /// Set when any ToolCallStart is observed for this request. Tools may have
    /// side effects, so a turn that already ran one must not be auto-retried.
    any_tool_started: std::sync::atomic::AtomicBool,
    /// SSE broadcast：把 SOP 执行结果实时推给前台客户端（sop_result 事件）。
    event_tx: Option<tokio::sync::broadcast::Sender<serde_json::Value>>,
}

impl SopStartedSignal {
    fn new(
        inner: Arc<dyn crate::observability::Observer>,
        sop_tx: tokio::sync::oneshot::Sender<String>,
        webhook_url: Option<String>,
        webhook_secret: Option<String>,
        owner_openid: Option<String>,
        event_tx: Option<tokio::sync::broadcast::Sender<serde_json::Value>>,
    ) -> Self {
        Self {
            inner,
            sop_tx: parking_lot::Mutex::new(Some(sop_tx)),
            webhook_url,
            webhook_secret,
            owner_openid,
            sop_name_started: parking_lot::Mutex::new(None),
            any_tool_started: std::sync::atomic::AtomicBool::new(false),
            event_tx,
        }
    }

    /// True once any tool has started executing for this request (including
    /// `sop_execute`). Used to suppress the overflow retry: tools that
    /// already ran may have side effects (shell commands, file writes,
    /// outbound messages, SOP starts), and re-running the turn would execute
    /// them a second time. The channel runtime avoids auto-retry entirely for
    /// the same reason; here we retry only when the overflow happened on the
    /// turn's first LLM call, before any tool ran.
    fn any_tool_started(&self) -> bool {
        self.any_tool_started
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Fire an event payload to the configured webhook URL in a detached OS thread.
    fn fire_webhook(&self, payload: serde_json::Value) {
        let Some(url) = self.webhook_url.clone() else {
            return;
        };
        let secret = self.webhook_secret.clone();
        std::thread::spawn(move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default();
            let mut req = client.post(&url).json(&payload);
            if let Some(s) = &secret {
                req = req.header("X-Webhook-Secret", s.as_str());
            }
            if let Err(e) = req.send() {
                tracing::warn!("SOP event webhook failed: {e}");
            }
        });
    }
}

impl crate::observability::Observer for SopStartedSignal {
    fn record_event(&self, event: &crate::observability::ObserverEvent) {
        use crate::observability::ObserverEvent;
        self.inner.record_event(event);
        match event {
            ObserverEvent::ToolCallStart { tool, arguments } => {
                // Any tool start makes the turn unsafe to auto-retry: tools
                // may have side effects (shell, file writes, outbound sends).
                self.any_tool_started
                    .store(true, std::sync::atomic::Ordering::Relaxed);

                if tool == "sop_execute" {
                    // Parse SOP name from the tool arguments JSON string.
                    // LLM 对参数名不忠实：name / sop_name / sop 都见过
                    let sop_name = arguments
                        .as_ref()
                        .and_then(|a| serde_json::from_str::<serde_json::Value>(a).ok())
                        .and_then(|v| {
                            ["name", "sop_name", "sop"]
                                .iter()
                                .find_map(|k| v.get(k).and_then(|n| n.as_str()).map(String::from))
                        })
                        .unwrap_or_else(|| "unknown".to_string());

                    *self.sop_name_started.lock() = Some(sop_name.clone());

                    // Signal gateway to return early.
                    if let Some(tx) = self.sop_tx.lock().take() {
                        let _ = tx.send(sop_name.clone());
                    }

                    // Fire "starting" webhook so consumers can create a pending task immediately.
                    self.fire_webhook(serde_json::json!({
                        "event": "starting",
                        "sop_name": sop_name,
                        "openid": self.owner_openid,
                    }));
                }
            }
            ObserverEvent::ChatTurnCompleted { response_text } => {
                // Fire "done" webhook only if a SOP was started this turn.
                if let Some(sop_name) = self.sop_name_started.lock().as_deref() {
                    self.fire_webhook(serde_json::json!({
                        "event": "done",
                        "sop_name": sop_name,
                        "openid": self.owner_openid,
                        "response_text": response_text,
                    }));
                    // 前台客户端经 SSE 实时收到执行结果（否则早返回后结果无人接收）
                    if let Some(tx) = &self.event_tx {
                        let _ = tx.send(serde_json::json!({
                            "type": "sop_result",
                            "sop_name": sop_name,
                            "response": response_text,
                            "timestamp": chrono::Utc::now().timestamp(),
                        }));
                    }
                }
            }
            _ => {}
        }
    }

    fn record_metric(&self, metric: &crate::observability::ObserverMetric) {
        self.inner.record_metric(metric);
    }

    fn name(&self) -> &str {
        "sop-started-signal"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// POST /api/chat — Bearer-authenticated chat that runs the full agent
/// loop (tool execution, multi-turn LLM calls). Same auth/rate-limit/
/// idempotency contract as `/webhook`, but routes through
/// `run_gateway_chat_with_tools` so tool_calls are actually executed
/// instead of being returned verbatim as text.
///
/// Request:  POST /api/chat   {"message": "..."}   Authorization: Bearer <token>
/// Response: 200 {"response": "...", "model": "..."}
async fn handle_api_chat(
    State(state): State<AppState>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Json<WebhookBody>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let rate_key =
        client_key_from_request(Some(peer_addr), &headers, state.trust_forwarded_headers);
    if !state.rate_limiter.allow_webhook(&rate_key) {
        tracing::warn!("/api/chat rate limit exceeded");
        let err = serde_json::json!({
            "error": "Too many chat requests. Please retry later.",
            "retry_after": RATE_LIMIT_WINDOW_SECS,
        });
        return (StatusCode::TOO_MANY_REQUESTS, Json(err));
    }

    // Bearer auth (same model as /webhook, no X-Webhook-Secret layer)
    if state.pairing.require_pairing() {
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth.strip_prefix("Bearer ").unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            tracing::warn!("/api/chat: rejected — invalid bearer token");
            let err = serde_json::json!({
                "error": "Unauthorized — pair first via POST /pair, then send Authorization: Bearer <token>"
            });
            return (StatusCode::UNAUTHORIZED, Json(err));
        }
    }

    let Json(req_body) = match body {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("/api/chat JSON parse error: {e}");
            let err = serde_json::json!({
                "error": "Invalid JSON body. Expected: {\"message\": \"...\"}"
            });
            return (StatusCode::BAD_REQUEST, Json(err));
        }
    };

    if let Some(idempotency_key) = headers
        .get("X-Idempotency-Key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if !state.idempotency_store.record_if_new(idempotency_key) {
            tracing::info!("/api/chat duplicate ignored (idempotency key: {idempotency_key})");
            let body = serde_json::json!({
                "status": "duplicate",
                "idempotent": true,
                "message": "Request already processed for this idempotency key"
            });
            return (StatusCode::OK, Json(body));
        }
    }

    let message = req_body.message.clone();

    if state.auto_save {
        let key = webhook_memory_key();
        let _ = state
            .mem
            .store(
                &key,
                &message,
                MemoryCategory::Conversation,
                Some(crate::memory::GATEWAY_API_CHAT_SESSION_ID),
            )
            .await;
    }

    let provider_label = state
        .config
        .lock()
        .default_provider
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let model_label = state.model.clone();
    let started_at = Instant::now();

    // Persistent multi-turn history: clone out, call agent, store back.
    // One daemon = one user, so /api/chat is a single memory scope
    // (`GATEWAY_API_CHAT_SESSION_ID`) shared by every caller of this endpoint.
    let prior_history = {
        let guard = state.api_chat_history.lock();
        if guard.is_empty() {
            None
        } else {
            Some(guard.clone())
        }
    };

    let config = state.config.lock().clone();
    let webhook_url = config.observability.event_webhook_url.clone();
    let webhook_secret = config.observability.event_webhook_secret.clone();
    let owner_openid = config.gateway.owner_openid.clone();

    // Wrap the global observer in a per-request SopStartedSignal.  When the
    // agent calls `sop_execute`, the signal fires immediately and lets the
    // gateway return an early "task started" reply while the agent loop
    // continues in a background thread (block_in_place occupies one worker).
    let (sop_tx, sop_rx) = tokio::sync::oneshot::channel::<String>();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel::<
        anyhow::Result<(String, Vec<crate::providers::ChatMessage>)>,
    >();

    let signal_obs = Arc::new(SopStartedSignal::new(
        state.observer.clone(),
        sop_tx,
        webhook_url,
        webhook_secret,
        owner_openid,
        Some(state.event_tx.clone()),
    ));
    let signal_obs_bg = signal_obs.clone();
    let history_store = state.api_chat_history.clone();
    let global_obs = state.observer.clone();
    let model_label_clone = state.model.clone();
    // 共享 daemon SopEngine：/api/chat 里启动的 SOP run 才能被 /sop/approve 批到
    let sop_engine_shared = state.sop_engine.clone();

    // Spawn the agent loop as a separate tokio task. process_message_with_history
    // internally uses block_in_place + an isolated single-thread runtime, which
    // is compatible with tokio::spawn on a multi-thread runtime — block_in_place
    // moves the worker out so other tasks continue while this one blocks.
    tokio::spawn(async move {
        let mut result = crate::agent::process_message_with_history(
            config.clone(),
            &message,
            prior_history,
            Some(signal_obs_bg.clone() as Arc<dyn crate::observability::Observer>),
            Some(sop_engine_shared.clone()),
            Some(crate::memory::GATEWAY_API_CHAT_SESSION_ID),
        )
        .await;

        // Context-window overflow self-heal: compact the stored history to
        // recent plain turns so the session un-wedges (the channel runtime
        // does the same via compact_sender_history). Additionally, when the
        // overflow happened on the turn's first LLM call — before any tool
        // ran — retry once with the compacted context so the caller gets a
        // normal answer instead of an error. If a tool already executed, the
        // retry is skipped (it would re-run side-effectful tools); the error
        // propagates, but the compacted history makes the next request work.
        if let Err(e) = &result {
            if crate::providers::reliable::is_context_window_overflow_error(e) {
                let compacted = {
                    let mut guard = history_store.lock();
                    compact_api_chat_history(&mut guard).then(|| guard.clone())
                };
                match compacted {
                    Some(prior) if !signal_obs_bg.any_tool_started() => {
                        tracing::warn!(
                            "/api/chat hit the model context window before any tool ran; \
                             compacted history and retrying once"
                        );
                        result = crate::agent::process_message_with_history(
                            config,
                            &message,
                            Some(prior),
                            Some(signal_obs_bg.clone() as Arc<dyn crate::observability::Observer>),
                            Some(sop_engine_shared.clone()),
                            Some(crate::memory::GATEWAY_API_CHAT_SESSION_ID),
                        )
                        .await;
                    }
                    Some(_) => {
                        tracing::warn!(
                            "/api/chat hit the model context window mid-turn; compacted \
                             stored history (no retry — tools already executed this turn)"
                        );
                    }
                    None => {}
                }
            }
        }

        // Update conversation history and fire ChatTurnCompleted regardless of
        // whether the gateway already returned early (SOP case) or is still waiting.
        match &result {
            Ok((response_text, new_history)) => {
                {
                    let mut stored = new_history.clone();
                    trim_api_chat_history(&mut stored);
                    *history_store.lock() = stored;
                }
                crate::observability::Observer::record_event(
                    signal_obs_bg.as_ref(),
                    &crate::observability::ObserverEvent::ChatTurnCompleted {
                        response_text: response_text.clone(),
                    },
                );
                global_obs.record_metric(&crate::observability::ObserverMetric::RequestLatency(
                    started_at.elapsed(),
                ));
            }
            Err(e) => {
                let sanitized = providers::sanitize_api_error(&e.to_string());
                global_obs.record_event(&crate::observability::ObserverEvent::Error {
                    component: "gateway".to_string(),
                    message: sanitized,
                });
            }
        }
        let _ = result_tx.send(result);
    });

    let _ = (provider_label, model_label);

    // Select: return as soon as the agent finishes (quick non-SOP path) OR as
    // soon as a SOP is detected (early-return path — agent keeps running in bg).
    tokio::select! {
        res = result_rx => {
            // Agent completed before (or without) triggering a SOP.
            match res {
                Ok(Ok((response, _))) => {
                    let body = serde_json::json!({"response": response, "model": model_label_clone});
                    (StatusCode::OK, Json(body))
                }
                Ok(Err(e)) => {
                    let sanitized = providers::sanitize_api_error(&e.to_string());
                    tracing::error!("/api/chat agent error: {}", sanitized);
                    let err = serde_json::json!({"error": "Chat failed", "detail": sanitized});
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(err))
                }
                Err(_) => {
                    // Channel closed — background thread panicked
                    let err = serde_json::json!({"error": "Chat failed", "detail": "agent thread panicked"});
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(err))
                }
            }
        }
        Ok(sop_name) = sop_rx => {
            // SOP detected — return immediately; agent loop continues in background.
            // The SopStartedSignal will fire the "done" webhook when it finishes.
            tracing::info!(sop_name = %sop_name, "SOP started — returning early from /api/chat");
            let body = serde_json::json!({
                "response": if sop_name == "unknown" {
                    "已为您发起流程，正在后台执行；执行结果和需要你确认的步骤都会自动出现。".to_string()
                } else {
                    format!("已为您发起「{sop_name}」，正在后台执行；执行结果和需要你确认的步骤都会自动出现。")
                },
                "model": model_label_clone,
                "sop_started": true,
                "sop_name": sop_name,
            });
            (StatusCode::OK, Json(body))
        }
    }
}

/// `WhatsApp` verification query params
#[derive(serde::Deserialize)]
pub struct WhatsAppVerifyQuery {
    #[serde(rename = "hub.mode")]
    pub mode: Option<String>,
    #[serde(rename = "hub.verify_token")]
    pub verify_token: Option<String>,
    #[serde(rename = "hub.challenge")]
    pub challenge: Option<String>,
}

/// GET /whatsapp — Meta webhook verification
async fn handle_whatsapp_verify(
    State(state): State<AppState>,
    Query(params): Query<WhatsAppVerifyQuery>,
) -> impl IntoResponse {
    let Some(ref wa) = state.whatsapp else {
        return (StatusCode::NOT_FOUND, "WhatsApp not configured".to_string());
    };

    // Verify the token matches (constant-time comparison to prevent timing attacks)
    let token_matches = params
        .verify_token
        .as_deref()
        .is_some_and(|t| constant_time_eq(t, wa.verify_token()));
    if params.mode.as_deref() == Some("subscribe") && token_matches {
        if let Some(ch) = params.challenge {
            tracing::info!("WhatsApp webhook verified successfully");
            return (StatusCode::OK, ch);
        }
        return (StatusCode::BAD_REQUEST, "Missing hub.challenge".to_string());
    }

    tracing::warn!("WhatsApp webhook verification failed — token mismatch");
    (StatusCode::FORBIDDEN, "Forbidden".to_string())
}

/// Verify `WhatsApp` webhook signature (`X-Hub-Signature-256`).
/// Returns true if the signature is valid, false otherwise.
/// See: <https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests>
pub fn verify_whatsapp_signature(app_secret: &str, body: &[u8], signature_header: &str) -> bool {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    // Signature format: "sha256=<hex_signature>"
    let Some(hex_sig) = signature_header.strip_prefix("sha256=") else {
        return false;
    };

    // Decode hex signature
    let Ok(expected) = hex::decode(hex_sig) else {
        return false;
    };

    // Compute HMAC-SHA256
    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(app_secret.as_bytes()) else {
        return false;
    };
    mac.update(body);

    // Constant-time comparison
    mac.verify_slice(&expected).is_ok()
}

/// POST /whatsapp — incoming message webhook
async fn handle_whatsapp_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let Some(ref wa) = state.whatsapp else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "WhatsApp not configured"})),
        );
    };

    // ── Security: Verify X-Hub-Signature-256 if app_secret is configured ──
    if let Some(ref app_secret) = state.whatsapp_app_secret {
        let signature = headers
            .get("X-Hub-Signature-256")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !verify_whatsapp_signature(app_secret, &body, signature) {
            tracing::warn!(
                "WhatsApp webhook signature verification failed (signature: {})",
                if signature.is_empty() {
                    "missing"
                } else {
                    "invalid"
                }
            );
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid signature"})),
            );
        }
    }

    // Parse JSON body
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid JSON payload"})),
        );
    };

    // Parse messages from the webhook payload
    let messages = wa.parse_webhook_payload(&payload);

    if messages.is_empty() {
        // Acknowledge the webhook even if no messages (could be status updates)
        return (StatusCode::OK, Json(serde_json::json!({"status": "ok"})));
    }

    // Process each message
    for msg in &messages {
        tracing::info!(
            "WhatsApp message from {}: {}",
            msg.sender,
            truncate_with_ellipsis(&msg.content, 50)
        );

        // Memory scope for this conversation — shared by the auto-save below
        // and the agent turn, so a conversation only ever recalls its own turns.
        let session_id = crate::channels::conversation_history_key(msg);

        // Auto-save to memory
        if state.auto_save {
            let key = whatsapp_memory_key(msg);
            let _ = state
                .mem
                .store(
                    &key,
                    &msg.content,
                    MemoryCategory::Conversation,
                    Some(&session_id),
                )
                .await;
        }

        match Box::pin(run_gateway_chat_with_tools(
            &state,
            &msg.content,
            Some(&session_id),
        ))
        .await
        {
            Ok(response) => {
                // Send reply via WhatsApp
                if let Err(e) = wa
                    .send(&SendMessage::new(response, &msg.reply_target))
                    .await
                {
                    tracing::error!("Failed to send WhatsApp reply: {e}");
                }
            }
            Err(e) => {
                tracing::error!("LLM error for WhatsApp message: {e:#}");
                let _ = wa
                    .send(&SendMessage::new(
                        "Sorry, I couldn't process your message right now.",
                        &msg.reply_target,
                    ))
                    .await;
            }
        }
    }

    // Acknowledge the webhook
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

/// POST /linq — incoming message webhook (iMessage/RCS/SMS via Linq)
async fn handle_linq_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let Some(ref linq) = state.linq else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Linq not configured"})),
        );
    };

    let body_str = String::from_utf8_lossy(&body);

    // ── Security: Verify X-Webhook-Signature if signing_secret is configured ──
    if let Some(ref signing_secret) = state.linq_signing_secret {
        let timestamp = headers
            .get("X-Webhook-Timestamp")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let signature = headers
            .get("X-Webhook-Signature")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !crate::channels::linq::verify_linq_signature(
            signing_secret,
            &body_str,
            timestamp,
            signature,
        ) {
            tracing::warn!(
                "Linq webhook signature verification failed (signature: {})",
                if signature.is_empty() {
                    "missing"
                } else {
                    "invalid"
                }
            );
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid signature"})),
            );
        }
    }

    // Parse JSON body
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid JSON payload"})),
        );
    };

    // Parse messages from the webhook payload
    let messages = linq.parse_webhook_payload(&payload);

    if messages.is_empty() {
        // Acknowledge the webhook even if no messages (could be status/delivery events)
        return (StatusCode::OK, Json(serde_json::json!({"status": "ok"})));
    }

    // Process each message
    for msg in &messages {
        tracing::info!(
            "Linq message from {}: {}",
            msg.sender,
            truncate_with_ellipsis(&msg.content, 50)
        );

        // Memory scope for this conversation — shared by the auto-save below
        // and the agent turn, so a conversation only ever recalls its own turns.
        let session_id = crate::channels::conversation_history_key(msg);

        // Auto-save to memory
        if state.auto_save {
            let key = linq_memory_key(msg);
            let _ = state
                .mem
                .store(
                    &key,
                    &msg.content,
                    MemoryCategory::Conversation,
                    Some(&session_id),
                )
                .await;
        }

        // Call the LLM
        match Box::pin(run_gateway_chat_with_tools(
            &state,
            &msg.content,
            Some(&session_id),
        ))
        .await
        {
            Ok(response) => {
                // Send reply via Linq
                if let Err(e) = linq
                    .send(&SendMessage::new(response, &msg.reply_target))
                    .await
                {
                    tracing::error!("Failed to send Linq reply: {e}");
                }
            }
            Err(e) => {
                tracing::error!("LLM error for Linq message: {e:#}");
                let _ = linq
                    .send(&SendMessage::new(
                        "Sorry, I couldn't process your message right now.",
                        &msg.reply_target,
                    ))
                    .await;
            }
        }
    }

    // Acknowledge the webhook
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

/// GET /wati — WATI webhook verification (echoes hub.challenge)
async fn handle_wati_verify(
    State(state): State<AppState>,
    Query(params): Query<WatiVerifyQuery>,
) -> impl IntoResponse {
    if state.wati.is_none() {
        return (StatusCode::NOT_FOUND, "WATI not configured".to_string());
    }

    // WATI may use Meta-style webhook verification; echo the challenge
    if let Some(challenge) = params.challenge {
        tracing::info!("WATI webhook verified successfully");
        return (StatusCode::OK, challenge);
    }

    (StatusCode::BAD_REQUEST, "Missing hub.challenge".to_string())
}

#[derive(Debug, serde::Deserialize)]
pub struct WatiVerifyQuery {
    #[serde(rename = "hub.challenge")]
    pub challenge: Option<String>,
}

/// POST /wati — incoming WATI WhatsApp message webhook
async fn handle_wati_webhook(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let Some(ref wati) = state.wati else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "WATI not configured"})),
        );
    };

    // Parse JSON body
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid JSON payload"})),
        );
    };

    // Parse messages from the webhook payload
    let messages = wati.parse_webhook_payload(&payload);

    if messages.is_empty() {
        return (StatusCode::OK, Json(serde_json::json!({"status": "ok"})));
    }

    // Process each message
    for msg in &messages {
        tracing::info!(
            "WATI message from {}: {}",
            msg.sender,
            truncate_with_ellipsis(&msg.content, 50)
        );

        // Memory scope for this conversation — shared by the auto-save below
        // and the agent turn, so a conversation only ever recalls its own turns.
        let session_id = crate::channels::conversation_history_key(msg);

        // Auto-save to memory
        if state.auto_save {
            let key = wati_memory_key(msg);
            let _ = state
                .mem
                .store(
                    &key,
                    &msg.content,
                    MemoryCategory::Conversation,
                    Some(&session_id),
                )
                .await;
        }

        // Call the LLM
        match Box::pin(run_gateway_chat_with_tools(
            &state,
            &msg.content,
            Some(&session_id),
        ))
        .await
        {
            Ok(response) => {
                // Send reply via WATI
                if let Err(e) = wati
                    .send(&SendMessage::new(response, &msg.reply_target))
                    .await
                {
                    tracing::error!("Failed to send WATI reply: {e}");
                }
            }
            Err(e) => {
                tracing::error!("LLM error for WATI message: {e:#}");
                let _ = wati
                    .send(&SendMessage::new(
                        "Sorry, I couldn't process your message right now.",
                        &msg.reply_target,
                    ))
                    .await;
            }
        }
    }

    // Acknowledge the webhook
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

/// POST /nextcloud-talk — incoming message webhook (Nextcloud Talk bot API)
async fn handle_nextcloud_talk_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let Some(ref nextcloud_talk) = state.nextcloud_talk else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Nextcloud Talk not configured"})),
        );
    };

    let body_str = String::from_utf8_lossy(&body);

    // ── Security: Verify Nextcloud Talk HMAC signature if secret is configured ──
    if let Some(ref webhook_secret) = state.nextcloud_talk_webhook_secret {
        let random = headers
            .get("X-Nextcloud-Talk-Random")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let signature = headers
            .get("X-Nextcloud-Talk-Signature")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !crate::channels::nextcloud_talk::verify_nextcloud_talk_signature(
            webhook_secret,
            random,
            &body_str,
            signature,
        ) {
            tracing::warn!(
                "Nextcloud Talk webhook signature verification failed (signature: {})",
                if signature.is_empty() {
                    "missing"
                } else {
                    "invalid"
                }
            );
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid signature"})),
            );
        }
    }

    // Parse JSON body
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid JSON payload"})),
        );
    };

    // Parse messages from webhook payload
    let messages = nextcloud_talk.parse_webhook_payload(&payload);
    if messages.is_empty() {
        // Acknowledge webhook even if payload does not contain actionable user messages.
        return (StatusCode::OK, Json(serde_json::json!({"status": "ok"})));
    }

    for msg in &messages {
        tracing::info!(
            "Nextcloud Talk message from {}: {}",
            msg.sender,
            truncate_with_ellipsis(&msg.content, 50)
        );

        // Memory scope for this conversation — shared by the auto-save below
        // and the agent turn, so a conversation only ever recalls its own turns.
        let session_id = crate::channels::conversation_history_key(msg);

        if state.auto_save {
            let key = nextcloud_talk_memory_key(msg);
            let _ = state
                .mem
                .store(
                    &key,
                    &msg.content,
                    MemoryCategory::Conversation,
                    Some(&session_id),
                )
                .await;
        }

        match Box::pin(run_gateway_chat_with_tools(
            &state,
            &msg.content,
            Some(&session_id),
        ))
        .await
        {
            Ok(response) => {
                if let Err(e) = nextcloud_talk
                    .send(&SendMessage::new(response, &msg.reply_target))
                    .await
                {
                    tracing::error!("Failed to send Nextcloud Talk reply: {e}");
                }
            }
            Err(e) => {
                tracing::error!("LLM error for Nextcloud Talk message: {e:#}");
                let _ = nextcloud_talk
                    .send(&SendMessage::new(
                        "Sorry, I couldn't process your message right now.",
                        &msg.reply_target,
                    ))
                    .await;
            }
        }
    }

    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

// ══════════════════════════════════════════════════════════════════════════════
// ADMIN HANDLERS (for CLI management)
// ══════════════════════════════════════════════════════════════════════════════

/// Response for admin endpoints
#[derive(serde::Serialize)]
struct AdminResponse {
    success: bool,
    message: String,
}

/// Reject requests that do not originate from a loopback address.
fn require_localhost(peer: &SocketAddr) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if peer.ip().is_loopback() {
        Ok(())
    } else {
        Err((
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "Admin endpoints are restricted to localhost"
            })),
        ))
    }
}

/// POST /admin/shutdown — graceful shutdown from CLI (localhost only)
async fn handle_admin_shutdown(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_localhost(&peer)?;
    tracing::info!("🔌 Admin shutdown request received — initiating graceful shutdown");

    let body = AdminResponse {
        success: true,
        message: "Gateway shutdown initiated".to_string(),
    };

    let _ = state.shutdown_tx.send(true);

    Ok((StatusCode::OK, Json(body)))
}

/// GET /admin/paircode — fetch current pairing code (localhost only)
async fn handle_admin_paircode(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_localhost(&peer)?;
    let code = state.pairing.pairing_code();

    let body = if let Some(c) = code {
        serde_json::json!({
            "success": true,
            "pairing_required": state.pairing.require_pairing(),
            "pairing_code": c,
            "message": "Use this one-time code to pair"
        })
    } else {
        serde_json::json!({
            "success": true,
            "pairing_required": state.pairing.require_pairing(),
            "pairing_code": null,
            "message": if state.pairing.require_pairing() {
                "Pairing is active but no new code available (already paired or code expired)"
            } else {
                "Pairing is disabled for this gateway"
            }
        })
    };

    Ok((StatusCode::OK, Json(body)))
}

/// POST /admin/paircode/new — generate a new pairing code (localhost only)
async fn handle_admin_paircode_new(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_localhost(&peer)?;
    match state.pairing.generate_new_pairing_code() {
        Some(code) => {
            tracing::info!("🔐 New pairing code generated via admin endpoint");
            let body = serde_json::json!({
                "success": true,
                "pairing_required": state.pairing.require_pairing(),
                "pairing_code": code,
                "message": "New pairing code generated — use this one-time code to pair"
            });
            Ok((StatusCode::OK, Json(body)))
        }
        None => {
            let body = serde_json::json!({
                "success": false,
                "pairing_required": false,
                "pairing_code": null,
                "message": "Pairing is disabled for this gateway"
            });
            Ok((StatusCode::BAD_REQUEST, Json(body)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::traits::ChannelMessage;
    use crate::memory::{Memory, MemoryCategory, MemoryEntry};
    use crate::providers::Provider;
    use async_trait::async_trait;
    use axum::http::HeaderValue;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Generate a random hex secret at runtime to avoid hard-coded cryptographic values.
    fn generate_test_secret() -> String {
        let bytes: [u8; 32] = rand::random();
        hex::encode(bytes)
    }

    #[test]
    fn security_body_limit_is_64kb() {
        assert_eq!(MAX_BODY_SIZE, 65_536);
    }

    #[test]
    fn trim_api_chat_history_noop_under_cap() {
        let mut history = vec![
            crate::providers::ChatMessage::system("sys"),
            crate::providers::ChatMessage::user("hi"),
            crate::providers::ChatMessage::assistant("hello"),
        ];
        let before = history.clone();
        trim_api_chat_history(&mut history);
        assert_eq!(history.len(), before.len());
        assert_eq!(history[0].content, "sys");
    }

    #[test]
    fn trim_api_chat_history_preserves_system_and_user_boundary() {
        let mut history = vec![crate::providers::ChatMessage::system("sys")];
        // 30 agentic turns × 4 messages = 120 entries, far over the cap.
        for i in 0..30 {
            history.push(crate::providers::ChatMessage::user(format!("question {i}")));
            history.push(crate::providers::ChatMessage::assistant(format!(
                "calling tool {i}"
            )));
            history.push(crate::providers::ChatMessage::tool(format!(
                "{{\"tool_call_id\":\"call_{i}\",\"content\":\"result {i}\"}}"
            )));
            history.push(crate::providers::ChatMessage::assistant(format!(
                "answer {i}"
            )));
        }
        trim_api_chat_history(&mut history);
        assert!(history.len() <= MAX_API_CHAT_HISTORY_MESSAGES);
        assert_eq!(history[0].role, "system");
        // The first retained message after the system prompt must start a
        // clean turn — never an orphaned tool result.
        assert_eq!(history[1].role, "user");
    }

    #[test]
    fn trim_api_chat_history_drops_body_without_user_boundary() {
        let mut history = vec![crate::providers::ChatMessage::system("sys")];
        for i in 0..(MAX_API_CHAT_HISTORY_MESSAGES + 5) {
            history.push(crate::providers::ChatMessage::tool(format!("orphan {i}")));
        }
        trim_api_chat_history(&mut history);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");
    }

    #[test]
    fn compact_api_chat_history_keeps_recent_plain_turns() {
        let long_text = "x".repeat(5 * API_CHAT_HISTORY_COMPACT_CONTENT_CHARS);
        let mut history = vec![crate::providers::ChatMessage::system("sys")];
        for i in 0..20 {
            history.push(crate::providers::ChatMessage::user(format!(
                "q{i} {long_text}"
            )));
            history.push(crate::providers::ChatMessage::assistant(format!(
                "tool call {i}"
            )));
            history.push(crate::providers::ChatMessage::tool(long_text.clone()));
            history.push(crate::providers::ChatMessage::assistant(format!(
                "a{i} {long_text}"
            )));
        }

        assert!(compact_api_chat_history(&mut history));
        assert_eq!(history[0].role, "system");
        assert!(history.len() <= API_CHAT_HISTORY_COMPACT_KEEP_MESSAGES + 1);
        assert!(history.iter().all(|m| m.role != "tool"));
        // truncate_with_ellipsis appends "..." (3 chars) after the cap.
        assert!(history
            .iter()
            .all(|m| m.content.chars().count() <= API_CHAT_HISTORY_COMPACT_CONTENT_CHARS + 3));
        // Most recent turn must survive compaction.
        assert!(history.iter().any(|m| m.content.starts_with("a19")));
    }

    #[test]
    fn compact_api_chat_history_drops_native_tool_call_scaffolding() {
        // Native-tools mode stores assistant tool-call turns as a JSON
        // envelope; keeping it without its role:"tool" responses produces an
        // orphaned tool_calls message that strict endpoints reject with 400.
        let scaffolding =
            r#"{"content":null,"tool_calls":[{"id":"call_1","name":"shell","arguments":"{}"}]}"#;
        let mut history = vec![
            crate::providers::ChatMessage::system("sys"),
            crate::providers::ChatMessage::user("question"),
            crate::providers::ChatMessage::assistant(scaffolding),
            crate::providers::ChatMessage::tool(r#"{"tool_call_id":"call_1","content":"result"}"#),
            crate::providers::ChatMessage::user("[Tool results]\nraw dump"),
            crate::providers::ChatMessage::assistant("final answer"),
        ];

        assert!(compact_api_chat_history(&mut history));
        assert!(history
            .iter()
            .all(|m| !m.content.contains("tool_calls") && m.role != "tool"));
        assert!(history
            .iter()
            .all(|m| !m.content.starts_with("[Tool results]")));
        assert!(history.iter().any(|m| m.content == "question"));
        assert!(history.iter().any(|m| m.content == "final answer"));
    }

    #[test]
    fn trim_api_chat_history_heavy_turn_falls_back_to_compaction() {
        // A single agentic turn appending 60+ scaffolding messages after its
        // user message must not wipe the conversation: the fallback compacts
        // to recent plain turns, preserving the final answer.
        let scaffolding =
            r#"{"content":null,"tool_calls":[{"id":"c","name":"shell","arguments":"{}"}]}"#;
        let mut history = vec![
            crate::providers::ChatMessage::system("sys"),
            crate::providers::ChatMessage::user("the question"),
        ];
        for _ in 0..35 {
            history.push(crate::providers::ChatMessage::assistant(scaffolding));
            history.push(crate::providers::ChatMessage::tool(
                r#"{"tool_call_id":"c","content":"r"}"#,
            ));
        }
        history.push(crate::providers::ChatMessage::assistant("final answer"));

        trim_api_chat_history(&mut history);
        assert_eq!(history[0].role, "system");
        assert!(history.iter().any(|m| m.content == "the question"));
        assert!(history.iter().any(|m| m.content == "final answer"));
        assert!(history
            .iter()
            .all(|m| m.role != "tool" && !m.content.contains("tool_calls")));
    }

    #[test]
    fn compact_api_chat_history_false_when_empty_or_already_small() {
        let mut empty: Vec<crate::providers::ChatMessage> = Vec::new();
        assert!(!compact_api_chat_history(&mut empty));

        let mut small = vec![
            crate::providers::ChatMessage::system("sys"),
            crate::providers::ChatMessage::user("hi"),
            crate::providers::ChatMessage::assistant("hello"),
        ];
        assert!(!compact_api_chat_history(&mut small));
        assert_eq!(small.len(), 3);
    }

    #[test]
    fn security_timeout_matches_constant() {
        // Bumped from 30s to 300s in commit e2051da2 — slow-loris is now
        // handled at the reverse proxy, the inner deadline only needs to
        // bound pathological agent loops.
        assert_eq!(REQUEST_TIMEOUT_SECS, 300);
    }

    #[test]
    fn webhook_body_requires_message_field() {
        let valid = r#"{"message": "hello"}"#;
        let parsed: Result<WebhookBody, _> = serde_json::from_str(valid);
        assert!(parsed.is_ok());
        assert_eq!(parsed.unwrap().message, "hello");

        let missing = r#"{"other": "field"}"#;
        let parsed: Result<WebhookBody, _> = serde_json::from_str(missing);
        assert!(parsed.is_err());
    }

    #[test]
    fn whatsapp_query_fields_are_optional() {
        let q = WhatsAppVerifyQuery {
            mode: None,
            verify_token: None,
            challenge: None,
        };
        assert!(q.mode.is_none());
    }

    #[test]
    fn app_state_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<AppState>();
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_hint_when_prometheus_is_disabled() {
        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider: Arc::new(MockProvider::default()),
            model: "test-model".into(),
            temperature: 0.0,
            mem: Arc::new(MockMemory),
            auto_save: false,
            webhook_secret_hash: None,
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let response = handle_metrics(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some(PROMETHEUS_CONTENT_TYPE)
        );

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("Prometheus backend not enabled"));
    }

    #[tokio::test]
    async fn metrics_endpoint_renders_prometheus_output() {
        let prom = Arc::new(crate::observability::PrometheusObserver::new());
        crate::observability::Observer::record_event(
            prom.as_ref(),
            &crate::observability::ObserverEvent::HeartbeatTick,
        );

        let observer: Arc<dyn crate::observability::Observer> = prom;
        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider: Arc::new(MockProvider::default()),
            model: "test-model".into(),
            temperature: 0.0,
            mem: Arc::new(MockMemory),
            auto_save: false,
            webhook_secret_hash: None,
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer,
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let response = handle_metrics(State(state)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("zeroclaw_heartbeat_ticks_total 1"));
    }

    #[test]
    fn gateway_rate_limiter_blocks_after_limit() {
        let limiter = GatewayRateLimiter::new(2, 2, 100);
        assert!(limiter.allow_pair("127.0.0.1"));
        assert!(limiter.allow_pair("127.0.0.1"));
        assert!(!limiter.allow_pair("127.0.0.1"));
    }

    #[test]
    fn rate_limiter_sweep_removes_stale_entries() {
        let limiter = SlidingWindowRateLimiter::new(10, Duration::from_secs(60), 100);
        // Add entries for multiple IPs
        assert!(limiter.allow("ip-1"));
        assert!(limiter.allow("ip-2"));
        assert!(limiter.allow("ip-3"));

        {
            let guard = limiter.requests.lock();
            assert_eq!(guard.0.len(), 3);
        }

        // Force a sweep by backdating last_sweep
        {
            let mut guard = limiter.requests.lock();
            guard.1 = Instant::now()
                .checked_sub(Duration::from_secs(RATE_LIMITER_SWEEP_INTERVAL_SECS + 1))
                .unwrap();
            // Clear timestamps for ip-2 and ip-3 to simulate stale entries
            guard.0.get_mut("ip-2").unwrap().clear();
            guard.0.get_mut("ip-3").unwrap().clear();
        }

        // Next allow() call should trigger sweep and remove stale entries
        assert!(limiter.allow("ip-1"));

        {
            let guard = limiter.requests.lock();
            assert_eq!(guard.0.len(), 1, "Stale entries should have been swept");
            assert!(guard.0.contains_key("ip-1"));
        }
    }

    #[test]
    fn rate_limiter_zero_limit_always_allows() {
        let limiter = SlidingWindowRateLimiter::new(0, Duration::from_secs(60), 10);
        for _ in 0..100 {
            assert!(limiter.allow("any-key"));
        }
    }

    #[test]
    fn idempotency_store_rejects_duplicate_key() {
        let store = IdempotencyStore::new(Duration::from_secs(30), 10);
        assert!(store.record_if_new("req-1"));
        assert!(!store.record_if_new("req-1"));
        assert!(store.record_if_new("req-2"));
    }

    #[test]
    fn rate_limiter_bounded_cardinality_evicts_oldest_key() {
        let limiter = SlidingWindowRateLimiter::new(5, Duration::from_secs(60), 2);
        assert!(limiter.allow("ip-1"));
        assert!(limiter.allow("ip-2"));
        assert!(limiter.allow("ip-3"));

        let guard = limiter.requests.lock();
        assert_eq!(guard.0.len(), 2);
        assert!(guard.0.contains_key("ip-2"));
        assert!(guard.0.contains_key("ip-3"));
    }

    #[test]
    fn idempotency_store_bounded_cardinality_evicts_oldest_key() {
        let store = IdempotencyStore::new(Duration::from_secs(300), 2);
        assert!(store.record_if_new("k1"));
        std::thread::sleep(Duration::from_millis(2));
        assert!(store.record_if_new("k2"));
        std::thread::sleep(Duration::from_millis(2));
        assert!(store.record_if_new("k3"));

        let keys = store.keys.lock();
        assert_eq!(keys.len(), 2);
        assert!(!keys.contains_key("k1"));
        assert!(keys.contains_key("k2"));
        assert!(keys.contains_key("k3"));
    }

    #[test]
    fn client_key_defaults_to_peer_addr_when_untrusted_proxy_mode() {
        let peer = SocketAddr::from(([10, 0, 0, 5], 42617));
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Forwarded-For",
            HeaderValue::from_static("198.51.100.10, 203.0.113.11"),
        );

        let key = client_key_from_request(Some(peer), &headers, false);
        assert_eq!(key, "10.0.0.5");
    }

    #[test]
    fn client_key_uses_forwarded_ip_only_in_trusted_proxy_mode() {
        let peer = SocketAddr::from(([10, 0, 0, 5], 42617));
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Forwarded-For",
            HeaderValue::from_static("198.51.100.10, 203.0.113.11"),
        );

        let key = client_key_from_request(Some(peer), &headers, true);
        assert_eq!(key, "198.51.100.10");
    }

    #[test]
    fn client_key_falls_back_to_peer_when_forwarded_header_invalid() {
        let peer = SocketAddr::from(([10, 0, 0, 5], 42617));
        let mut headers = HeaderMap::new();
        headers.insert("X-Forwarded-For", HeaderValue::from_static("garbage-value"));

        let key = client_key_from_request(Some(peer), &headers, true);
        assert_eq!(key, "10.0.0.5");
    }

    #[test]
    fn normalize_max_keys_uses_fallback_for_zero() {
        assert_eq!(normalize_max_keys(0, 10_000), 10_000);
        assert_eq!(normalize_max_keys(0, 0), 1);
    }

    #[test]
    fn normalize_max_keys_preserves_nonzero_values() {
        assert_eq!(normalize_max_keys(2_048, 10_000), 2_048);
        assert_eq!(normalize_max_keys(1, 10_000), 1);
    }

    #[tokio::test]
    async fn persist_pairing_tokens_writes_config_tokens() {
        let temp = tempfile::tempdir().unwrap();
        let config_path = temp.path().join("config.toml");
        let workspace_path = temp.path().join("workspace");

        let mut config = Config::default();
        config.config_path = config_path.clone();
        config.workspace_dir = workspace_path;
        config.save().await.unwrap();

        let guard = PairingGuard::new(true, &[]);
        let code = guard.pairing_code().unwrap();
        let token = guard.try_pair(&code, "test_client").await.unwrap().unwrap();
        assert!(guard.is_authenticated(&token));

        let shared_config = Arc::new(Mutex::new(config));
        persist_pairing_tokens(shared_config.clone(), &guard)
            .await
            .unwrap();

        // In-memory tokens should remain as plaintext 64-char hex hashes.
        let plaintext = {
            let in_memory = shared_config.lock();
            assert_eq!(in_memory.gateway.paired_tokens.len(), 1);
            in_memory.gateway.paired_tokens[0].clone()
        };
        assert_eq!(plaintext.len(), 64);
        assert!(plaintext.chars().all(|c: char| c.is_ascii_hexdigit()));

        // On disk, the token should be encrypted (secrets.encrypt defaults to true).
        let saved = tokio::fs::read_to_string(config_path).await.unwrap();
        let raw_parsed: Config = toml::from_str(&saved).unwrap();
        assert_eq!(raw_parsed.gateway.paired_tokens.len(), 1);
        let on_disk = &raw_parsed.gateway.paired_tokens[0];
        assert!(
            crate::security::SecretStore::is_encrypted(on_disk),
            "paired_token should be encrypted on disk"
        );
    }

    #[test]
    fn webhook_memory_key_is_unique() {
        let key1 = webhook_memory_key();
        let key2 = webhook_memory_key();

        assert!(key1.starts_with("webhook_msg_"));
        assert!(key2.starts_with("webhook_msg_"));
        assert_ne!(key1, key2);
    }

    #[test]
    fn whatsapp_memory_key_includes_sender_and_message_id() {
        let msg = ChannelMessage {
            id: "wamid-123".into(),
            sender: "+1234567890".into(),
            reply_target: "+1234567890".into(),
            content: "hello".into(),
            channel: "whatsapp".into(),
            timestamp: 1,
            thread_ts: None,
        };

        let key = whatsapp_memory_key(&msg);
        assert_eq!(key, "whatsapp_+1234567890_wamid-123");
    }

    #[derive(Default)]
    struct MockMemory;

    #[async_trait]
    impl Memory for MockMemory {
        fn name(&self) -> &str {
            "mock"
        }

        async fn store(
            &self,
            _key: &str,
            _content: &str,
            _category: MemoryCategory,
            _session_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn recall(
            &self,
            _query: &str,
            _limit: usize,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(Vec::new())
        }

        async fn get(&self, _key: &str) -> anyhow::Result<Option<MemoryEntry>> {
            Ok(None)
        }

        async fn list(
            &self,
            _category: Option<&MemoryCategory>,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(Vec::new())
        }

        async fn forget(&self, _key: &str) -> anyhow::Result<bool> {
            Ok(false)
        }

        async fn count(&self) -> anyhow::Result<usize> {
            Ok(0)
        }

        async fn health_check(&self) -> bool {
            true
        }
    }

    #[derive(Default)]
    struct MockProvider {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("ok".into())
        }
    }

    #[derive(Default)]
    struct TrackingMemory {
        keys: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Memory for TrackingMemory {
        fn name(&self) -> &str {
            "tracking"
        }

        async fn store(
            &self,
            key: &str,
            _content: &str,
            _category: MemoryCategory,
            _session_id: Option<&str>,
        ) -> anyhow::Result<()> {
            self.keys.lock().push(key.to_string());
            Ok(())
        }

        async fn recall(
            &self,
            _query: &str,
            _limit: usize,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(Vec::new())
        }

        async fn get(&self, _key: &str) -> anyhow::Result<Option<MemoryEntry>> {
            Ok(None)
        }

        async fn list(
            &self,
            _category: Option<&MemoryCategory>,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(Vec::new())
        }

        async fn forget(&self, _key: &str) -> anyhow::Result<bool> {
            Ok(false)
        }

        async fn count(&self) -> anyhow::Result<usize> {
            let size = self.keys.lock().len();
            Ok(size)
        }

        async fn health_check(&self) -> bool {
            true
        }
    }

    fn test_connect_info() -> ConnectInfo<SocketAddr> {
        ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 30_300)))
    }

    #[tokio::test]
    async fn webhook_idempotency_skips_duplicate_provider_calls() {
        let provider_impl = Arc::new(MockProvider::default());
        let provider: Arc<dyn Provider> = provider_impl.clone();
        let memory: Arc<dyn Memory> = Arc::new(MockMemory);

        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider,
            model: "test-model".into(),
            temperature: 0.0,
            mem: memory,
            auto_save: false,
            webhook_secret_hash: None,
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let mut headers = HeaderMap::new();
        headers.insert("X-Idempotency-Key", HeaderValue::from_static("abc-123"));

        let body = Ok(Json(WebhookBody {
            message: "hello".into(),
        }));
        let first = handle_webhook(
            State(state.clone()),
            test_connect_info(),
            headers.clone(),
            body,
        )
        .await
        .into_response();
        assert_eq!(first.status(), StatusCode::OK);

        let body = Ok(Json(WebhookBody {
            message: "hello".into(),
        }));
        let second = handle_webhook(State(state), test_connect_info(), headers, body)
            .await
            .into_response();
        assert_eq!(second.status(), StatusCode::OK);

        let payload = second.into_body().collect().await.unwrap().to_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(parsed["status"], "duplicate");
        assert_eq!(parsed["idempotent"], true);
        assert_eq!(provider_impl.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn webhook_autosave_stores_distinct_keys_per_request() {
        let provider_impl = Arc::new(MockProvider::default());
        let provider: Arc<dyn Provider> = provider_impl.clone();

        let tracking_impl = Arc::new(TrackingMemory::default());
        let memory: Arc<dyn Memory> = tracking_impl.clone();

        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider,
            model: "test-model".into(),
            temperature: 0.0,
            mem: memory,
            auto_save: true,
            webhook_secret_hash: None,
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let headers = HeaderMap::new();

        let body1 = Ok(Json(WebhookBody {
            message: "hello one".into(),
        }));
        let first = handle_webhook(
            State(state.clone()),
            test_connect_info(),
            headers.clone(),
            body1,
        )
        .await
        .into_response();
        assert_eq!(first.status(), StatusCode::OK);

        let body2 = Ok(Json(WebhookBody {
            message: "hello two".into(),
        }));
        let second = handle_webhook(State(state), test_connect_info(), headers, body2)
            .await
            .into_response();
        assert_eq!(second.status(), StatusCode::OK);

        let keys = tracking_impl.keys.lock().clone();
        assert_eq!(keys.len(), 2);
        assert_ne!(keys[0], keys[1]);
        assert!(keys[0].starts_with("webhook_msg_"));
        assert!(keys[1].starts_with("webhook_msg_"));
        assert_eq!(provider_impl.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn webhook_secret_hash_is_deterministic_and_nonempty() {
        let secret_a = generate_test_secret();
        let secret_b = generate_test_secret();
        let one = hash_webhook_secret(&secret_a);
        let two = hash_webhook_secret(&secret_a);
        let other = hash_webhook_secret(&secret_b);

        assert_eq!(one, two);
        assert_ne!(one, other);
        assert_eq!(one.len(), 64);
    }

    #[tokio::test]
    async fn webhook_secret_hash_rejects_missing_header() {
        let provider_impl = Arc::new(MockProvider::default());
        let provider: Arc<dyn Provider> = provider_impl.clone();
        let memory: Arc<dyn Memory> = Arc::new(MockMemory);
        let secret = generate_test_secret();

        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider,
            model: "test-model".into(),
            temperature: 0.0,
            mem: memory,
            auto_save: false,
            webhook_secret_hash: Some(Arc::from(hash_webhook_secret(&secret))),
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let response = handle_webhook(
            State(state),
            test_connect_info(),
            HeaderMap::new(),
            Ok(Json(WebhookBody {
                message: "hello".into(),
            })),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(provider_impl.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn webhook_secret_hash_rejects_invalid_header() {
        let provider_impl = Arc::new(MockProvider::default());
        let provider: Arc<dyn Provider> = provider_impl.clone();
        let memory: Arc<dyn Memory> = Arc::new(MockMemory);
        let valid_secret = generate_test_secret();
        let wrong_secret = generate_test_secret();

        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider,
            model: "test-model".into(),
            temperature: 0.0,
            mem: memory,
            auto_save: false,
            webhook_secret_hash: Some(Arc::from(hash_webhook_secret(&valid_secret))),
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Webhook-Secret",
            HeaderValue::from_str(&wrong_secret).unwrap(),
        );

        let response = handle_webhook(
            State(state),
            test_connect_info(),
            headers,
            Ok(Json(WebhookBody {
                message: "hello".into(),
            })),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(provider_impl.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn webhook_secret_hash_accepts_valid_header() {
        let provider_impl = Arc::new(MockProvider::default());
        let provider: Arc<dyn Provider> = provider_impl.clone();
        let memory: Arc<dyn Memory> = Arc::new(MockMemory);
        let secret = generate_test_secret();

        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider,
            model: "test-model".into(),
            temperature: 0.0,
            mem: memory,
            auto_save: false,
            webhook_secret_hash: Some(Arc::from(hash_webhook_secret(&secret))),
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let mut headers = HeaderMap::new();
        headers.insert("X-Webhook-Secret", HeaderValue::from_str(&secret).unwrap());

        let response = handle_webhook(
            State(state),
            test_connect_info(),
            headers,
            Ok(Json(WebhookBody {
                message: "hello".into(),
            })),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(provider_impl.calls.load(Ordering::SeqCst), 1);
    }

    fn compute_nextcloud_signature_hex(secret: &str, random: &str, body: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let payload = format!("{random}{body}");
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    #[tokio::test]
    async fn nextcloud_talk_webhook_returns_not_found_when_not_configured() {
        let provider: Arc<dyn Provider> = Arc::new(MockProvider::default());
        let memory: Arc<dyn Memory> = Arc::new(MockMemory);

        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider,
            model: "test-model".into(),
            temperature: 0.0,
            mem: memory,
            auto_save: false,
            webhook_secret_hash: None,
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: None,
            nextcloud_talk_webhook_secret: None,
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let response = handle_nextcloud_talk_webhook(
            State(state),
            HeaderMap::new(),
            Bytes::from_static(br#"{"type":"message"}"#),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn nextcloud_talk_webhook_rejects_invalid_signature() {
        let provider_impl = Arc::new(MockProvider::default());
        let provider: Arc<dyn Provider> = provider_impl.clone();
        let memory: Arc<dyn Memory> = Arc::new(MockMemory);

        let channel = Arc::new(NextcloudTalkChannel::new(
            "https://cloud.example.com".into(),
            "app-token".into(),
            vec!["*".into()],
        ));

        let secret = "nextcloud-test-secret";
        let random = "seed-value";
        let body = r#"{"type":"message","object":{"token":"room-token"},"message":{"actorType":"users","actorId":"user_a","message":"hello"}}"#;
        let _valid_signature = compute_nextcloud_signature_hex(secret, random, body);
        let invalid_signature = "deadbeef";

        let state = AppState {
            config: Arc::new(Mutex::new(Config::default())),
            provider,
            model: "test-model".into(),
            temperature: 0.0,
            mem: memory,
            auto_save: false,
            webhook_secret_hash: None,
            pairing: Arc::new(PairingGuard::new(false, &[])),
            trust_forwarded_headers: false,
            rate_limiter: Arc::new(GatewayRateLimiter::new(100, 100, 100)),
            idempotency_store: Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000)),
            whatsapp: None,
            whatsapp_app_secret: None,
            linq: None,
            linq_signing_secret: None,
            nextcloud_talk: Some(channel),
            nextcloud_talk_webhook_secret: Some(Arc::from(secret)),
            wati: None,
            observer: Arc::new(crate::observability::NoopObserver),
            tools_registry: Arc::new(Vec::new()),
            cost_tracker: None,
            event_tx: tokio::sync::broadcast::channel(16).0,
            shutdown_tx: tokio::sync::watch::channel(false).0,
            api_chat_history: Arc::new(Mutex::new(Vec::new())),
            recent_sop_results: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            sop_engine: Arc::new(std::sync::Mutex::new(crate::sop::SopEngine::new(
                crate::config::SopConfig::default(),
            ))),
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Nextcloud-Talk-Random",
            HeaderValue::from_str(random).unwrap(),
        );
        headers.insert(
            "X-Nextcloud-Talk-Signature",
            HeaderValue::from_str(invalid_signature).unwrap(),
        );

        let response = handle_nextcloud_talk_webhook(State(state), headers, Bytes::from(body))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(provider_impl.calls.load(Ordering::SeqCst), 0);
    }

    // ══════════════════════════════════════════════════════════
    // WhatsApp Signature Verification Tests (CWE-345 Prevention)
    // ══════════════════════════════════════════════════════════

    fn compute_whatsapp_signature_hex(secret: &str, body: &[u8]) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        hex::encode(mac.finalize().into_bytes())
    }

    fn compute_whatsapp_signature_header(secret: &str, body: &[u8]) -> String {
        format!("sha256={}", compute_whatsapp_signature_hex(secret, body))
    }

    #[test]
    fn whatsapp_signature_valid() {
        let app_secret = generate_test_secret();
        let body = b"test body content";

        let signature_header = compute_whatsapp_signature_header(&app_secret, body);

        assert!(verify_whatsapp_signature(
            &app_secret,
            body,
            &signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_invalid_wrong_secret() {
        let app_secret = generate_test_secret();
        let wrong_secret = generate_test_secret();
        let body = b"test body content";

        let signature_header = compute_whatsapp_signature_header(&wrong_secret, body);

        assert!(!verify_whatsapp_signature(
            &app_secret,
            body,
            &signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_invalid_wrong_body() {
        let app_secret = generate_test_secret();
        let original_body = b"original body";
        let tampered_body = b"tampered body";

        let signature_header = compute_whatsapp_signature_header(&app_secret, original_body);

        // Verify with tampered body should fail
        assert!(!verify_whatsapp_signature(
            &app_secret,
            tampered_body,
            &signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_missing_prefix() {
        let app_secret = generate_test_secret();
        let body = b"test body";

        // Signature without "sha256=" prefix
        let signature_header = "abc123def456";

        assert!(!verify_whatsapp_signature(
            &app_secret,
            body,
            signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_empty_header() {
        let app_secret = generate_test_secret();
        let body = b"test body";

        assert!(!verify_whatsapp_signature(&app_secret, body, ""));
    }

    #[test]
    fn whatsapp_signature_invalid_hex() {
        let app_secret = generate_test_secret();
        let body = b"test body";

        // Invalid hex characters
        let signature_header = "sha256=not_valid_hex_zzz";

        assert!(!verify_whatsapp_signature(
            &app_secret,
            body,
            signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_empty_body() {
        let app_secret = generate_test_secret();
        let body = b"";

        let signature_header = compute_whatsapp_signature_header(&app_secret, body);

        assert!(verify_whatsapp_signature(
            &app_secret,
            body,
            &signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_unicode_body() {
        let app_secret = generate_test_secret();
        let body = "Hello 🦀 World".as_bytes();

        let signature_header = compute_whatsapp_signature_header(&app_secret, body);

        assert!(verify_whatsapp_signature(
            &app_secret,
            body,
            &signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_json_payload() {
        let app_secret = generate_test_secret();
        let body = br#"{"entry":[{"changes":[{"value":{"messages":[{"from":"1234567890","text":{"body":"Hello"}}]}}]}]}"#;

        let signature_header = compute_whatsapp_signature_header(&app_secret, body);

        assert!(verify_whatsapp_signature(
            &app_secret,
            body,
            &signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_case_sensitive_prefix() {
        let app_secret = generate_test_secret();
        let body = b"test body";

        let hex_sig = compute_whatsapp_signature_hex(&app_secret, body);

        // Wrong case prefix should fail
        let wrong_prefix = format!("SHA256={hex_sig}");
        assert!(!verify_whatsapp_signature(&app_secret, body, &wrong_prefix));

        // Correct prefix should pass
        let correct_prefix = format!("sha256={hex_sig}");
        assert!(verify_whatsapp_signature(
            &app_secret,
            body,
            &correct_prefix
        ));
    }

    #[test]
    fn whatsapp_signature_truncated_hex() {
        let app_secret = generate_test_secret();
        let body = b"test body";

        let hex_sig = compute_whatsapp_signature_hex(&app_secret, body);
        let truncated = &hex_sig[..32]; // Only half the signature
        let signature_header = format!("sha256={truncated}");

        assert!(!verify_whatsapp_signature(
            &app_secret,
            body,
            &signature_header
        ));
    }

    #[test]
    fn whatsapp_signature_extra_bytes() {
        let app_secret = generate_test_secret();
        let body = b"test body";

        let hex_sig = compute_whatsapp_signature_hex(&app_secret, body);
        let extended = format!("{hex_sig}deadbeef");
        let signature_header = format!("sha256={extended}");

        assert!(!verify_whatsapp_signature(
            &app_secret,
            body,
            &signature_header
        ));
    }

    // ══════════════════════════════════════════════════════════
    // IdempotencyStore Edge-Case Tests
    // ══════════════════════════════════════════════════════════

    #[test]
    fn idempotency_store_allows_different_keys() {
        let store = IdempotencyStore::new(Duration::from_secs(60), 100);
        assert!(store.record_if_new("key-a"));
        assert!(store.record_if_new("key-b"));
        assert!(store.record_if_new("key-c"));
        assert!(store.record_if_new("key-d"));
    }

    #[test]
    fn idempotency_store_max_keys_clamped_to_one() {
        let store = IdempotencyStore::new(Duration::from_secs(60), 0);
        assert!(store.record_if_new("only-key"));
        assert!(!store.record_if_new("only-key"));
    }

    #[test]
    fn idempotency_store_rapid_duplicate_rejected() {
        let store = IdempotencyStore::new(Duration::from_secs(300), 100);
        assert!(store.record_if_new("rapid"));
        assert!(!store.record_if_new("rapid"));
    }

    #[test]
    fn idempotency_store_accepts_after_ttl_expires() {
        let store = IdempotencyStore::new(Duration::from_millis(1), 100);
        assert!(store.record_if_new("ttl-key"));
        std::thread::sleep(Duration::from_millis(10));
        assert!(store.record_if_new("ttl-key"));
    }

    #[test]
    fn idempotency_store_eviction_preserves_newest() {
        let store = IdempotencyStore::new(Duration::from_secs(300), 1);
        assert!(store.record_if_new("old-key"));
        std::thread::sleep(Duration::from_millis(2));
        assert!(store.record_if_new("new-key"));

        let keys = store.keys.lock();
        assert_eq!(keys.len(), 1);
        assert!(!keys.contains_key("old-key"));
        assert!(keys.contains_key("new-key"));
    }

    #[test]
    fn rate_limiter_allows_after_window_expires() {
        let window = Duration::from_millis(50);
        let limiter = SlidingWindowRateLimiter::new(2, window, 100);
        assert!(limiter.allow("ip-1"));
        assert!(limiter.allow("ip-1"));
        assert!(!limiter.allow("ip-1")); // blocked

        // Wait for window to expire
        std::thread::sleep(Duration::from_millis(60));

        // Should be allowed again
        assert!(limiter.allow("ip-1"));
    }

    #[test]
    fn rate_limiter_independent_keys_tracked_separately() {
        let limiter = SlidingWindowRateLimiter::new(2, Duration::from_secs(60), 100);
        assert!(limiter.allow("ip-1"));
        assert!(limiter.allow("ip-1"));
        assert!(!limiter.allow("ip-1")); // ip-1 blocked

        // ip-2 should still work
        assert!(limiter.allow("ip-2"));
        assert!(limiter.allow("ip-2"));
        assert!(!limiter.allow("ip-2")); // ip-2 now blocked
    }

    #[test]
    fn rate_limiter_exact_boundary_at_max_keys() {
        let limiter = SlidingWindowRateLimiter::new(10, Duration::from_secs(60), 3);
        assert!(limiter.allow("ip-1"));
        assert!(limiter.allow("ip-2"));
        assert!(limiter.allow("ip-3"));
        // At capacity now
        assert!(limiter.allow("ip-4")); // should evict ip-1

        let guard = limiter.requests.lock();
        assert_eq!(guard.0.len(), 3);
        assert!(
            !guard.0.contains_key("ip-1"),
            "ip-1 should have been evicted"
        );
        assert!(guard.0.contains_key("ip-2"));
        assert!(guard.0.contains_key("ip-3"));
        assert!(guard.0.contains_key("ip-4"));
    }

    #[test]
    fn gateway_rate_limiter_pair_and_webhook_are_independent() {
        let limiter = GatewayRateLimiter::new(2, 3, 100);

        // Exhaust pair limit
        assert!(limiter.allow_pair("ip-1"));
        assert!(limiter.allow_pair("ip-1"));
        assert!(!limiter.allow_pair("ip-1")); // pair blocked

        // Webhook should still work
        assert!(limiter.allow_webhook("ip-1"));
        assert!(limiter.allow_webhook("ip-1"));
        assert!(limiter.allow_webhook("ip-1"));
        assert!(!limiter.allow_webhook("ip-1")); // webhook now blocked
    }

    #[test]
    fn rate_limiter_single_key_max_allows_one_request() {
        let limiter = SlidingWindowRateLimiter::new(5, Duration::from_secs(60), 1);
        assert!(limiter.allow("ip-1"));
        assert!(limiter.allow("ip-2")); // evicts ip-1

        let guard = limiter.requests.lock();
        assert_eq!(guard.0.len(), 1);
        assert!(guard.0.contains_key("ip-2"));
        assert!(!guard.0.contains_key("ip-1"));
    }

    #[test]
    fn rate_limiter_concurrent_access_safe() {
        use std::sync::Arc;

        let limiter = Arc::new(SlidingWindowRateLimiter::new(
            1000,
            Duration::from_secs(60),
            1000,
        ));
        let mut handles = Vec::new();

        for i in 0..10 {
            let limiter = limiter.clone();
            handles.push(std::thread::spawn(move || {
                for j in 0..100 {
                    limiter.allow(&format!("thread-{i}-req-{j}"));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should not panic or deadlock
        let guard = limiter.requests.lock();
        assert!(guard.0.len() <= 1000, "should respect max_keys");
    }

    #[test]
    fn idempotency_store_concurrent_access_safe() {
        use std::sync::Arc;

        let store = Arc::new(IdempotencyStore::new(Duration::from_secs(300), 1000));
        let mut handles = Vec::new();

        for i in 0..10 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for j in 0..100 {
                    store.record_if_new(&format!("thread-{i}-key-{j}"));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let keys = store.keys.lock();
        assert!(keys.len() <= 1000, "should respect max_keys");
    }

    #[test]
    fn rate_limiter_rapid_burst_then_cooldown() {
        let limiter = SlidingWindowRateLimiter::new(5, Duration::from_millis(50), 100);

        // Burst: use all 5 requests
        for _ in 0..5 {
            assert!(limiter.allow("burst-ip"));
        }
        assert!(!limiter.allow("burst-ip")); // 6th should fail

        // Cooldown
        std::thread::sleep(Duration::from_millis(60));

        // Should be allowed again
        assert!(limiter.allow("burst-ip"));
    }

    #[test]
    fn require_localhost_accepts_ipv4_loopback() {
        let peer = SocketAddr::from(([127, 0, 0, 1], 12345));
        assert!(require_localhost(&peer).is_ok());
    }

    #[test]
    fn require_localhost_accepts_ipv6_loopback() {
        let peer = SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, 12345));
        assert!(require_localhost(&peer).is_ok());
    }

    #[test]
    fn require_localhost_rejects_non_loopback_ipv4() {
        let peer = SocketAddr::from(([192, 168, 1, 100], 12345));
        let err = require_localhost(&peer).unwrap_err();
        assert_eq!(err.0, StatusCode::FORBIDDEN);
    }

    #[test]
    fn require_localhost_rejects_non_loopback_ipv6() {
        let peer = SocketAddr::from((
            std::net::Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1),
            12345,
        ));
        let err = require_localhost(&peer).unwrap_err();
        assert_eq!(err.0, StatusCode::FORBIDDEN);
    }
}

#[cfg(test)]
mod sop_result_replay_tests {
    use super::push_recent_sop_result;
    use parking_lot::Mutex;
    use std::collections::VecDeque;
    use std::sync::Arc;

    fn ev(id: &str, ts: i64) -> serde_json::Value {
        serde_json::json!({"type": "sop_result", "id": id, "timestamp": ts})
    }

    #[test]
    fn keeps_newest_and_caps_at_20() {
        let buf = Arc::new(Mutex::new(VecDeque::new()));
        let now = chrono::Utc::now().timestamp();
        for i in 0..25 {
            push_recent_sop_result(&buf, ev(&format!("r{i}"), now));
        }
        let q = buf.lock();
        assert_eq!(q.len(), 20, "buffer must stay bounded");
        // 最早的 5 条被挤掉，留下的是最新的 r5..r24
        assert_eq!(q.front().unwrap()["id"], "r5");
        assert_eq!(q.back().unwrap()["id"], "r24");
    }

    #[test]
    fn drops_entries_older_than_12h() {
        let buf = Arc::new(Mutex::new(VecDeque::new()));
        let now = chrono::Utc::now().timestamp();
        push_recent_sop_result(&buf, ev("stale", now - 13 * 3600));
        push_recent_sop_result(&buf, ev("fresh", now));
        let q = buf.lock();
        // 隔天回来不该被灌陈年汇报
        assert_eq!(q.len(), 1);
        assert_eq!(q.front().unwrap()["id"], "fresh");
    }
}
