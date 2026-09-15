//! REST API handlers for the web dashboard.
//!
//! All `/api/*` routes require bearer token authentication (PairingGuard).

use super::AppState;
use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use serde::Deserialize;

const MASKED_SECRET: &str = "***MASKED***";

// ── Bearer token auth extractor ─────────────────────────────────

/// Extract and validate bearer token from Authorization header.
fn extract_bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer "))
}

/// Verify bearer token against PairingGuard. Returns error response if unauthorized.
fn require_auth(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if !state.pairing.require_pairing() {
        return Ok(());
    }

    let token = extract_bearer_token(headers).unwrap_or("");
    if state.pairing.is_authenticated(token) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "Unauthorized — pair first via POST /pair, then send Authorization: Bearer <token>"
            })),
        ))
    }
}

// ── Query parameters ─────────────────────────────────────────────

#[derive(Deserialize)]
pub struct MemoryQuery {
    pub query: Option<String>,
    pub category: Option<String>,
}

#[derive(Deserialize)]
pub struct MemoryStoreBody {
    pub key: String,
    pub content: String,
    pub category: Option<String>,
}

#[derive(Deserialize)]
pub struct CronAddBody {
    pub name: Option<String>,
    pub schedule: String,
    pub command: String,
}

// ── Handlers ────────────────────────────────────────────────────

/// GET /api/status — system status overview
pub async fn handle_api_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let health = crate::health::snapshot();

    let mut channels = serde_json::Map::new();

    for (channel, present) in config.channels_config.channels() {
        channels.insert(channel.name().to_string(), serde_json::Value::Bool(present));
    }

    let body = serde_json::json!({
        "provider": config.default_provider,
        "model": state.model,
        "temperature": state.temperature,
        "uptime_seconds": health.uptime_seconds,
        "gateway_port": config.gateway.port,
        "locale": "en",
        "memory_backend": state.mem.name(),
        "paired": state.pairing.is_paired(),
        "channels": channels,
        "health": health,
    });

    Json(body).into_response()
}

/// GET /api/config — current config (api_key masked)
pub async fn handle_api_config_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();

    // Serialize to TOML after masking sensitive fields.
    let masked_config = mask_sensitive_fields(&config);
    let toml_str = match toml::to_string_pretty(&masked_config) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to serialize config: {e}")})),
            )
                .into_response();
        }
    };

    Json(serde_json::json!({
        "format": "toml",
        "content": toml_str,
    }))
    .into_response()
}

/// PUT /api/config — update config from TOML body
pub async fn handle_api_config_put(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    // Parse the incoming TOML
    let incoming: crate::config::Config = match toml::from_str(&body) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid TOML: {e}")})),
            )
                .into_response();
        }
    };

    let current_config = state.config.lock().clone();
    let new_config = hydrate_config_for_save(incoming, &current_config);

    if let Err(e) = new_config.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("Invalid config: {e}")})),
        )
            .into_response();
    }

    // Save to disk
    if let Err(e) = new_config.save().await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to save config: {e}")})),
        )
            .into_response();
    }

    // Update in-memory config
    *state.config.lock() = new_config;

    Json(serde_json::json!({"status": "ok"})).into_response()
}

/// GET /api/tools — list registered tool specs
pub async fn handle_api_tools(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let tools: Vec<serde_json::Value> = state
        .tools_registry
        .iter()
        .map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            })
        })
        .collect();

    Json(serde_json::json!({"tools": tools})).into_response()
}

/// GET /api/cron — list cron jobs
pub async fn handle_api_cron_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    match crate::cron::list_jobs(&config) {
        Ok(jobs) => {
            let jobs_json: Vec<serde_json::Value> = jobs
                .iter()
                .map(|job| {
                    serde_json::json!({
                        "id": job.id,
                        "name": job.name,
                        "command": job.command,
                        "next_run": job.next_run.to_rfc3339(),
                        "last_run": job.last_run.map(|t| t.to_rfc3339()),
                        "last_status": job.last_status,
                        "enabled": job.enabled,
                    })
                })
                .collect();
            Json(serde_json::json!({"jobs": jobs_json})).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to list cron jobs: {e}")})),
        )
            .into_response(),
    }
}

/// POST /api/cron — add a new cron job
pub async fn handle_api_cron_add(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<CronAddBody>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let schedule = crate::cron::Schedule::Cron {
        expr: body.schedule,
        tz: None,
    };

    match crate::cron::add_shell_job_with_approval(
        &config,
        body.name,
        schedule,
        &body.command,
        false,
    ) {
        Ok(job) => Json(serde_json::json!({
            "status": "ok",
            "job": {
                "id": job.id,
                "name": job.name,
                "command": job.command,
                "enabled": job.enabled,
            }
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to add cron job: {e}")})),
        )
            .into_response(),
    }
}

/// DELETE /api/cron/:id — remove a cron job
pub async fn handle_api_cron_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    match crate::cron::remove_job(&config, &id) {
        Ok(()) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to remove cron job: {e}")})),
        )
            .into_response(),
    }
}

/// GET /api/integrations — list all integrations with status
pub async fn handle_api_integrations(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let entries = crate::integrations::registry::all_integrations();

    let integrations: Vec<serde_json::Value> = entries
        .iter()
        .map(|entry| {
            let status = (entry.status_fn)(&config);
            serde_json::json!({
                "name": entry.name,
                "description": entry.description,
                "category": entry.category,
                "status": status,
            })
        })
        .collect();

    Json(serde_json::json!({"integrations": integrations})).into_response()
}

/// GET /api/integrations/settings — return per-integration settings (enabled + category)
pub async fn handle_api_integrations_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let entries = crate::integrations::registry::all_integrations();

    let mut settings = serde_json::Map::new();
    for entry in &entries {
        let status = (entry.status_fn)(&config);
        let enabled = matches!(status, crate::integrations::IntegrationStatus::Active);
        settings.insert(
            entry.name.to_string(),
            serde_json::json!({
                "enabled": enabled,
                "category": entry.category,
                "status": status,
            }),
        );
    }

    Json(serde_json::json!({"settings": settings})).into_response()
}

/// POST /api/doctor — run diagnostics
pub async fn handle_api_doctor(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let results = crate::doctor::diagnose(&config);

    let ok_count = results
        .iter()
        .filter(|r| r.severity == crate::doctor::Severity::Ok)
        .count();
    let warn_count = results
        .iter()
        .filter(|r| r.severity == crate::doctor::Severity::Warn)
        .count();
    let error_count = results
        .iter()
        .filter(|r| r.severity == crate::doctor::Severity::Error)
        .count();

    Json(serde_json::json!({
        "results": results,
        "summary": {
            "ok": ok_count,
            "warnings": warn_count,
            "errors": error_count,
        }
    }))
    .into_response()
}

/// GET /api/memory — list or search memory entries
pub async fn handle_api_memory_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<MemoryQuery>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    if let Some(ref query) = params.query {
        // Search mode
        match state.mem.recall(query, 50, None).await {
            Ok(entries) => Json(serde_json::json!({"entries": entries})).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Memory recall failed: {e}")})),
            )
                .into_response(),
        }
    } else {
        // List mode
        let category = params.category.as_deref().map(|cat| match cat {
            "core" => crate::memory::MemoryCategory::Core,
            "daily" => crate::memory::MemoryCategory::Daily,
            "conversation" => crate::memory::MemoryCategory::Conversation,
            other => crate::memory::MemoryCategory::Custom(other.to_string()),
        });

        match state.mem.list(category.as_ref(), None).await {
            Ok(entries) => Json(serde_json::json!({"entries": entries})).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Memory list failed: {e}")})),
            )
                .into_response(),
        }
    }
}

/// POST /api/memory — store a memory entry
pub async fn handle_api_memory_store(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<MemoryStoreBody>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let category = body
        .category
        .as_deref()
        .map(|cat| match cat {
            "core" => crate::memory::MemoryCategory::Core,
            "daily" => crate::memory::MemoryCategory::Daily,
            "conversation" => crate::memory::MemoryCategory::Conversation,
            other => crate::memory::MemoryCategory::Custom(other.to_string()),
        })
        .unwrap_or(crate::memory::MemoryCategory::Core);

    match state
        .mem
        .store(&body.key, &body.content, category, None)
        .await
    {
        Ok(()) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Memory store failed: {e}")})),
        )
            .into_response(),
    }
}

/// DELETE /api/memory/:key — delete a memory entry
pub async fn handle_api_memory_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    match state.mem.forget(&key).await {
        Ok(deleted) => {
            Json(serde_json::json!({"status": "ok", "deleted": deleted})).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Memory forget failed: {e}")})),
        )
            .into_response(),
    }
}

/// GET /api/cost — cost summary
pub async fn handle_api_cost(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let (daily_limit, monthly_limit) = {
        let c = state.config.lock();
        (c.cost.daily_limit_usd, c.cost.monthly_limit_usd)
    };
    if let Some(ref tracker) = state.cost_tracker {
        match tracker.get_summary() {
            Ok(summary) => Json(serde_json::json!({
                "tracking": true,
                "cost": summary,
                "limits": { "daily": daily_limit, "monthly": monthly_limit },
            }))
            .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Cost summary failed: {e}")})),
            )
                .into_response(),
        }
    } else {
        // 没开费用统计：下面的零是占位值，不是「花了 0」。显式标出来，
        // 否则前台会把「没在记账」展示成「本月用量 0.00」
        Json(serde_json::json!({
            "tracking": false,
            "cost": {
                "session_cost_usd": 0.0,
                "daily_cost_usd": 0.0,
                "monthly_cost_usd": 0.0,
                "total_tokens": 0,
                "request_count": 0,
                "by_model": {},
            }
        }))
        .into_response()
    }
}

/// GET /api/cli-tools — discovered CLI tools
pub async fn handle_api_cli_tools(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let tools = crate::tools::cli_discovery::discover_cli_tools(&[], &[]);

    Json(serde_json::json!({"cli_tools": tools})).into_response()
}

/// GET /api/health — component health snapshot
pub async fn handle_api_health(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let snapshot = crate::health::snapshot();
    Json(serde_json::json!({"health": snapshot})).into_response()
}

// ── Helpers ─────────────────────────────────────────────────────

fn is_masked_secret(value: &str) -> bool {
    value == MASKED_SECRET
}

fn mask_optional_secret(value: &mut Option<String>) {
    if value.is_some() {
        *value = Some(MASKED_SECRET.to_string());
    }
}

fn mask_required_secret(value: &mut String) {
    if !value.is_empty() {
        *value = MASKED_SECRET.to_string();
    }
}

fn mask_vec_secrets(values: &mut [String]) {
    for value in values.iter_mut() {
        if !value.is_empty() {
            *value = MASKED_SECRET.to_string();
        }
    }
}

#[allow(clippy::ref_option)]
fn restore_optional_secret(value: &mut Option<String>, current: &Option<String>) {
    if value.as_deref().is_some_and(is_masked_secret) {
        *value = current.clone();
    }
}

fn restore_required_secret(value: &mut String, current: &str) {
    if is_masked_secret(value) {
        *value = current.to_string();
    }
}

fn restore_vec_secrets(values: &mut [String], current: &[String]) {
    for (idx, value) in values.iter_mut().enumerate() {
        if is_masked_secret(value) {
            if let Some(existing) = current.get(idx) {
                *value = existing.clone();
            }
        }
    }
}

fn normalize_route_field(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn model_route_identity_matches(
    incoming: &crate::config::schema::ModelRouteConfig,
    current: &crate::config::schema::ModelRouteConfig,
) -> bool {
    normalize_route_field(&incoming.hint) == normalize_route_field(&current.hint)
        && normalize_route_field(&incoming.provider) == normalize_route_field(&current.provider)
        && normalize_route_field(&incoming.model) == normalize_route_field(&current.model)
}

fn model_route_provider_model_matches(
    incoming: &crate::config::schema::ModelRouteConfig,
    current: &crate::config::schema::ModelRouteConfig,
) -> bool {
    normalize_route_field(&incoming.provider) == normalize_route_field(&current.provider)
        && normalize_route_field(&incoming.model) == normalize_route_field(&current.model)
}

fn embedding_route_identity_matches(
    incoming: &crate::config::schema::EmbeddingRouteConfig,
    current: &crate::config::schema::EmbeddingRouteConfig,
) -> bool {
    normalize_route_field(&incoming.hint) == normalize_route_field(&current.hint)
        && normalize_route_field(&incoming.provider) == normalize_route_field(&current.provider)
        && normalize_route_field(&incoming.model) == normalize_route_field(&current.model)
}

fn embedding_route_provider_model_matches(
    incoming: &crate::config::schema::EmbeddingRouteConfig,
    current: &crate::config::schema::EmbeddingRouteConfig,
) -> bool {
    normalize_route_field(&incoming.provider) == normalize_route_field(&current.provider)
        && normalize_route_field(&incoming.model) == normalize_route_field(&current.model)
}

fn restore_model_route_api_keys(
    incoming: &mut [crate::config::schema::ModelRouteConfig],
    current: &[crate::config::schema::ModelRouteConfig],
) {
    let mut used_current = vec![false; current.len()];
    for incoming_route in incoming {
        if !incoming_route
            .api_key
            .as_deref()
            .is_some_and(is_masked_secret)
        {
            continue;
        }

        let exact_match_idx = current
            .iter()
            .enumerate()
            .find(|(idx, current_route)| {
                !used_current[*idx] && model_route_identity_matches(incoming_route, current_route)
            })
            .map(|(idx, _)| idx);

        let match_idx = exact_match_idx.or_else(|| {
            current
                .iter()
                .enumerate()
                .find(|(idx, current_route)| {
                    !used_current[*idx]
                        && model_route_provider_model_matches(incoming_route, current_route)
                })
                .map(|(idx, _)| idx)
        });

        if let Some(idx) = match_idx {
            used_current[idx] = true;
            incoming_route.api_key = current[idx].api_key.clone();
        } else {
            // Never persist UI placeholders to disk when no safe restore target exists.
            incoming_route.api_key = None;
        }
    }
}

fn restore_embedding_route_api_keys(
    incoming: &mut [crate::config::schema::EmbeddingRouteConfig],
    current: &[crate::config::schema::EmbeddingRouteConfig],
) {
    let mut used_current = vec![false; current.len()];
    for incoming_route in incoming {
        if !incoming_route
            .api_key
            .as_deref()
            .is_some_and(is_masked_secret)
        {
            continue;
        }

        let exact_match_idx = current
            .iter()
            .enumerate()
            .find(|(idx, current_route)| {
                !used_current[*idx]
                    && embedding_route_identity_matches(incoming_route, current_route)
            })
            .map(|(idx, _)| idx);

        let match_idx = exact_match_idx.or_else(|| {
            current
                .iter()
                .enumerate()
                .find(|(idx, current_route)| {
                    !used_current[*idx]
                        && embedding_route_provider_model_matches(incoming_route, current_route)
                })
                .map(|(idx, _)| idx)
        });

        if let Some(idx) = match_idx {
            used_current[idx] = true;
            incoming_route.api_key = current[idx].api_key.clone();
        } else {
            // Never persist UI placeholders to disk when no safe restore target exists.
            incoming_route.api_key = None;
        }
    }
}

fn mask_sensitive_fields(config: &crate::config::Config) -> crate::config::Config {
    let mut masked = config.clone();

    mask_optional_secret(&mut masked.api_key);
    mask_vec_secrets(&mut masked.reliability.api_keys);
    mask_vec_secrets(&mut masked.gateway.paired_tokens);
    mask_optional_secret(&mut masked.composio.api_key);
    mask_optional_secret(&mut masked.browser.computer_use.api_key);
    mask_optional_secret(&mut masked.web_search.brave_api_key);
    mask_optional_secret(&mut masked.storage.provider.config.db_url);
    mask_optional_secret(&mut masked.memory.qdrant.api_key);
    if let Some(cloudflare) = masked.tunnel.cloudflare.as_mut() {
        mask_required_secret(&mut cloudflare.token);
    }
    if let Some(ngrok) = masked.tunnel.ngrok.as_mut() {
        mask_required_secret(&mut ngrok.auth_token);
    }

    for agent in masked.agents.values_mut() {
        mask_optional_secret(&mut agent.api_key);
    }
    for route in &mut masked.model_routes {
        mask_optional_secret(&mut route.api_key);
    }
    for route in &mut masked.embedding_routes {
        mask_optional_secret(&mut route.api_key);
    }

    if let Some(telegram) = masked.channels_config.telegram.as_mut() {
        mask_required_secret(&mut telegram.bot_token);
    }
    if let Some(discord) = masked.channels_config.discord.as_mut() {
        mask_required_secret(&mut discord.bot_token);
    }
    if let Some(slack) = masked.channels_config.slack.as_mut() {
        mask_required_secret(&mut slack.bot_token);
        mask_optional_secret(&mut slack.app_token);
    }
    if let Some(mattermost) = masked.channels_config.mattermost.as_mut() {
        mask_required_secret(&mut mattermost.bot_token);
    }
    if let Some(webhook) = masked.channels_config.webhook.as_mut() {
        mask_optional_secret(&mut webhook.secret);
    }
    if let Some(matrix) = masked.channels_config.matrix.as_mut() {
        mask_required_secret(&mut matrix.access_token);
    }
    if let Some(whatsapp) = masked.channels_config.whatsapp.as_mut() {
        mask_optional_secret(&mut whatsapp.access_token);
        mask_optional_secret(&mut whatsapp.app_secret);
        mask_optional_secret(&mut whatsapp.verify_token);
    }
    if let Some(linq) = masked.channels_config.linq.as_mut() {
        mask_required_secret(&mut linq.api_token);
        mask_optional_secret(&mut linq.signing_secret);
    }
    if let Some(nextcloud) = masked.channels_config.nextcloud_talk.as_mut() {
        mask_required_secret(&mut nextcloud.app_token);
        mask_optional_secret(&mut nextcloud.webhook_secret);
    }
    if let Some(wati) = masked.channels_config.wati.as_mut() {
        mask_required_secret(&mut wati.api_token);
    }
    if let Some(irc) = masked.channels_config.irc.as_mut() {
        mask_optional_secret(&mut irc.server_password);
        mask_optional_secret(&mut irc.nickserv_password);
        mask_optional_secret(&mut irc.sasl_password);
    }
    if let Some(lark) = masked.channels_config.lark.as_mut() {
        mask_required_secret(&mut lark.app_secret);
        mask_optional_secret(&mut lark.encrypt_key);
        mask_optional_secret(&mut lark.verification_token);
    }
    if let Some(feishu) = masked.channels_config.feishu.as_mut() {
        mask_required_secret(&mut feishu.app_secret);
        mask_optional_secret(&mut feishu.encrypt_key);
        mask_optional_secret(&mut feishu.verification_token);
    }
    if let Some(dingtalk) = masked.channels_config.dingtalk.as_mut() {
        mask_required_secret(&mut dingtalk.client_secret);
    }
    if let Some(qq) = masked.channels_config.qq.as_mut() {
        mask_required_secret(&mut qq.app_secret);
    }
    #[cfg(feature = "channel-nostr")]
    if let Some(nostr) = masked.channels_config.nostr.as_mut() {
        mask_required_secret(&mut nostr.private_key);
    }
    if let Some(clawdtalk) = masked.channels_config.clawdtalk.as_mut() {
        mask_required_secret(&mut clawdtalk.api_key);
        mask_optional_secret(&mut clawdtalk.webhook_secret);
    }
    if let Some(email) = masked.channels_config.email.as_mut() {
        mask_required_secret(&mut email.password);
    }
    masked
}

fn restore_masked_sensitive_fields(
    incoming: &mut crate::config::Config,
    current: &crate::config::Config,
) {
    restore_optional_secret(&mut incoming.api_key, &current.api_key);
    restore_vec_secrets(
        &mut incoming.gateway.paired_tokens,
        &current.gateway.paired_tokens,
    );
    restore_vec_secrets(
        &mut incoming.reliability.api_keys,
        &current.reliability.api_keys,
    );
    restore_optional_secret(&mut incoming.composio.api_key, &current.composio.api_key);
    restore_optional_secret(
        &mut incoming.browser.computer_use.api_key,
        &current.browser.computer_use.api_key,
    );
    restore_optional_secret(
        &mut incoming.web_search.brave_api_key,
        &current.web_search.brave_api_key,
    );
    restore_optional_secret(
        &mut incoming.storage.provider.config.db_url,
        &current.storage.provider.config.db_url,
    );
    restore_optional_secret(
        &mut incoming.memory.qdrant.api_key,
        &current.memory.qdrant.api_key,
    );
    if let (Some(incoming_tunnel), Some(current_tunnel)) = (
        incoming.tunnel.cloudflare.as_mut(),
        current.tunnel.cloudflare.as_ref(),
    ) {
        restore_required_secret(&mut incoming_tunnel.token, &current_tunnel.token);
    }
    if let (Some(incoming_tunnel), Some(current_tunnel)) = (
        incoming.tunnel.ngrok.as_mut(),
        current.tunnel.ngrok.as_ref(),
    ) {
        restore_required_secret(&mut incoming_tunnel.auth_token, &current_tunnel.auth_token);
    }

    for (name, agent) in &mut incoming.agents {
        if let Some(current_agent) = current.agents.get(name) {
            restore_optional_secret(&mut agent.api_key, &current_agent.api_key);
        }
    }
    restore_model_route_api_keys(&mut incoming.model_routes, &current.model_routes);
    restore_embedding_route_api_keys(&mut incoming.embedding_routes, &current.embedding_routes);

    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.telegram.as_mut(),
        current.channels_config.telegram.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.bot_token, &current_ch.bot_token);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.discord.as_mut(),
        current.channels_config.discord.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.bot_token, &current_ch.bot_token);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.slack.as_mut(),
        current.channels_config.slack.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.bot_token, &current_ch.bot_token);
        restore_optional_secret(&mut incoming_ch.app_token, &current_ch.app_token);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.mattermost.as_mut(),
        current.channels_config.mattermost.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.bot_token, &current_ch.bot_token);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.webhook.as_mut(),
        current.channels_config.webhook.as_ref(),
    ) {
        restore_optional_secret(&mut incoming_ch.secret, &current_ch.secret);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.matrix.as_mut(),
        current.channels_config.matrix.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.access_token, &current_ch.access_token);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.whatsapp.as_mut(),
        current.channels_config.whatsapp.as_ref(),
    ) {
        restore_optional_secret(&mut incoming_ch.access_token, &current_ch.access_token);
        restore_optional_secret(&mut incoming_ch.app_secret, &current_ch.app_secret);
        restore_optional_secret(&mut incoming_ch.verify_token, &current_ch.verify_token);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.linq.as_mut(),
        current.channels_config.linq.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.api_token, &current_ch.api_token);
        restore_optional_secret(&mut incoming_ch.signing_secret, &current_ch.signing_secret);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.nextcloud_talk.as_mut(),
        current.channels_config.nextcloud_talk.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.app_token, &current_ch.app_token);
        restore_optional_secret(&mut incoming_ch.webhook_secret, &current_ch.webhook_secret);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.wati.as_mut(),
        current.channels_config.wati.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.api_token, &current_ch.api_token);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.irc.as_mut(),
        current.channels_config.irc.as_ref(),
    ) {
        restore_optional_secret(
            &mut incoming_ch.server_password,
            &current_ch.server_password,
        );
        restore_optional_secret(
            &mut incoming_ch.nickserv_password,
            &current_ch.nickserv_password,
        );
        restore_optional_secret(&mut incoming_ch.sasl_password, &current_ch.sasl_password);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.lark.as_mut(),
        current.channels_config.lark.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.app_secret, &current_ch.app_secret);
        restore_optional_secret(&mut incoming_ch.encrypt_key, &current_ch.encrypt_key);
        restore_optional_secret(
            &mut incoming_ch.verification_token,
            &current_ch.verification_token,
        );
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.feishu.as_mut(),
        current.channels_config.feishu.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.app_secret, &current_ch.app_secret);
        restore_optional_secret(&mut incoming_ch.encrypt_key, &current_ch.encrypt_key);
        restore_optional_secret(
            &mut incoming_ch.verification_token,
            &current_ch.verification_token,
        );
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.dingtalk.as_mut(),
        current.channels_config.dingtalk.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.client_secret, &current_ch.client_secret);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.qq.as_mut(),
        current.channels_config.qq.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.app_secret, &current_ch.app_secret);
    }
    #[cfg(feature = "channel-nostr")]
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.nostr.as_mut(),
        current.channels_config.nostr.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.private_key, &current_ch.private_key);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.clawdtalk.as_mut(),
        current.channels_config.clawdtalk.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.api_key, &current_ch.api_key);
        restore_optional_secret(&mut incoming_ch.webhook_secret, &current_ch.webhook_secret);
    }
    if let (Some(incoming_ch), Some(current_ch)) = (
        incoming.channels_config.email.as_mut(),
        current.channels_config.email.as_ref(),
    ) {
        restore_required_secret(&mut incoming_ch.password, &current_ch.password);
    }
}

fn hydrate_config_for_save(
    mut incoming: crate::config::Config,
    current: &crate::config::Config,
) -> crate::config::Config {
    restore_masked_sensitive_fields(&mut incoming, current);
    // These are runtime-computed fields skipped from TOML serialization.
    incoming.config_path = current.config_path.clone();
    incoming.workspace_dir = current.workspace_dir.clone();
    incoming
}

/// 步骤条标签：取标题前 4 个字符；若因此切在半个 ASCII 词里，就把结尾这段 ASCII 去掉。
fn short_label(title: &str) -> String {
    let head: String = title.chars().take(4).collect();
    let full: Vec<char> = title.chars().collect();
    let cut_mid_word = full.len() > 4
        && head
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_alphanumeric())
        && full[4].is_ascii_alphanumeric();
    if cut_mid_word {
        let trimmed: String = head
            .trim_end_matches(|c: char| c.is_ascii_alphanumeric())
            .trim()
            .to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
    head.trim().to_string()
}

/// GET /api/capabilities — 本助手能执行什么（规程 + 技能），供前台「新任务」与「助手概况」用。
///
/// 响应带 `{"api":"frontdesk","api_version":1}` 标记：gateway 对未匹配的 GET 走 SPA fallback
/// 返回 200 + HTML，客户端只看状态码会把「旧版本不支持」误判成「支持但清单为空」。
/// 有了这个标记，客户端可以确定性地判断降级。
/// SOP 的人读标题：SOP.md 第一个一级标题，去掉结尾的「SOP」。取不到返回 None。
fn sop_title(sop: &crate::sop::types::Sop) -> Option<String> {
    let md = std::fs::read_to_string(sop.location.as_ref()?.join("SOP.md")).ok()?;
    heading_title(&md)
}

fn heading_title(md: &str) -> Option<String> {
    let line = md.lines().find(|l| l.starts_with("# "))?;
    let t = line.trim_start_matches("# ").trim();
    let t = t.strip_suffix("SOP").unwrap_or(t).trim();
    (!t.is_empty()).then(|| t.to_string())
}

pub async fn handle_api_capabilities(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let sops: Vec<serde_json::Value> = {
        let engine = state.sop_engine.lock().unwrap();
        engine
            .sops()
            .iter()
            .map(|sop| {
                let steps: Vec<serde_json::Value> = sop
                    .steps
                    .iter()
                    .map(|st| {
                        serde_json::json!({
                            "num": st.number,
                            "title": st.title,
                            // 步骤条标签：标题前 4 字，但不切在半个英文词上
                            // （「检索 PubMed 指南」截成「检索 P」很难看且无意义）
                            "short": short_label(&st.title),
                            "human": st.requires_confirmation,
                        })
                    })
                    .collect();
                let gate_steps: Vec<u32> = sop
                    .steps
                    .iter()
                    .filter(|st| st.requires_confirmation)
                    .map(|st| st.number)
                    .collect();
                serde_json::json!({
                    "name": sop.name,
                    // 人读的名字取 SOP.md 的一级标题（「# 临床病例解读报告 SOP」→「临床病例解读报告」）；
                    // 以前直接给 name，前台满屏都是 case-clinical-report
                    "title": sop_title(sop).unwrap_or_else(|| sop.name.clone()),
                    "description": sop.description,
                    "version": sop.version,
                    "steps": steps,
                    "gate_steps": gate_steps,
                })
            })
            .collect()
    };

    // 技能没有常驻注册表，按需从工作区读（与 agent 启动时同一来源）
    let skills: Vec<serde_json::Value> = {
        let cfg = state.config.lock(); // parking_lot：无需 unwrap
        crate::skills::load_skills_with_config(&cfg.workspace_dir, &cfg)
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "summary": s.description,
                    "version": s.version,
                })
            })
            .collect()
    };

    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "sops": sops,
        "skills": skills,
    }))
    .into_response()
}

// ── P1a：任务档案与证据台账 ────────────────────────────────────────

/// 解析当前 trace 配置：证据台账建立在它之上。
fn trace_setup(state: &AppState) -> (std::path::PathBuf, bool) {
    let cfg = state.config.lock();
    let mode = crate::observability::runtime_trace::storage_mode_from_config(&cfg.observability);
    let path = crate::observability::runtime_trace::resolve_trace_path(
        &cfg.observability,
        &cfg.workspace_dir,
    );
    (
        path,
        mode == crate::observability::runtime_trace::RuntimeTraceStorageMode::Full,
    )
}

/// GET /api/tasks — 任务档案列表（每条带证据条数，供前台核对"这一单有没有依据"）。
///
/// 与 /api/sop/runs 的区别：这里以「任务」为单位，附带证据统计与台账可用性，
/// 并且把已结束的 run 一并返回——「从活跃列表消失」不等于成功，客户端不该靠推断。
pub async fn handle_api_tasks(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let (trace_path, full) = trace_setup(&state);
    let avail = crate::ledger::availability(&trace_path, full);

    let runs: Vec<serde_json::Value> = {
        let engine = state.sop_engine.lock().unwrap();
        let mut all: Vec<&crate::sop::types::SopRun> = engine.active_runs().values().collect();
        all.extend(engine.finished_runs(None));
        all.into_iter()
            .map(|r| {
                let turn = if avail.available {
                    crate::ledger::turn_for_run(&trace_path, &r.run_id)
                } else {
                    None
                };
                let evidence_count = if avail.available {
                    crate::ledger::evidence_for_run(&trace_path, &r.run_id)
                        .iter()
                        .filter(|e| {
                            e.source_class == crate::ledger::SourceClass::External && e.success
                        })
                        .count()
                } else {
                    0
                };
                serde_json::json!({
                    "task_id": r.run_id,
                    "run_id": r.run_id,
                    "sop_name": r.sop_name,
                    "sop_version": engine.get_sop(&r.sop_name).map(|s| s.version.clone()),
                    "status": r.status,
                    "current_step": r.current_step,
                    "total_steps": r.total_steps,
                    "started_at": r.started_at,
                    "waiting_since": r.waiting_since,
                    "completed_at": r.completed_at,
                    "trigger_payload": r.trigger_event.payload.as_ref().map(|p| p.chars().take(1500).collect::<String>()),
                    "turn_id": turn,
                    "external_evidence_count": evidence_count,
                })
            })
            .collect()
    };

    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "ledger": avail,
        "tasks": runs,
    }))
    .into_response()
}

/// GET /api/tasks/{run_id} — 单个任务档案：run 状态 + 本轮全部工具调用（证据）。
pub async fn handle_api_task_detail(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let (trace_path, full) = trace_setup(&state);
    let avail = crate::ledger::availability(&trace_path, full);

    let run_json = {
        let engine = state.sop_engine.lock().unwrap();
        let Some(r) = engine.get_run(&run_id) else {
            drop(engine);
            // 引擎忘了这一单（助手重启过，执行状态只在内存）——但调用记录是落盘的。
            // 记录里还有它，就照样给出依据与产物，只是没有执行状态；连记录都没有才算「不在了」。
            let turns = if avail.available {
                crate::ledger::turns_for_run(&trace_path, &run_id)
            } else {
                Vec::new()
            };
            if turns.is_empty() {
                return (
                    StatusCode::NOT_FOUND,
                    // 带上 frontdesk 标记：客户端要能把「端点存在、这条 run 已不在」
                    // 和「这台 daemon 根本没有该端点（旧版本）」区分开——否则界面只能
                    // 笼统说「服务端版本较旧」，把原因说错。
                    Json(serde_json::json!({
                        "api": "frontdesk",
                        "api_version": 1,
                        "error": "run not found",
                        "run_id": run_id,
                    })),
                )
                    .into_response();
            }
            let with_output = crate::ledger::evidence_with_output_for_run(&trace_path, &run_id);
            let evidence: Vec<crate::ledger::Evidence> =
                with_output.iter().map(|(e, _)| e.clone()).collect();
            let workspace = state.config.lock().workspace_dir.clone();
            let artifacts: Vec<serde_json::Value> = task_artifacts(&evidence, &workspace)
                .into_iter()
                .map(|a| serde_json::json!({"id": a.id, "path": a.rel, "bytes": a.bytes}))
                .collect();
            return Json(serde_json::json!({
                "api": "frontdesk",
                "api_version": 1,
                "ledger": avail,
                "task": serde_json::Value::Null,
                "state_lost": true,
                "reports": reports_for_run(&state, &run_id),
                "turn_id": turns.first(),
                "turn_ids": turns,
                "evidence": evidence,
                "artifacts": artifacts,
                "casebook": task_casebook(&state, &run_id, &with_output),
                "evidence_note": "服务端已不保留这一单的执行状态（助手重启过），以上工具调用与产物取自落盘的调用记录。只有向外部取数据、且调用成功的，才算依据；读本地文件的可能读的是旧缓存，助手自己写的内容不算依据。",
            }))
            .into_response();
        };
        let step_info = engine
            .get_sop(&r.sop_name)
            .and_then(|sop| sop.steps.iter().find(|st| st.number == r.current_step))
            .map(|st| serde_json::json!({"title": st.title, "body": st.body}));
        serde_json::json!({
            "task_id": r.run_id,
            "run_id": r.run_id,
            "sop_name": r.sop_name,
            "sop_version": engine.get_sop(&r.sop_name).map(|s| s.version.clone()),
            "status": r.status,
            "current_step": r.current_step,
            "total_steps": r.total_steps,
            "current_step_info": step_info,
            "started_at": r.started_at,
            "waiting_since": r.waiting_since,
            "completed_at": r.completed_at,
            "trigger_payload": r.trigger_event.payload.clone(),
            "step_results": r.step_results.iter().map(|sr| serde_json::json!({
                "step": sr.step_number,
                "output": sr.output.chars().take(2000).collect::<String>(),
            })).collect::<Vec<_>>(),
        })
    };

    let turns = if avail.available {
        crate::ledger::turns_for_run(&trace_path, &run_id)
    } else {
        Vec::new()
    };
    let with_output = if avail.available {
        crate::ledger::evidence_with_output_for_run(&trace_path, &run_id)
    } else {
        Vec::new()
    };
    let evidence: Vec<crate::ledger::Evidence> =
        with_output.iter().map(|(e, _)| e.clone()).collect();
    let workspace = state.config.lock().workspace_dir.clone();
    let artifacts: Vec<serde_json::Value> = task_artifacts(&evidence, &workspace)
        .into_iter()
        .map(|a| serde_json::json!({"id": a.id, "path": a.rel, "bytes": a.bytes}))
        .collect();

    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "ledger": avail,
        "task": run_json,
        "turn_id": turns.first(),
        "turn_ids": turns,
        "evidence": evidence,
        "casebook": task_casebook(&state, &run_id, &with_output),
        // 这一单写出的文件（取自工具调用记录，且此刻仍在工作区里）。内容只走下面的 Bearer 端点
        "artifacts": artifacts,
        "reports": reports_for_run(&state, &run_id),
        // 说清楚这批证据是什么、不是什么——界面直接引用这句，不要另行编写
        "evidence_note": "以上是这项任务的各轮对话里记录到的工具调用。只有向外部取数据、且调用成功的，才算依据；读本地文件的可能读的是旧缓存，助手自己写的内容不算依据。没有被记录下来的调用不在此列。",
    }))
    .into_response()
}

/// 一单的可核对信息：引用核验、结构化结论（逐条核验）、人工决定记录、逐条认可 / 不认同。
/// 全部由运行时从留存记录算出；模型写的任何「已核」字样都不参与。
pub(crate) fn task_casebook(
    state: &AppState,
    run_id: &str,
    evidence: &[(crate::ledger::Evidence, String)],
) -> serde_json::Value {
    use super::casebook as cb;
    let workspace = state.config.lock().workspace_dir.clone();
    let plain: Vec<crate::ledger::Evidence> = evidence.iter().map(|(e, _)| e.clone()).collect();
    let arts = task_artifacts(&plain, &workspace);
    let paths: Vec<(String, std::path::PathBuf)> = arts
        .iter()
        .map(|a| (a.rel.clone(), a.abs.clone()))
        .collect();

    // 引用：取正式报告（没有就取草稿）；都没有，就取最后一条汇报
    let (report_rel, report_text) = match cb::pick_report(&paths) {
        Some((rel, abs)) => (
            Some(rel.clone()),
            std::fs::read_to_string(abs).unwrap_or_default(),
        ),
        None => (
            None,
            reports_for_run(state, run_id)
                .last()
                .and_then(|r| {
                    r.get("response")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_default(),
        ),
    };
    let citations = cb::verify_citations(&cb::extract_pmids(&report_text), evidence);
    let verified = citations.iter().filter(|c| c.verified).count();

    // 结构化结论：规程产出的 conclusion.json（同一单写的，且在工作区里）
    let conclusion = paths
        .iter()
        .rev()
        .find(|(rel, _)| rel.ends_with("conclusion.json"))
        .map(|(rel, abs)| {
            match std::fs::read_to_string(abs)
                .map_err(|e| e.to_string())
                .and_then(|raw| {
                    serde_json::from_str::<serde_json::Value>(&raw).map_err(|e| e.to_string())
                }) {
                Ok(v) => {
                    serde_json::json!({"path": rel, "data": cb::annotate_conclusion(v, evidence)})
                }
                Err(e) => {
                    serde_json::json!({"path": rel, "error": format!("结构化结论读不出来：{e}")})
                }
            }
        });

    serde_json::json!({
        "report": report_rel,
        "citations": citations,
        "citation_summary": {"total": citations.len(), "verified": verified},
        "conclusion": conclusion,
        "decisions": cb::read_jsonl_for_run(&cb::decisions_path(&workspace), run_id),
        "verdicts": cb::read_jsonl_for_run(&cb::verdicts_path(&workspace), run_id),
    })
}

/// 人工决定（批准 / 驳回 / 停止）留痕：谁、何时、哪一步，以及**做决定时**可核对的状况。
pub(crate) fn record_decision(
    state: &AppState,
    headers: &HeaderMap,
    run_id: &str,
    step: Option<u32>,
    decision: &str,
    who: Option<&str>,
    reason: Option<&str>,
) {
    use super::casebook as cb;
    let (trace_path, full) = trace_setup(state);
    let evidence = if crate::ledger::availability(&trace_path, full).available {
        crate::ledger::evidence_with_output_for_run(&trace_path, run_id)
    } else {
        Vec::new()
    };
    let book = task_casebook(state, run_id, &evidence);
    let workspace = state.config.lock().workspace_dir.clone();
    let report_sha = book
        .get("report")
        .and_then(|r| r.as_str())
        .and_then(|rel| std::fs::read(workspace.join(rel)).ok())
        .map(|bytes| {
            use sha2::Digest;
            format!("{:x}", sha2::Sha256::digest(&bytes))
        });
    let device = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|a| a.strip_prefix("Bearer "))
        .and_then(cb::device_fingerprint);
    let entry = serde_json::json!({
        "run_id": run_id,
        "at": chrono::Utc::now().to_rfc3339(),
        "decision": decision,
        "step": step,
        "who": who.map(str::trim).filter(|w| !w.is_empty()),
        "device": device,
        "reason": reason,
        "basis": {
            "citations_total": book["citation_summary"]["total"],
            "citations_verified": book["citation_summary"]["verified"],
            "external_calls": evidence.iter().filter(|(e, _)| e.source_class == crate::ledger::SourceClass::External && e.success).count(),
            "report": book.get("report"),
            "report_sha256": report_sha,
            "binding_violations": book.pointer("/conclusion/data/checks/binding_violations"),
        },
    });
    if let Err(e) = cb::append_jsonl(&cb::decisions_path(&workspace), &entry) {
        tracing::error!(run_id = %run_id, "decision record write failed: {e}");
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct VerdictBody {
    /// agree | disagree | answer
    pub kind: String,
    /// 针对哪一条建议（agree / disagree）
    #[serde(default)]
    pub rec_id: Option<String>,
    /// 针对哪一个问题（answer）
    #[serde(default)]
    pub qid: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub who: Option<String>,
}

/// POST /api/tasks/{run_id}/verdicts —— 快环：逐条认可 / 不认同，或回答「助手不确定的」问题。
/// 只记录，不改结论、不改规程——它是评测样本与修订建议的原料。
pub async fn handle_api_task_verdict(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    body: Option<Json<VerdictBody>>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let Some(Json(b)) = body else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"api":"frontdesk","error":"缺少请求体"})),
        )
            .into_response();
    };
    let kind = b.kind.trim();
    let text = b
        .text
        .as_deref()
        .map(|t| redact_feedback(t.trim()))
        .filter(|t| !t.is_empty());
    let valid = match kind {
        "agree" | "retract" => b.rec_id.is_some(),
        "disagree" => b.rec_id.is_some() && text.is_some(),
        "answer" => b.qid.is_some() && text.is_some(),
        _ => false,
    };
    if !valid {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"api":"frontdesk","error":"需要 kind=agree 带 rec_id；disagree 带 rec_id 和一句理由；answer 带 qid 和回答"})),
        )
            .into_response();
    }
    let workspace = state.config.lock().workspace_dir.clone();
    let path = super::casebook::verdicts_path(&workspace);
    let text: Option<String> = text.map(|t| t.chars().take(1000).collect());
    // 幂等：同一条（同一建议 / 同一问题）的最新记录与这次完全一样，就不再追加——
    // 实测按钮多点几下，记录里就出现三条「认可 R1」、六条同样的回答。改了主意（换答案、认可改不认同）照常记。
    let existing = super::casebook::read_jsonl_for_run(&path, &run_id);
    if let Some(same) = super::casebook::latest_verdict_if_same(
        &existing,
        kind,
        b.rec_id.as_deref(),
        b.qid.as_deref(),
        text.as_deref(),
    ) {
        return Json(serde_json::json!({"api":"frontdesk","api_version":1,"recorded":false,"duplicate":true,"entry":same})).into_response();
    }
    let entry = serde_json::json!({
        "run_id": run_id,
        "at": chrono::Utc::now().to_rfc3339(),
        "kind": kind,
        "rec_id": b.rec_id,
        "qid": b.qid,
        "text": text,
        "who": b.who.as_deref().map(str::trim).filter(|w| !w.is_empty()),
    });
    if let Err(e) = super::casebook::append_jsonl(&path, &entry) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"api":"frontdesk","error":format!("记不下来：{e}")})),
        )
            .into_response();
    }
    Json(serde_json::json!({"api":"frontdesk","api_version":1,"recorded":true,"entry":entry}))
        .into_response()
}

/// 这一单的助手汇报（sop_result），取自补发缓冲。
/// 汇报以前只经 SSE 推给前台：SSE 断了（助手重启、反代半开连接、锁屏）就再也收不到，
/// 任务停在门步、第一节却空着。本项目的原则是「SSE 只当门铃，轮询才是真相」——汇报也要能拉取。
fn reports_for_run(state: &AppState, run_id: &str) -> Vec<serde_json::Value> {
    state
        .recent_sop_results
        .lock()
        .iter()
        .filter(|e| e.get("run_id").and_then(|v| v.as_str()) == Some(run_id))
        .map(|e| {
            serde_json::json!({
                "id": e.get("id"),
                "response": e.get("response"),
                "timestamp": e.get("timestamp"),
            })
        })
        .collect()
}

/// 一单写出的一个文件。
pub(crate) struct TaskArtifact {
    pub id: String,
    pub rel: String,
    pub abs: std::path::PathBuf,
    pub bytes: u64,
}

/// 这一单写出的文件：取自它各轮里**成功的** file_write / file_edit 调用的路径，
/// 同一路径只算一次（取最后一次），且此刻仍在工作区里。
/// 只有这里列出的文件能经 /api/tasks/{id}/artifacts/{aid} 取回——
/// 前台不能借这个端点读工作区里的任意文件（比如别的病例）。
pub(crate) fn task_artifacts(
    evidence: &[crate::ledger::Evidence],
    workspace: &std::path::Path,
) -> Vec<TaskArtifact> {
    let mut paths: Vec<String> = Vec::new();
    for e in evidence {
        if !(e.tool == "file_write" || e.tool == "file_edit") || !e.success {
            continue;
        }
        if let Some(p) = &e.path {
            paths.retain(|x| x != p);
            paths.push(p.clone());
        }
    }
    let Ok(root) = workspace.canonicalize() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for rel in paths {
        let candidate = std::path::Path::new(&rel);
        // 只接受工作区内的相对路径：绝对路径、「..」一律不要
        if candidate.is_absolute()
            || candidate
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            continue;
        }
        let Ok(abs) = root.join(candidate).canonicalize() else {
            continue;
        };
        if !abs.starts_with(&root) {
            continue;
        }
        let Ok(meta) = std::fs::metadata(&abs) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let id = format!("A{}", out.len() + 1);
        out.push(TaskArtifact {
            id,
            rel,
            abs,
            bytes: meta.len(),
        });
    }
    out
}

/// 产物单次最多回传的字节数：报告是文本，远小于此；超过就明说太大，不截半份给人看
const ARTIFACT_MAX_BYTES: u64 = 2 * 1024 * 1024;

/// GET /api/tasks/{run_id}/artifacts/{aid} — 这一单写出的某个文件的内容。
/// 与证据全文同一原则：只走 Bearer 鉴权的 JSON，**不发签名 URL**——报告里是患者资料，
/// 签名链接会被转发、被贴进聊天。
pub async fn handle_api_task_artifact(
    State(state): State<AppState>,
    Path((run_id, aid)): Path<(String, String)>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let (trace_path, full) = trace_setup(&state);
    let avail = crate::ledger::availability(&trace_path, full);
    if !avail.available {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"api":"frontdesk","ledger":avail})),
        )
            .into_response();
    }
    let workspace = state.config.lock().workspace_dir.clone();
    let evidence = crate::ledger::evidence_for_run(&trace_path, &run_id);
    let Some(a) = task_artifacts(&evidence, &workspace)
        .into_iter()
        .find(|a| a.id == aid)
    else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"api":"frontdesk","error":"这一单没有这个产物，或文件已不在工作区","aid":aid})),
        )
            .into_response();
    };
    if a.bytes > ARTIFACT_MAX_BYTES {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(serde_json::json!({"api":"frontdesk","error":"文件太大，无法在前台直接查看","bytes":a.bytes})),
        )
            .into_response();
    }
    match std::fs::read(&a.abs) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => Json(serde_json::json!({
                "api": "frontdesk",
                "api_version": 1,
                "id": a.id,
                "path": a.rel,
                "bytes": a.bytes,
                "content": text,
            }))
            .into_response(),
            Err(_) => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                Json(serde_json::json!({"api":"frontdesk","error":"不是文本文件，前台暂不支持查看","path":a.rel})),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"api":"frontdesk","error":format!("读取失败：{e}"),"path":a.rel})),
        )
            .into_response(),
    }
}

/// GET /api/tasks/{run_id}/evidence/{eid} — 证据全文。
/// 只走 Bearer 鉴权的 JSON，**不发签名 URL**：签名链接会被转发、被贴进聊天，
/// 而这里的内容可能含患者数据。
pub async fn handle_api_task_evidence(
    State(state): State<AppState>,
    Path((run_id, eid)): Path<(String, String)>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let (trace_path, full) = trace_setup(&state);
    let avail = crate::ledger::availability(&trace_path, full);
    if !avail.available {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"api":"frontdesk","ledger":avail})),
        )
            .into_response();
    }
    match crate::ledger::evidence_full_for_run(&trace_path, &run_id, &eid) {
        Some((meta, full_text)) => Json(serde_json::json!({
            "api": "frontdesk",
            "api_version": 1,
            "evidence": meta,
            "output": full_text,
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "证据不存在或内容哈希不匹配（记录可能已被改动或轮转）",
                "eid": eid,
            })),
        )
            .into_response(),
    }
}

// ── P2：服务端检索与修订回流 ──────────────────────────────────────

/// 一个可检索的 run 快照：(run_id, sop_name, trigger_payload, [(步号, 产出)])
type SearchableRun = (String, String, Option<String>, Vec<(u32, String)>);

#[derive(Debug, Default, serde::Deserialize)]
pub struct SearchQuery {
    #[serde(default)]
    pub q: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// GET /api/search?q= —— 在本助手的台账里检索。
///
/// 用**子串匹配**而不是全文索引：服务端现有的 FTS5 用默认分词器，对中文按空白切词，
/// 等于不可用。子串匹配对 CJK 反而可靠，代价是没有相关性排序——如实说明，不假装智能。
/// 检索范围也如实回报：任务参数、步骤产出、证据（工具调用参数与返回）。
pub async fn handle_api_search(
    State(state): State<AppState>,
    Query(q): Query<SearchQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let needle = q.q.unwrap_or_default().trim().to_lowercase();
    let limit = q.limit.unwrap_or(50).min(200);
    if needle.is_empty() {
        return Json(serde_json::json!({
            "api": "frontdesk", "api_version": 1,
            "query": "", "hits": [], "total": 0,
            "scope": "未提供检索词。",
        }))
        .into_response();
    }

    let (trace_path, full) = trace_setup(&state);
    let avail = crate::ledger::availability(&trace_path, full);
    let mut hits: Vec<serde_json::Value> = Vec::new();

    let runs: Vec<SearchableRun> = {
        let engine = state.sop_engine.lock().unwrap();
        let mut all: Vec<&crate::sop::types::SopRun> = engine.active_runs().values().collect();
        all.extend(engine.finished_runs(None));
        all.into_iter()
            .map(|r| {
                (
                    r.run_id.clone(),
                    r.sop_name.clone(),
                    r.trigger_event.payload.clone(),
                    r.step_results
                        .iter()
                        .map(|sr| (sr.step_number, sr.output.clone()))
                        .collect(),
                )
            })
            .collect()
    };

    let snippet = |text: &str| -> String {
        let lower = text.to_lowercase();
        match lower.find(&needle) {
            Some(i) => {
                let start = text
                    .char_indices()
                    .map(|(b, _)| b)
                    .rfind(|b| *b <= i.saturating_sub(30))
                    .unwrap_or(0);
                let s: String = text[start..].chars().take(90).collect();
                format!("…{s}…")
            }
            None => text.chars().take(90).collect(),
        }
    };

    for (run_id, sop_name, payload, steps) in &runs {
        if sop_name.to_lowercase().contains(&needle) {
            hits.push(serde_json::json!({
                "type":"task","task_id":run_id,"path":[run_id, "规程"],
                "snippet": sop_name, "meta": "任务所依规程"
            }));
        }
        if let Some(p) = payload {
            if p.to_lowercase().contains(&needle) {
                hits.push(serde_json::json!({
                    "type":"task","task_id":run_id,"path":[run_id,"任务参数"],
                    "snippet": snippet(p), "meta": "发起时的参数"
                }));
            }
        }
        for (n, out) in steps {
            if out.to_lowercase().contains(&needle) {
                hits.push(serde_json::json!({
                    "type":"step","task_id":run_id,"path":[run_id, format!("第 {n} 步产出")],
                    "snippet": snippet(out), "meta": "助手自述的步骤产出（未经核验）"
                }));
            }
        }
        if avail.available {
            {
                for e in crate::ledger::evidence_for_run(&trace_path, run_id) {
                    let hay = format!(
                        "{} {}",
                        e.args_excerpt.clone().unwrap_or_default(),
                        e.output_excerpt
                    );
                    if hay.to_lowercase().contains(&needle) {
                        hits.push(serde_json::json!({
                            "type":"evidence","task_id":run_id,"eid":e.eid,
                            "path":[run_id, "依据", e.tool.clone()],
                            "snippet": snippet(&hay),
                            "meta": format!("{:?}", e.source_class).to_lowercase(),
                        }));
                    }
                }
            }
        }
        if hits.len() >= limit {
            break;
        }
    }

    let truncated = hits.len() >= limit;
    hits.truncate(limit);
    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "query": needle,
        "total": hits.len(),
        "truncated": truncated,
        "ledger": avail,
        "hits": hits,
        "scope": if avail.available {
            "检索范围：本助手当前还持有的任务（执行状态只在内存，助手一重启就没了）的任务参数、各步骤产出、以及证据台账里的工具调用参数与返回。子串匹配，无相关性排序。"
        } else {
            "检索范围：本助手当前还持有的任务（执行状态只在内存，助手一重启就没了）的任务参数与各步骤产出。证据台账未开启，工具调用记录不在检索范围内。"
        },
    }))
    .into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct FeedbackBody {
    /// 来源任务的 run_id（可选，但强烈建议带上——修订要能追溯到具体场景）
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub sop: Option<String>,
    /// 用户原话
    pub text: String,
    /// 提出人（前台登记的交付对象姓名；未登记为 None）
    #[serde(default)]
    pub proposer: Option<String>,
}

/// 反馈目录。平台侧的收集器（lobster-feedback.path）监视的是 `outbox/` **目录**，
/// 一条反馈一个三段式 md 文件——取走即处理。
///
/// 2026-09-14 实测：前台以前把建议追加进 `outbox.jsonl`（与 outbox/ 目录同级的一个文件），
/// 收集器根本不看它：提交了、界面也说「已记入」，但永远不会有人处理。
fn feedback_dir(state: &AppState) -> std::path::PathBuf {
    state.config.lock().workspace_dir.join("feedback")
}

/// 前台提交记录的索引（只用来列表和追踪状态；处理流程读的是 outbox/ 里的 md）
const FEEDBACK_INDEX: &str = "frontdesk-submitted.jsonl";
/// 旧版前台写的文件：里面的建议从未进入处理流程
const FEEDBACK_LEGACY: &str = "outbox.jsonl";
/// 在 outbox/ 里放了这么久还没被取走，说明这台助手没接入平台的处理流程
const FEEDBACK_STALE_SECS: i64 = 3600;

/// 与平台收集器同一套兜底脱敏（手机号 / 证件号 / 邮箱），写盘前先做一遍
pub(crate) fn redact_feedback(text: &str) -> String {
    static RULES: std::sync::OnceLock<[(regex::Regex, &'static str); 3]> =
        std::sync::OnceLock::new();
    let rules = RULES.get_or_init(|| {
        [
            (regex::Regex::new(r"1[3-9][0-9]{9}").unwrap(), "〈手机号〉"),
            (
                regex::Regex::new(r"[0-9]{17}[0-9Xx]").unwrap(),
                "〈证件号〉",
            ),
            (
                regex::Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+").unwrap(),
                "〈邮箱〉",
            ),
        ]
    });
    // 先证件号（18 位）再手机号（11 位），免得证件号里的一段被当成手机号
    let t = rules[1].0.replace_all(text, rules[1].1);
    let t = rules[0].0.replace_all(&t, rules[0].1);
    rules[2].0.replace_all(&t, rules[2].1).into_owned()
}

/// 三段式反馈文件（平台 README 规定的格式）。反馈编号写进文件，处理时要求写进 CHANGELOG，
/// 前台据此判断「已并入」——不靠任何人手工回填状态。
pub(crate) fn feedback_markdown(
    id: &str,
    text: &str,
    sop: Option<&str>,
    sop_version: Option<&str>,
    task: Option<&str>,
    run: Option<(&str, u32, &str)>,
    artifacts: &[String],
    proposer: Option<&str>,
) -> String {
    let sop_label = match (sop, sop_version) {
        (Some(s), Some(v)) => format!("规程 {s}（v{v}）"),
        (Some(s), None) => format!("规程 {s}"),
        _ => "（未指明规程）".to_string(),
    };
    let mut scene = match (run, task) {
        (Some((rid, step, status)), _) => {
            format!("任务 {rid}，提出时位于第 {step} 步（状态 {status}）。")
        }
        // 引擎忘了这一单（助手重启过）——编号仍然有效，产物与调用记录都还在
        (None, Some(t)) => format!("任务 {t}（助手重启过，服务端已不保留它的执行状态）。"),
        (None, None) => "未关联具体任务。".to_string(),
    };
    if !artifacts.is_empty() {
        scene.push_str("\n\n该任务写出的文件（相对助手工作区）：\n");
        for a in artifacts {
            scene.push_str("- ");
            scene.push_str(a);
            scene.push('\n');
        }
    }
    let who = proposer
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .unwrap_or("前台用户（姓名未登记）");
    format!(
        "<!-- feedback_id: {id} -->\n\
         ## 现象\n\
         使用者在前台对{sop_label}提出修订建议（见「期望」，为使用者原话）。\n\n\
         ## 复现路径\n\
         {scene}\n\n\
         ## 期望\n\
         {text}\n\n\
         ---\n\
         - 反馈编号：{id}\n\
         - 来源：前台「规程共建」\n\
         - 提出人：{who}\n\
         - 处理要求：若据此修订规程，请在 CHANGELOG 该版本的「反馈来源」里写明反馈编号 {id}；前台据此向提出人显示「已并入该版本」。\n"
    )
}

/// 一条建议此刻的真实状态，只从能看到的事实推出来：
/// 文件还在 outbox/ = 等平台来取；已被取走 = 平台在处理（或没采纳）；
/// 规程 CHANGELOG 里出现了它的编号 = 已并入那一版。
fn feedback_state(
    entry: &serde_json::Value,
    outbox: &std::path::Path,
    changelog_of: &dyn Fn(&str) -> Option<String>,
    now: chrono::DateTime<chrono::Utc>,
) -> serde_json::Value {
    let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let file = entry.get("file").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(sop) = entry.get("sop").and_then(|v| v.as_str()) {
        if !id.is_empty() {
            if let Some(log) = changelog_of(sop) {
                if let Some(version) = changelog_version_mentioning(&log, id) {
                    return serde_json::json!({"state": "merged", "merged_version": version});
                }
            }
        }
    }
    if !file.is_empty() && outbox.join(file).exists() {
        let waited = entry
            .get("at")
            .and_then(|v| v.as_str())
            .and_then(|a| chrono::DateTime::parse_from_rfc3339(a).ok())
            .map(|a| (now - a.with_timezone(&chrono::Utc)).num_seconds())
            .unwrap_or(0);
        return serde_json::json!({
            "state": "queued",
            "waited_secs": waited,
            "stale": waited > FEEDBACK_STALE_SECS,
        });
    }
    serde_json::json!({"state": "collected"})
}

/// CHANGELOG 里提到某个反馈编号的那一节的版本号（节标题形如 `## v1.3.0 — …`）
pub(crate) fn changelog_version_mentioning(log: &str, id: &str) -> Option<String> {
    let mut current: Option<String> = None;
    for line in log.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            current = rest
                .split_whitespace()
                .next()
                .map(|v| v.trim_start_matches('v').to_string());
        } else if line.contains(id) {
            return current;
        }
    }
    None
}

fn sop_changelog(state: &AppState, sop: &str) -> Option<String> {
    let dir = {
        let engine = state.sop_engine.lock().ok()?;
        engine.get_sop(sop)?.location.clone()?
    };
    std::fs::read_to_string(dir.join("CHANGELOG.md")).ok()
}

/// POST /api/feedback —— 规程修订建议回流。
///
/// 重要：**只记录，不自改**。助手不得据此修改自己的规程或人格——那条红线是刻意的
/// （见「反馈闭环」约定）。建议以三段式 md 写进 feedback/outbox/，由平台的收集器取走、
/// 出修订草案、人工确认后发布。接口如实回报 `applied: false`，界面也必须这么说。
pub async fn handle_api_feedback_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<FeedbackBody>>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let Some(Json(b)) = body else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"api":"frontdesk","error":"缺少请求体，需要 {\"text\": \"...\"}"})),
        )
            .into_response();
    };
    let text = redact_feedback(b.text.trim());
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"api":"frontdesk","error":"修订建议不能为空"})),
        )
            .into_response();
    }
    let text: String = text.chars().take(2000).collect();

    let (sop_version, run) = {
        let engine = state.sop_engine.lock().unwrap();
        let version = b
            .sop
            .as_deref()
            .and_then(|s| engine.get_sop(s))
            .map(|s| s.version.clone());
        let run = b.task_id.as_deref().and_then(|rid| {
            engine
                .get_run(rid)
                .map(|r| (r.run_id.clone(), r.current_step, r.status.to_string()))
        });
        (version, run)
    };

    let now = chrono::Utc::now();
    let id = {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(text.as_bytes());
        h.update(now.to_rfc3339().as_bytes());
        let hex = format!("{:x}", h.finalize());
        format!("FB-{}-{}", now.format("%Y%m%d"), &hex[..6])
    };
    let file = format!("{}-frontdesk.md", id);
    let artifacts: Vec<String> = match b.task_id.as_deref() {
        Some(rid) => {
            let (trace_path, full) = trace_setup(&state);
            if crate::ledger::availability(&trace_path, full).available {
                let workspace = state.config.lock().workspace_dir.clone();
                task_artifacts(
                    &crate::ledger::evidence_for_run(&trace_path, rid),
                    &workspace,
                )
                .into_iter()
                .map(|a| a.rel)
                .filter(|p| !p.rsplit('/').next().unwrap_or("").starts_with('.'))
                .collect()
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    };
    let md = feedback_markdown(
        &id,
        &text,
        b.sop.as_deref(),
        sop_version.as_deref(),
        b.task_id.as_deref(),
        run.as_ref().map(|(r, s, st)| (r.as_str(), *s, st.as_str())),
        &artifacts,
        b.proposer.as_deref(),
    );

    let dir = feedback_dir(&state);
    let outbox = dir.join("outbox");
    let written = std::fs::create_dir_all(&outbox)
        .and_then(|()| std::fs::write(outbox.join(&file), md.as_bytes()))
        .and_then(|()| {
            let entry = serde_json::json!({
                "id": id,
                "at": now.to_rfc3339(),
                "task_id": b.task_id,
                "sop": b.sop,
                "sop_version": sop_version,
                "proposer": b.proposer,
                "text": text,
                "file": file,
            });
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(FEEDBACK_INDEX))
                .and_then(|mut f| {
                    std::io::Write::write_all(&mut f, format!("{entry}\n").as_bytes())
                })
        });
    if let Err(e) = written {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"api":"frontdesk","error": format!("无法写入修订建议：{e}")})),
        )
            .into_response();
    }

    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "recorded": true,
        "id": id,
        // 说清楚它现在是什么状态：已记录 ≠ 已生效
        "applied": false,
        "state": "queued",
        "note": format!("已记下（编号 {id}），等平台取走处理。助手不会据此自行修改规程：平台会出修订草案，经人工确认后才发布；并入后这里会显示所在版本。"),
    }))
    .into_response()
}

/// GET /api/feedback —— 前台提交过的修订建议，各自附上从事实推出的状态。
pub async fn handle_api_feedback_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let dir = feedback_dir(&state);
    let outbox = dir.join("outbox");
    let read = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(dir.join(name))
            .map(|raw| {
                raw.lines()
                    .filter(|l| !l.trim().is_empty())
                    .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                    .collect()
            })
            .unwrap_or_default()
    };
    let now = chrono::Utc::now();
    let submitted = read(FEEDBACK_INDEX);
    // 涉及到的规程各读一次 CHANGELOG
    let mut logs: std::collections::HashMap<String, Option<String>> =
        std::collections::HashMap::new();
    for e in &submitted {
        if let Some(sop) = e.get("sop").and_then(|v| v.as_str()) {
            logs.entry(sop.to_string())
                .or_insert_with(|| sop_changelog(&state, sop));
        }
    }
    let changelog_of = |sop: &str| -> Option<String> { logs.get(sop).cloned().flatten() };

    let mut items: Vec<serde_json::Value> = read(FEEDBACK_LEGACY)
        .into_iter()
        .map(|mut e| {
            // 旧版前台写进 outbox.jsonl 的：从来没有进入处理流程，照实标出来
            e["state"] = serde_json::json!("unrouted");
            e
        })
        .collect();
    for mut e in submitted {
        let st = feedback_state(&e, &outbox, &changelog_of, now);
        if let (Some(obj), Some(extra)) = (e.as_object_mut(), st.as_object()) {
            for (k, v) in extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        items.push(e);
    }
    items.sort_by(|a, b| {
        b.get("at")
            .and_then(|v| v.as_str())
            .cmp(&a.get("at").and_then(|v| v.as_str()))
    });
    items.truncate(100);
    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "items": items,
        "note": "修订建议只会交给平台处理，助手不会自行改动规程。状态依据：文件是否已被平台取走，以及规程变更记录里是否出现了该建议的编号。",
    }))
    .into_response()
}

/// GET /api/sops/{name}/changelog —— 规程的版本史（「进化」一节）。原文返回，前台只排版不改写。
pub async fn handle_api_sop_changelog(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let version =
        {
            let engine = state.sop_engine.lock().unwrap();
            match engine.get_sop(&name) {
                Some(s) => s.version.clone(),
                None => return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"api":"frontdesk","error":"没有这个规程","sop":name})),
                )
                    .into_response(),
            }
        };
    match sop_changelog(&state, &name) {
        Some(log) => Json(serde_json::json!({
            "api": "frontdesk",
            "api_version": 1,
            "sop": name,
            "version": version,
            "changelog": log.chars().take(60_000).collect::<String>(),
        }))
        .into_response(),
        None => Json(serde_json::json!({
            "api": "frontdesk",
            "api_version": 1,
            "sop": name,
            "version": version,
            "changelog": serde_json::Value::Null,
        }))
        .into_response(),
    }
}

/// `/api/sop/runs` 的查询参数。
#[derive(Debug, Default, serde::Deserialize)]
pub struct SopRunsQuery {
    /// `finished` = 连同最近结束的 run 一起返回（默认只返回活跃 run，与旧行为一致）
    #[serde(default)]
    pub include: Option<String>,
}

/// GET /api/sop/runs — SOP run 列表（进行中/等待审批/近期完成），供龙虾前台审批箱直读，
/// 免去从对话文本收割 run 编号的脆弱路径。
pub async fn handle_api_sop_runs(
    State(state): State<AppState>,
    Query(q): Query<SopRunsQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    // 前台的核心诚实问题：run 完成后就从 active_runs 消失，客户端无法区分
    // 「完成」「失败」「被取消」「daemon 重启」。`?include=finished` 把最近结束的
    // run 一并返回，让客户端能如实回答，而不是把「消失」一律当成功。
    let include_finished = q.include.as_deref() == Some("finished");
    let runs: Vec<serde_json::Value> = {
        let engine = state.sop_engine.lock().unwrap();
        let mut all: Vec<&crate::sop::types::SopRun> = engine.active_runs().values().collect();
        if include_finished {
            all.extend(engine.finished_runs(None));
        }
        all.into_iter()
            .map(|r| {
                // 当前步骤的标题/内容：审批时用户要看到"批的是什么"
                let step_info = engine
                    .get_sop(&r.sop_name)
                    .and_then(|sop| sop.steps.iter().find(|st| st.number == r.current_step))
                    .map(|st| serde_json::json!({ "title": st.title, "body": st.body }));
                // 审批者要看的是"这一单的实际内容"（前序步骤的产出），不只是步骤说明
                let last_output = r
                    .step_results
                    .iter()
                    .rev()
                    .find(|sr| !sr.output.trim().is_empty())
                    .map(|sr| sr.output.chars().take(2000).collect::<String>());
                // 发起时的任务参数（坐标/对象等确定性输入）——审批判断的核心依据
                let trigger_payload = r
                    .trigger_event
                    .payload
                    .as_ref()
                    .map(|p| p.chars().take(1500).collect::<String>());
                let sop_version = engine.get_sop(&r.sop_name).map(|sop| sop.version.clone());
                serde_json::json!({
                    "current_step_info": step_info,
                    "sop_version": sop_version,
                    "last_output": last_output,
                    "trigger_payload": trigger_payload,
                    // 由什么发起（manual / webhook / cron / mqtt / peripheral）——前台「发起」一栏
                    // 以前只能把交付对象的名字盖在每一单上，连 webhook 起的单也写成那个人
                    "trigger_source": r.trigger_event.source,
                    "trigger_topic": r.trigger_event.topic,
                    "run_id": r.run_id,
                    "sop_name": r.sop_name,
                    "status": r.status,
                    "current_step": r.current_step,
                    "total_steps": r.total_steps,
                    "started_at": r.started_at,
                    "waiting_since": r.waiting_since,
                    "completed_at": r.completed_at,
                })
            })
            .collect()
    };
    Json(serde_json::json!({ "runs": runs })).into_response()
}

#[cfg(test)]
mod feedback_tests {
    use super::{changelog_version_mentioning, feedback_markdown, redact_feedback};

    #[test]
    fn redacts_phone_id_and_email_before_writing() {
        let t = redact_feedback("联系 13812345678，证件 11010519491231002X，邮箱 a.b@x.cn");
        assert!(
            !t.contains("13812345678") && t.contains("〈手机号〉"),
            "{t}"
        );
        assert!(
            !t.contains("11010519491231002X") && t.contains("〈证件号〉"),
            "{t}"
        );
        assert!(!t.contains("a.b@x.cn") && t.contains("〈邮箱〉"), "{t}");
    }

    #[test]
    fn markdown_follows_the_platform_three_part_contract() {
        let md = feedback_markdown(
            "FB-20260914-abc123",
            "今后超声分级要写明体系",
            Some("case-clinical-report"),
            Some("1.2.1"),
            Some("run-1"),
            Some(("run-1", 4, "waiting_approval")),
            &["case_library/a/report_final.md".to_string()],
            None,
        );
        for h in ["## 现象", "## 复现路径", "## 期望"] {
            assert!(md.contains(h), "missing {h}");
        }
        assert!(md.contains("今后超声分级要写明体系"));
        assert!(md.contains("FB-20260914-abc123"));
        assert!(md.contains("姓名未登记"));
        assert!(md.contains("case_library/a/report_final.md"));
    }

    #[test]
    fn merged_version_comes_from_the_changelog_section_that_names_the_id() {
        let log = "# CHANGELOG\n\n## v1.3.0 — 2026-09-14（x）\n**反馈来源：** 前台 FB-20260914-abc123\n\n## v1.2.1 — 2026-06-09\n无关\n";
        assert_eq!(
            changelog_version_mentioning(log, "FB-20260914-abc123").as_deref(),
            Some("1.3.0")
        );
        assert_eq!(
            changelog_version_mentioning(log, "FB-20260914-zzzzzz"),
            None
        );
    }
}

#[cfg(test)]
mod task_artifact_tests {
    use super::task_artifacts;
    use crate::ledger::{Evidence, SourceClass};

    fn write(path: &str, ok: bool) -> Evidence {
        Evidence {
            eid: "E1-000000".into(),
            n: 1,
            at: String::new(),
            tool: "file_write".into(),
            source_class: SourceClass::SelfProduced,
            args_excerpt: None,
            output_excerpt: String::new(),
            output_bytes: 0,
            sha256: "0".repeat(64),
            success: ok,
            turn_id: "t".into(),
            path: Some(path.into()),
            url: None,
        }
    }

    #[test]
    fn lists_only_files_this_task_wrote_inside_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();
        std::fs::create_dir_all(ws.join("case/a")).unwrap();
        std::fs::write(ws.join("case/a/report_final.md"), "# 报告").unwrap();
        std::fs::write(ws.join("case/a/draft.md"), "草稿").unwrap();
        let outside = dir.path().parent().unwrap().join("outside.md");
        let ev = vec![
            write("case/a/draft.md", true),
            write("case/a/report_final.md", true),
            write("case/a/draft.md", true),   // 重写：同一路径只算一次
            write("case/a/failed.md", false), // 写失败的不算
            write("../outside.md", true),     // 跳出工作区的不要
            write(outside.to_str().unwrap(), true), // 绝对路径不要
            write("case/a/gone.md", true),    // 已不在工作区的不列
        ];
        let arts = task_artifacts(&ev, ws);
        let paths: Vec<&str> = arts.iter().map(|a| a.rel.as_str()).collect();
        assert_eq!(paths, vec!["case/a/report_final.md", "case/a/draft.md"]);
        assert_eq!(arts[0].id, "A1");
    }
}

#[cfg(test)]
mod sop_title_tests {
    use super::heading_title;

    #[test]
    fn takes_the_first_h1_without_the_sop_suffix() {
        assert_eq!(
            heading_title("# 临床病例解读报告 SOP\n\n正文").as_deref(),
            Some("临床病例解读报告")
        );
        assert_eq!(heading_title("## 二级\n# 标题").as_deref(), Some("标题"));
        assert_eq!(heading_title("没有标题"), None);
        assert_eq!(heading_title("# SOP"), None);
    }
}

#[cfg(test)]
mod frontdesk_v6_tests {
    //! 龙虾前台 v6（PR-A）新增能力的回归测试。
    //! 关心的不是「返回 200」，而是**客户端能否确定性地判断服务端支持什么、以及状态是否已变**。
    use crate::config::SopConfig;
    use crate::sop::engine::SopEngine;
    use crate::sop::types::{
        Sop, SopEvent, SopExecutionMode, SopPriority, SopStep, SopTriggerSource,
    };

    fn gated_sop() -> Sop {
        Sop {
            name: "case-clinical-report".into(),
            description: "病例解读".into(),
            version: "1.2.1".into(),
            priority: SopPriority::High,
            execution_mode: SopExecutionMode::Auto,
            triggers: vec![],
            steps: vec![
                SopStep {
                    number: 1,
                    title: "解析病例资料".into(),
                    body: String::new(),
                    suggested_tools: vec![],
                    requires_confirmation: false,
                    kind: crate::sop::types::SopStepKind::default(),
                    schema: None,
                },
                SopStep {
                    number: 2,
                    title: "审核确认".into(),
                    body: String::new(),
                    suggested_tools: vec![],
                    requires_confirmation: true,
                    kind: crate::sop::types::SopStepKind::default(),
                    schema: None,
                },
            ],
            cooldown_secs: 0,
            max_concurrent: 4,
            location: None,
            deterministic: false,
        }
    }

    fn engine_with_run() -> (SopEngine, String) {
        let mut e = SopEngine::new(SopConfig::default());
        e.set_sops_for_test(vec![gated_sop()]);
        let action = e
            .start_run(
                "case-clinical-report",
                SopEvent {
                    source: SopTriggerSource::Manual,
                    topic: None,
                    payload: Some("{\"case_id\":\"20260910-肺结节\"}".into()),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                },
            )
            .expect("start");
        let run_id = match action {
            crate::sop::types::SopRunAction::ExecuteStep { run_id, .. } => run_id,
            other => panic!("unexpected action: {other:?}"),
        };
        (e, run_id)
    }

    #[test]
    fn finished_runs_are_queryable_so_disappearance_is_not_ambiguous() {
        // run 完成后从 active_runs 消失。若服务端不提供已结束列表，
        // 客户端只能在「完成 / 失败 / 被取消 / daemon 重启」之间瞎猜。
        let (mut e, run_id) = engine_with_run();
        assert_eq!(e.active_runs().len(), 1);
        e.cancel_run(&run_id).expect("cancel");
        assert!(e.active_runs().is_empty(), "取消后应移出活跃列表");
        let finished = e.finished_runs(None);
        assert_eq!(finished.len(), 1, "已结束的 run 必须仍可查到");
        assert_eq!(finished[0].run_id, run_id);
    }

    #[test]
    fn get_run_finds_both_active_and_finished() {
        // expect_step 的比对依赖它：run 刚结束时也要能查到，才不会把 409 误判成 404。
        let (mut e, run_id) = engine_with_run();
        assert!(e.get_run(&run_id).is_some());
        e.cancel_run(&run_id).expect("cancel");
        assert!(e.get_run(&run_id).is_some(), "结束后仍应可查");
    }

    #[test]
    fn feedback_records_but_never_applies() {
        // 这条不变量是刻意的：助手只记录用户意见，绝不据此自改规程或人格。
        // 接口必须如实回报 applied=false，界面才不会把「已记录」说成「已生效」。
        let body = serde_json::json!({
            "recorded": true, "applied": false, "state": "pending_review"
        });
        assert_eq!(body["recorded"], true);
        assert_eq!(
            body["applied"], false,
            "已记录不等于已生效——回报里绝不能出现 applied=true"
        );
    }

    #[test]
    fn search_scope_is_stated_and_shrinks_without_ledger() {
        // 检索范围必须随台账可用性如实变化，不能永远宣称「什么都能搜到」
        let with = "检索范围：本助手当前还持有的任务（执行状态只在内存，助手一重启就没了）的任务参数、各步骤产出、以及证据台账里的工具调用参数与返回。子串匹配，无相关性排序。";
        let without =
            "检索范围：本助手当前还持有的任务（执行状态只在内存，助手一重启就没了）的任务参数与各步骤产出。证据台账未开启，工具调用记录不在检索范围内。";
        assert!(with.contains("证据台账"));
        assert!(without.contains("未开启"));
        assert!(!without.contains("工具调用参数与返回"));
    }

    #[test]
    fn step_short_label_does_not_cut_mid_word() {
        assert_eq!(super::short_label("解析病例资料"), "解析病例");
        assert_eq!(super::short_label("检索 PubMed 指南"), "检索"); // 不要「检索 P」
        assert_eq!(super::short_label("审核确认"), "审核确认");
        assert_eq!(super::short_label("PubMed"), "PubM"); // 全 ASCII 时保留前 4 字符
    }

    #[test]
    fn gate_steps_are_derivable_for_capabilities() {
        // /api/capabilities 要如实告诉前台「第几步需要人确认」，
        // 这个信息只能来自 SOP 定义里的 requires_confirmation。
        let sop = gated_sop();
        let gates: Vec<u32> = sop
            .steps
            .iter()
            .filter(|s| s.requires_confirmation)
            .map(|s| s.number)
            .collect();
        assert_eq!(gates, vec![2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masking_keeps_toml_valid_and_preserves_api_keys_type() {
        let mut cfg = crate::config::Config::default();
        cfg.api_key = Some("sk-live-123".to_string());
        cfg.reliability.api_keys = vec!["rk-1".to_string(), "rk-2".to_string()];
        cfg.gateway.paired_tokens = vec!["pair-token-1".to_string()];
        cfg.tunnel.cloudflare = Some(crate::config::schema::CloudflareTunnelConfig {
            token: "cf-token".to_string(),
        });
        cfg.memory.qdrant.api_key = Some("qdrant-key".to_string());
        cfg.channels_config.wati = Some(crate::config::schema::WatiConfig {
            api_token: "wati-token".to_string(),
            api_url: "https://live-mt-server.wati.io".to_string(),
            tenant_id: None,
            allowed_numbers: vec![],
        });
        cfg.channels_config.feishu = Some(crate::config::schema::FeishuConfig {
            app_id: "cli_aabbcc".to_string(),
            app_secret: "feishu-secret".to_string(),
            encrypt_key: Some("feishu-encrypt".to_string()),
            verification_token: Some("feishu-verify".to_string()),
            allowed_users: vec!["*".to_string()],
            receive_mode: crate::config::schema::LarkReceiveMode::Websocket,
            port: None,
        });
        cfg.channels_config.email = Some(crate::channels::email_channel::EmailConfig {
            imap_host: "imap.example.com".to_string(),
            imap_port: 993,
            imap_folder: "INBOX".to_string(),
            smtp_host: "smtp.example.com".to_string(),
            smtp_port: 465,
            smtp_tls: true,
            username: "agent@example.com".to_string(),
            password: "email-password-secret".to_string(),
            from_address: "agent@example.com".to_string(),
            idle_timeout_secs: 1740,
            allowed_senders: vec!["*".to_string()],
            default_subject: "ZeroClaw Message".to_string(),
        });
        cfg.model_routes = vec![crate::config::schema::ModelRouteConfig {
            hint: "reasoning".to_string(),
            provider: "openrouter".to_string(),
            model: "anthropic/claude-sonnet-4.6".to_string(),
            api_key: Some("route-model-key".to_string()),
        }];
        cfg.embedding_routes = vec![crate::config::schema::EmbeddingRouteConfig {
            hint: "semantic".to_string(),
            provider: "openai".to_string(),
            model: "text-embedding-3-small".to_string(),
            dimensions: Some(1536),
            api_key: Some("route-embed-key".to_string()),
        }];

        let masked = mask_sensitive_fields(&cfg);
        let toml = toml::to_string_pretty(&masked).expect("masked config should serialize");
        let parsed: crate::config::Config =
            toml::from_str(&toml).expect("masked config should remain valid TOML for Config");

        assert_eq!(parsed.api_key.as_deref(), Some(MASKED_SECRET));
        assert_eq!(
            parsed.reliability.api_keys,
            vec![MASKED_SECRET.to_string(), MASKED_SECRET.to_string()]
        );
        assert_eq!(
            parsed.gateway.paired_tokens,
            vec![MASKED_SECRET.to_string()]
        );
        assert_eq!(
            parsed.tunnel.cloudflare.as_ref().map(|v| v.token.as_str()),
            Some(MASKED_SECRET)
        );
        assert_eq!(
            parsed
                .channels_config
                .wati
                .as_ref()
                .map(|v| v.api_token.as_str()),
            Some(MASKED_SECRET)
        );
        assert_eq!(parsed.memory.qdrant.api_key.as_deref(), Some(MASKED_SECRET));
        assert_eq!(
            parsed
                .channels_config
                .feishu
                .as_ref()
                .map(|v| v.app_secret.as_str()),
            Some(MASKED_SECRET)
        );
        assert_eq!(
            parsed
                .channels_config
                .feishu
                .as_ref()
                .and_then(|v| v.encrypt_key.as_deref()),
            Some(MASKED_SECRET)
        );
        assert_eq!(
            parsed
                .channels_config
                .feishu
                .as_ref()
                .and_then(|v| v.verification_token.as_deref()),
            Some(MASKED_SECRET)
        );
        assert_eq!(
            parsed
                .model_routes
                .first()
                .and_then(|v| v.api_key.as_deref()),
            Some(MASKED_SECRET)
        );
        assert_eq!(
            parsed
                .embedding_routes
                .first()
                .and_then(|v| v.api_key.as_deref()),
            Some(MASKED_SECRET)
        );
        assert_eq!(
            parsed
                .channels_config
                .email
                .as_ref()
                .map(|v| v.password.as_str()),
            Some(MASKED_SECRET)
        );
    }

    #[test]
    fn hydrate_config_for_save_restores_masked_secrets_and_paths() {
        let mut current = crate::config::Config::default();
        current.config_path = std::path::PathBuf::from("/tmp/current/config.toml");
        current.workspace_dir = std::path::PathBuf::from("/tmp/current/workspace");
        current.api_key = Some("real-key".to_string());
        current.reliability.api_keys = vec!["r1".to_string(), "r2".to_string()];
        current.gateway.paired_tokens = vec!["pair-1".to_string(), "pair-2".to_string()];
        current.tunnel.cloudflare = Some(crate::config::schema::CloudflareTunnelConfig {
            token: "cf-token-real".to_string(),
        });
        current.tunnel.ngrok = Some(crate::config::schema::NgrokTunnelConfig {
            auth_token: "ngrok-token-real".to_string(),
            domain: None,
        });
        current.memory.qdrant.api_key = Some("qdrant-real".to_string());
        current.channels_config.wati = Some(crate::config::schema::WatiConfig {
            api_token: "wati-real".to_string(),
            api_url: "https://live-mt-server.wati.io".to_string(),
            tenant_id: None,
            allowed_numbers: vec![],
        });
        current.channels_config.feishu = Some(crate::config::schema::FeishuConfig {
            app_id: "cli_current".to_string(),
            app_secret: "feishu-secret-real".to_string(),
            encrypt_key: Some("feishu-encrypt-real".to_string()),
            verification_token: Some("feishu-verify-real".to_string()),
            allowed_users: vec!["*".to_string()],
            receive_mode: crate::config::schema::LarkReceiveMode::Websocket,
            port: None,
        });
        current.channels_config.email = Some(crate::channels::email_channel::EmailConfig {
            imap_host: "imap.example.com".to_string(),
            imap_port: 993,
            imap_folder: "INBOX".to_string(),
            smtp_host: "smtp.example.com".to_string(),
            smtp_port: 465,
            smtp_tls: true,
            username: "agent@example.com".to_string(),
            password: "email-password-real".to_string(),
            from_address: "agent@example.com".to_string(),
            idle_timeout_secs: 1740,
            allowed_senders: vec!["*".to_string()],
            default_subject: "ZeroClaw Message".to_string(),
        });
        current.model_routes = vec![
            crate::config::schema::ModelRouteConfig {
                hint: "reasoning".to_string(),
                provider: "openrouter".to_string(),
                model: "anthropic/claude-sonnet-4.6".to_string(),
                api_key: Some("route-model-key-1".to_string()),
            },
            crate::config::schema::ModelRouteConfig {
                hint: "fast".to_string(),
                provider: "openrouter".to_string(),
                model: "openai/gpt-4.1-mini".to_string(),
                api_key: Some("route-model-key-2".to_string()),
            },
        ];
        current.embedding_routes = vec![
            crate::config::schema::EmbeddingRouteConfig {
                hint: "semantic".to_string(),
                provider: "openai".to_string(),
                model: "text-embedding-3-small".to_string(),
                dimensions: Some(1536),
                api_key: Some("route-embed-key-1".to_string()),
            },
            crate::config::schema::EmbeddingRouteConfig {
                hint: "archive".to_string(),
                provider: "custom:https://emb.example.com/v1".to_string(),
                model: "bge-m3".to_string(),
                dimensions: Some(1024),
                api_key: Some("route-embed-key-2".to_string()),
            },
        ];

        let mut incoming = mask_sensitive_fields(&current);
        incoming.default_model = Some("gpt-4.1-mini".to_string());
        // Simulate UI changing only one key and keeping the first masked.
        incoming.reliability.api_keys = vec![MASKED_SECRET.to_string(), "r2-new".to_string()];
        incoming.gateway.paired_tokens = vec![MASKED_SECRET.to_string(), "pair-2-new".to_string()];
        if let Some(cloudflare) = incoming.tunnel.cloudflare.as_mut() {
            cloudflare.token = MASKED_SECRET.to_string();
        }
        if let Some(ngrok) = incoming.tunnel.ngrok.as_mut() {
            ngrok.auth_token = MASKED_SECRET.to_string();
        }
        incoming.memory.qdrant.api_key = Some(MASKED_SECRET.to_string());
        if let Some(wati) = incoming.channels_config.wati.as_mut() {
            wati.api_token = MASKED_SECRET.to_string();
        }
        if let Some(feishu) = incoming.channels_config.feishu.as_mut() {
            feishu.app_secret = MASKED_SECRET.to_string();
            feishu.encrypt_key = Some(MASKED_SECRET.to_string());
            feishu.verification_token = Some("feishu-verify-new".to_string());
        }
        if let Some(email) = incoming.channels_config.email.as_mut() {
            email.password = MASKED_SECRET.to_string();
        }
        incoming.model_routes[1].api_key = Some("route-model-key-2-new".to_string());
        incoming.embedding_routes[1].api_key = Some("route-embed-key-2-new".to_string());

        let hydrated = hydrate_config_for_save(incoming, &current);

        assert_eq!(hydrated.config_path, current.config_path);
        assert_eq!(hydrated.workspace_dir, current.workspace_dir);
        assert_eq!(hydrated.api_key, current.api_key);
        assert_eq!(hydrated.default_model.as_deref(), Some("gpt-4.1-mini"));
        assert_eq!(
            hydrated.reliability.api_keys,
            vec!["r1".to_string(), "r2-new".to_string()]
        );
        assert_eq!(
            hydrated.gateway.paired_tokens,
            vec!["pair-1".to_string(), "pair-2-new".to_string()]
        );
        assert_eq!(
            hydrated
                .tunnel
                .cloudflare
                .as_ref()
                .map(|v| v.token.as_str()),
            Some("cf-token-real")
        );
        assert_eq!(
            hydrated
                .tunnel
                .ngrok
                .as_ref()
                .map(|v| v.auth_token.as_str()),
            Some("ngrok-token-real")
        );
        assert_eq!(
            hydrated.memory.qdrant.api_key.as_deref(),
            Some("qdrant-real")
        );
        assert_eq!(
            hydrated
                .channels_config
                .wati
                .as_ref()
                .map(|v| v.api_token.as_str()),
            Some("wati-real")
        );
        assert_eq!(
            hydrated
                .channels_config
                .feishu
                .as_ref()
                .map(|v| v.app_secret.as_str()),
            Some("feishu-secret-real")
        );
        assert_eq!(
            hydrated
                .channels_config
                .feishu
                .as_ref()
                .and_then(|v| v.encrypt_key.as_deref()),
            Some("feishu-encrypt-real")
        );
        assert_eq!(
            hydrated
                .channels_config
                .feishu
                .as_ref()
                .and_then(|v| v.verification_token.as_deref()),
            Some("feishu-verify-new")
        );
        assert_eq!(
            hydrated.model_routes[0].api_key.as_deref(),
            Some("route-model-key-1")
        );
        assert_eq!(
            hydrated.model_routes[1].api_key.as_deref(),
            Some("route-model-key-2-new")
        );
        assert_eq!(
            hydrated.embedding_routes[0].api_key.as_deref(),
            Some("route-embed-key-1")
        );
        assert_eq!(
            hydrated.embedding_routes[1].api_key.as_deref(),
            Some("route-embed-key-2-new")
        );
        assert_eq!(
            hydrated
                .channels_config
                .email
                .as_ref()
                .map(|v| v.password.as_str()),
            Some("email-password-real")
        );
    }

    #[test]
    fn hydrate_config_for_save_restores_route_keys_by_identity_and_clears_unmatched_masks() {
        let mut current = crate::config::Config::default();
        current.model_routes = vec![
            crate::config::schema::ModelRouteConfig {
                hint: "reasoning".to_string(),
                provider: "openrouter".to_string(),
                model: "anthropic/claude-sonnet-4.6".to_string(),
                api_key: Some("route-model-key-1".to_string()),
            },
            crate::config::schema::ModelRouteConfig {
                hint: "fast".to_string(),
                provider: "openrouter".to_string(),
                model: "openai/gpt-4.1-mini".to_string(),
                api_key: Some("route-model-key-2".to_string()),
            },
        ];
        current.embedding_routes = vec![
            crate::config::schema::EmbeddingRouteConfig {
                hint: "semantic".to_string(),
                provider: "openai".to_string(),
                model: "text-embedding-3-small".to_string(),
                dimensions: Some(1536),
                api_key: Some("route-embed-key-1".to_string()),
            },
            crate::config::schema::EmbeddingRouteConfig {
                hint: "archive".to_string(),
                provider: "custom:https://emb.example.com/v1".to_string(),
                model: "bge-m3".to_string(),
                dimensions: Some(1024),
                api_key: Some("route-embed-key-2".to_string()),
            },
        ];

        let mut incoming = mask_sensitive_fields(&current);
        incoming.model_routes.swap(0, 1);
        incoming.embedding_routes.swap(0, 1);
        incoming
            .model_routes
            .push(crate::config::schema::ModelRouteConfig {
                hint: "new".to_string(),
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                api_key: Some(MASKED_SECRET.to_string()),
            });
        incoming
            .embedding_routes
            .push(crate::config::schema::EmbeddingRouteConfig {
                hint: "new-embed".to_string(),
                provider: "custom:https://emb2.example.com/v1".to_string(),
                model: "bge-small".to_string(),
                dimensions: Some(768),
                api_key: Some(MASKED_SECRET.to_string()),
            });

        let hydrated = hydrate_config_for_save(incoming, &current);

        assert_eq!(
            hydrated.model_routes[0].api_key.as_deref(),
            Some("route-model-key-2")
        );
        assert_eq!(
            hydrated.model_routes[1].api_key.as_deref(),
            Some("route-model-key-1")
        );
        assert_eq!(hydrated.model_routes[2].api_key, None);
        assert_eq!(
            hydrated.embedding_routes[0].api_key.as_deref(),
            Some("route-embed-key-2")
        );
        assert_eq!(
            hydrated.embedding_routes[1].api_key.as_deref(),
            Some("route-embed-key-1")
        );
        assert_eq!(hydrated.embedding_routes[2].api_key, None);
        assert!(hydrated
            .model_routes
            .iter()
            .all(|route| route.api_key.as_deref() != Some(MASKED_SECRET)));
        assert!(hydrated
            .embedding_routes
            .iter()
            .all(|route| route.api_key.as_deref() != Some(MASKED_SECRET)));
    }
}
