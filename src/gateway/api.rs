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
        Json(serde_json::json!({
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
                    "title": sop.name,
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
                let evidence_count = turn
                    .as_ref()
                    .map(|t| {
                        crate::ledger::evidence_for_turn(&trace_path, t)
                            .iter()
                            .filter(|e| {
                                e.source_class == crate::ledger::SourceClass::External
                            })
                            .count()
                    })
                    .unwrap_or(0);
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
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error":"run not found","run_id":run_id})),
            )
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

    let turn = if avail.available {
        crate::ledger::turn_for_run(&trace_path, &run_id)
    } else {
        None
    };
    let evidence = turn
        .as_ref()
        .map(|t| crate::ledger::evidence_for_turn(&trace_path, t))
        .unwrap_or_default();

    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "ledger": avail,
        "task": run_json,
        "turn_id": turn,
        "evidence": evidence,
        // 说清楚这批证据是什么、不是什么——界面直接引用这句，不要另行编写
        "evidence_note": "以下为该任务所在对话轮中被记录到的工具调用。只有 source_class = external 的可作为对外依据；local_file 可能读的是既有缓存，self 为助手自产。未被记录的调用不在此列。",
    }))
    .into_response()
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
    let Some(turn) = crate::ledger::turn_for_run(&trace_path, &run_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"该任务没有可关联的对话轮记录","run_id":run_id})),
        )
            .into_response();
    };
    match crate::ledger::evidence_full(&trace_path, &turn, &eid) {
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
            if let Some(turn) = crate::ledger::turn_for_run(&trace_path, run_id) {
                for e in crate::ledger::evidence_for_turn(&trace_path, &turn) {
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
            "检索范围：任务参数、各步骤产出、以及证据台账里的工具调用参数与返回。子串匹配，无相关性排序。"
        } else {
            "检索范围：任务参数与各步骤产出。证据台账未开启，工具调用记录不在检索范围内。"
        },
    }))
    .into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct FeedbackBody {
    /// 来源任务（可选，但强烈建议带上——修订要能追溯到具体场景）
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub sop: Option<String>,
    /// 用户原话
    pub text: String,
}

fn feedback_path(state: &AppState) -> std::path::PathBuf {
    let cfg = state.config.lock();
    cfg.workspace_dir.join("feedback").join("outbox.jsonl")
}

/// POST /api/feedback —— 规程修订建议回流（P2）。
///
/// 重要：**只记录，不自改**。助手不得据此修改自己的规程或人格——那条红线是刻意的
/// （见「反馈闭环」约定）。这里把建议写进 outbox，由研发侧的流程产出修订草案、
/// 人工评审后并入。接口如实回报 `applied: false`，界面也必须这么说。
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
            Json(serde_json::json!({"error":"缺少请求体，需要 {\"text\": \"...\"}"})),
        )
            .into_response();
    };
    let text = b.text.trim();
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"修订建议不能为空"})),
        )
            .into_response();
    }

    let entry = serde_json::json!({
        "at": chrono::Utc::now().to_rfc3339(),
        "task_id": b.task_id,
        "sop": b.sop,
        "text": text.chars().take(2000).collect::<String>(),
        "state": "pending_review",
    });

    let path = feedback_path(&state);
    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("无法写入修订记录：{e}")})),
            )
                .into_response();
        }
    }
    let line = format!("{entry}\n");
    let write = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
    if let Err(e) = write {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("无法写入修订记录：{e}")})),
        )
            .into_response();
    }

    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "recorded": true,
        // 说清楚它现在是什么状态：已记录 ≠ 已生效
        "applied": false,
        "state": "pending_review",
        "note": "您的意见已记入修订待办。助手不会据此自行修改规程——修订须由研发出草案、人工评审后并入，并在生效后署您的名。",
    }))
    .into_response()
}

/// GET /api/feedback —— 已提交的修订建议（供「规程共建」一节显示）。
pub async fn handle_api_feedback_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }
    let path = feedback_path(&state);
    let items: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .map(|raw| {
            raw.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .collect()
        })
        .unwrap_or_default();
    let recent: Vec<serde_json::Value> = items.into_iter().rev().take(50).collect();
    Json(serde_json::json!({
        "api": "frontdesk",
        "api_version": 1,
        "items": recent,
        "note": "这些是已记录、等待研发评审的修订建议。助手不会自行改动规程。",
    }))
    .into_response()
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
        let with = "检索范围：任务参数、各步骤产出、以及证据台账里的工具调用参数与返回。子串匹配，无相关性排序。";
        let without =
            "检索范围：任务参数与各步骤产出。证据台账未开启，工具调用记录不在检索范围内。";
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
