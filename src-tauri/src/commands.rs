//! Tauri commands: the bridge between the React UI and mint-core.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use base64::Engine as _;
use mint_core::chat::{
    extract_events, extract_memories, format_cloud_context, format_context, format_topic_overview,
    system_prompt, today,
};
use mint_core::conversations::{Conversation, ConversationStore, ConversationSummary};
use mint_core::engine::{
    MaintenanceReport, OrganizeReport, PolicySummary, SyncCounts, SyncDecision, TopicInfo,
};
use mint_core::graph::GraphData;
use mint_core::ollama::{ChatMessage, Delta, Ollama};
use mint_core::record::{
    Memory, MemoryKind, NewMemory, SearchRequest, SearchResponse, Stats,
};
use mint_core::sync::{SyncClient, SyncConfig, SyncReport};
use mint_core::MemoryEngine;

/// One entry in the sync activity log.
#[derive(Serialize, Clone)]
pub struct SyncEvent {
    pub at: String,
    /// "manual" | "reconnected" | "pending changes" | "periodic"
    pub trigger: String,
    pub ok: bool,
    pub report: Option<SyncReport>,
    pub error: Option<String>,
}

/// Sync runtime: the online toggle (airplane mode), auto-sync, pull mode,
/// server config, cached reachability, and the activity log.
pub struct SyncRuntime {
    pub online: bool,
    /// Sync automatically on reconnect / pending changes / periodically.
    pub auto: bool,
    /// Mirror everything (true) or tiered pull (false: consolidated knowledge
    /// + own memories; other devices' raw memories stay in the cloud).
    pub pull_all: bool,
    /// Last known server reachability (refreshed by the auto-sync loop), so
    /// chat can decide on cloud search without a network round-trip.
    pub reachable: bool,
    pub cfg: SyncConfig,
    pub last_sync: Option<String>,
    pub last_report: Option<SyncReport>,
    pub history: Vec<SyncEvent>,
    /// Where settings persist (`<data dir>/sync.json`); None = not persisted.
    pub path: Option<std::path::PathBuf>,
}

/// Sync settings saved across restarts. The API key is stored locally in the
/// app's data directory and is never sent back to the UI.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SyncSettings {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    online: Option<bool>,
    #[serde(default)]
    auto: Option<bool>,
    #[serde(default)]
    pull_all: Option<bool>,
}

impl SyncRuntime {
    /// Defaults, then saved settings; QDRANT_URL / QDRANT_API_KEY environment
    /// variables, when set, take precedence over saved values.
    pub fn load(data_dir: &std::path::Path) -> Self {
        let mut rt = Self::default();
        let path = data_dir.join("sync.json");
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(s) = serde_json::from_slice::<SyncSettings>(&bytes) {
                if std::env::var("QDRANT_URL").is_err() {
                    if let Some(u) = s.url.filter(|u| !u.trim().is_empty()) {
                        rt.cfg.url = u;
                    }
                }
                if std::env::var("QDRANT_API_KEY").is_err() && s.api_key.is_some() {
                    rt.cfg.api_key = s.api_key.filter(|k| !k.is_empty());
                }
                rt.online = s.online.unwrap_or(rt.online);
                rt.auto = s.auto.unwrap_or(rt.auto);
                rt.pull_all = s.pull_all.unwrap_or(rt.pull_all);
            }
        }
        rt.path = Some(path);
        rt
    }

    pub fn save(&self) {
        let Some(p) = &self.path else { return };
        let s = SyncSettings {
            url: Some(self.cfg.url.clone()),
            api_key: self.cfg.api_key.clone(),
            online: Some(self.online),
            auto: Some(self.auto),
            pull_all: Some(self.pull_all),
        };
        if let Ok(bytes) = serde_json::to_vec_pretty(&s) {
            let _ = std::fs::write(p, bytes);
        }
    }
}

impl Default for SyncRuntime {
    fn default() -> Self {
        let url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into());
        let api_key = std::env::var("QDRANT_API_KEY").ok().filter(|k| !k.is_empty());
        Self {
            online: true,
            auto: true,
            pull_all: false,
            reachable: false,
            cfg: SyncConfig {
                url,
                api_key,
                ..Default::default()
            },
            last_sync: None,
            last_report: None,
            history: Vec::new(),
            path: None,
        }
    }
}

const SYNC_HISTORY_MAX: usize = 25;

/// Run one sync (manual or automatic) and record it in the activity log.
/// Shared by the `sync_now` command and the background auto-sync loop.
pub fn run_sync(
    engine: &Arc<Mutex<MemoryEngine>>,
    sync: &Arc<Mutex<SyncRuntime>>,
    trigger: &str,
) -> Result<SyncReport, String> {
    let (online, cfg, pull_all) = {
        let s = lock_sync(sync)?;
        (s.online, s.cfg.clone(), s.pull_all)
    };
    if !online {
        return Err("offline (airplane mode is on)".into());
    }
    let client = SyncClient::new(cfg);
    let reachable = client.reachable();
    let result = if !reachable {
        Err("Qdrant server not reachable".to_string())
    } else {
        let eng = engine.lock().map_err(|_| "engine poisoned".to_string())?;
        eng.sync(&client, pull_all).map_err(|e| e.to_string())
    };
    if let Ok(mut s) = lock_sync(sync) {
        s.reachable = reachable;
        let active = match &result {
            Ok(r) => r.pushed + r.pulled + r.conflicts + r.retracted > 0,
            Err(_) => true,
        };
        // Manual syncs are always logged; automatic ones only when something
        // happened (keeps the log meaningful).
        if trigger == "manual" || active {
            s.history.insert(
                0,
                SyncEvent {
                    at: chrono_now(),
                    trigger: trigger.to_string(),
                    ok: result.is_ok(),
                    report: result.as_ref().ok().cloned(),
                    error: result.as_ref().err().cloned(),
                },
            );
            s.history.truncate(SYNC_HISTORY_MAX);
        }
        if let Ok(r) = &result {
            s.last_sync = Some(chrono_now());
            s.last_report = Some(r.clone());
        }
    }
    result
}

/// Seconds since the Unix epoch (for the chat-activity marker).
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Background worker, one tick every 10 s, two independent jobs:
///   1. Auto-sync: when online and enabled, sync if the server just became
///      reachable, if there are pending changes, or every 2 minutes (to
///      receive other devices' changes). All network calls time out quickly.
///   2. Entity enrichment: link memories that lack entities (pulled from
///      another device, or stored before entities existed) using the local
///      LLM. The LLM call runs OUTSIDE the engine lock, a few items per tick,
///      and only when chat has been idle for 20 s, so it never slows a chat.
pub fn spawn_background(
    app: AppHandle,
    engine: Arc<Mutex<MemoryEngine>>,
    sync: Arc<Mutex<SyncRuntime>>,
    ollama: Arc<Ollama>,
    fast_model: String,
    last_chat: Arc<std::sync::atomic::AtomicU64>,
) {
    std::thread::spawn(move || {
        let tick = std::time::Duration::from_secs(10);
        let periodic = std::time::Duration::from_secs(120);
        let mut was_reachable = false;
        let mut last_run: Option<std::time::Instant> = None;
        let mut enrich_idle_ticks = 0u32;
        loop {
            std::thread::sleep(tick);

            // ---- 1. auto-sync ------------------------------------------------
            let (online, auto, cfg) = match sync.lock() {
                Ok(s) => (s.online, s.auto, s.cfg.clone()),
                Err(_) => continue,
            };
            if online && auto {
                let reachable = SyncClient::new(cfg).reachable();
                if let Ok(mut s) = sync.lock() {
                    s.reachable = reachable;
                }
                if reachable {
                    let reconnected = !was_reachable;
                    was_reachable = true;
                    let pending = engine
                        .lock()
                        .ok()
                        .and_then(|e| e.sync_counts().ok())
                        .map(|c| c.pending)
                        .unwrap_or(0);
                    let due = last_run.map(|t| t.elapsed() >= periodic).unwrap_or(true);
                    let trigger = if reconnected {
                        Some("reconnected")
                    } else if pending > 0 {
                        Some("pending changes")
                    } else if due {
                        Some("periodic")
                    } else {
                        None
                    };
                    if let Some(trigger) = trigger {
                        let result = run_sync(&engine, &sync, trigger);
                        last_run = Some(std::time::Instant::now());
                        let _ = app.emit("sync:done", result.is_ok());
                        if result.map(|r| r.pulled > 0).unwrap_or(false) {
                            enrich_idle_ticks = 0; // new content: look now
                        }
                    }
                } else {
                    was_reachable = false;
                }
            } else {
                was_reachable = false;
            }

            // ---- 2. entity enrichment -----------------------------------------
            // After an empty backlog, re-check only every 6th tick (~1 min).
            if enrich_idle_ticks > 0 {
                enrich_idle_ticks -= 1;
                continue;
            }
            let chat_idle =
                now_secs().saturating_sub(last_chat.load(std::sync::atomic::Ordering::Relaxed)) >= 20;
            if !chat_idle || fast_model.is_empty() || !ollama.is_up() {
                continue;
            }
            let backlog = engine
                .lock()
                .ok()
                .and_then(|e| e.entity_backlog(3).ok())
                .unwrap_or_default();
            if backlog.is_empty() {
                enrich_idle_ticks = 6;
                continue;
            }
            let mut linked = 0;
            for (id, text) in backlog {
                // Extraction (slow, LLM) happens without the engine lock.
                let ents = match mint_core::entities::extract(&ollama, &fast_model, &text) {
                    Ok(e) => e,
                    Err(_) => break, // Ollama busy/unavailable: try next tick
                };
                if let Ok(e) = engine.lock() {
                    if e.link_extracted_entities(&id, &ents).is_ok() {
                        linked += 1;
                    }
                }
            }
            if linked > 0 {
                let _ = app.emit("memory:enriched", linked);
            }
        }
    });
}

/// Similarity above which an auto-captured memory is treated as a duplicate.
const DEDUP_THRESHOLD: f32 = 0.90;

/// Shared app state. Arc so the chat command can move handles into a blocking task.
pub struct AppState {
    pub engine: Arc<Mutex<MemoryEngine>>,
    pub conversations: Arc<Mutex<ConversationStore>>,
    pub ollama: Arc<Ollama>,
    pub sync: Arc<Mutex<SyncRuntime>>,
    pub chat_model: String,
    pub fast_model: String,
    /// Unix seconds of the last chat activity (background enrichment waits
    /// until chat has been idle, so it never competes with a turn for the LLM).
    pub last_chat: Arc<std::sync::atomic::AtomicU64>,
}

fn lock_convos(
    store: &Arc<Mutex<ConversationStore>>,
) -> Result<std::sync::MutexGuard<'_, ConversationStore>, String> {
    store.lock().map_err(|_| "conversation store poisoned".to_string())
}

fn lock_engine(
    engine: &Arc<Mutex<MemoryEngine>>,
) -> Result<std::sync::MutexGuard<'_, MemoryEngine>, String> {
    engine
        .lock()
        .map_err(|_| "memory engine mutex poisoned".to_string())
}

fn kind_str(kind: &MemoryKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

// ---- memory commands -----------------------------------------------------

#[tauri::command]
pub fn add_memory(state: State<AppState>, input: NewMemory) -> Result<Memory, String> {
    let eng = lock_engine(&state.engine)?;
    let m = eng.add(input).map_err(|e| e.to_string())?;
    // Version chains with the deterministic rules only: this command runs on
    // the UI thread, so no LLM call here.
    let _ = eng.link_versions(&m.id, "");
    eng.get(&m.id)
        .ok()
        .flatten()
        .map(Ok)
        .unwrap_or(Ok(m))
}

#[tauri::command]
pub fn search_memories(
    state: State<AppState>,
    req: SearchRequest,
) -> Result<SearchResponse, String> {
    let eng = lock_engine(&state.engine)?;
    let resp = eng.search(req).map_err(|e| e.to_string())?;
    let ids: Vec<String> = resp.results.iter().map(|r| r.memory.id.clone()).collect();
    eng.bump_access(&ids);
    Ok(resp)
}

#[tauri::command]
pub fn list_memories(state: State<AppState>) -> Result<Vec<Memory>, String> {
    lock_engine(&state.engine)?
        .list_active()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_memory(state: State<AppState>, id: String) -> Result<(), String> {
    lock_engine(&state.engine)?.delete(&id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn stats(state: State<AppState>) -> Result<Stats, String> {
    lock_engine(&state.engine)?.stats().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn graph_data(state: State<AppState>) -> Result<GraphData, String> {
    lock_engine(&state.engine)?.graph_data().map_err(|e| e.to_string())
}

// ---- maintenance (consolidation + decay) ---------------------------------

/// Run consolidation (entity summaries) + decay (archive stale chunks).
#[tauri::command]
pub async fn run_maintenance(state: State<'_, AppState>) -> Result<MaintenanceReport, String> {
    let engine = state.engine.clone();
    let model = state.fast_model.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let eng = engine.lock().map_err(|_| "engine poisoned".to_string())?;
        eng.run_maintenance(&model).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("maintenance task failed: {e}"))?
}

#[tauri::command]
pub fn clear_archive(state: State<AppState>) -> Result<usize, String> {
    lock_engine(&state.engine)?
        .clear_archive()
        .map_err(|e| e.to_string())
}

// ---- topics (schema layer: auto-routing + the user's overrides) ----------

#[tauri::command]
pub fn list_topics(state: State<AppState>) -> Result<Vec<TopicInfo>, String> {
    lock_engine(&state.engine)?
        .list_topics()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn rename_topic(state: State<AppState>, id: String, name: String) -> Result<(), String> {
    lock_engine(&state.engine)?
        .rename_topic(&id, &name)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn merge_topics(state: State<AppState>, from: String, into: String) -> Result<(), String> {
    lock_engine(&state.engine)?
        .merge_topics(&from, &into)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn move_to_topic(
    state: State<AppState>,
    memory_id: String,
    topic_id: String,
) -> Result<(), String> {
    lock_engine(&state.engine)?
        .move_to_topic(&memory_id, &topic_id)
        .map_err(|e| e.to_string())
}

/// Re-distill one topic's rolling summary (local LLM).
#[tauri::command]
pub async fn refresh_topic(state: State<'_, AppState>, id: String) -> Result<TopicInfo, String> {
    let engine = state.engine.clone();
    let model = state.fast_model.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let eng = engine.lock().map_err(|_| "engine poisoned".to_string())?;
        eng.refresh_topic(&id, &model).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("topic task failed: {e}"))?
}

/// File every not-yet-organized memory/document into topics, re-home
/// stragglers, and summarize (local LLM).
#[tauri::command]
pub async fn organize_topics(state: State<'_, AppState>) -> Result<OrganizeReport, String> {
    let engine = state.engine.clone();
    let model = state.fast_model.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let eng = engine.lock().map_err(|_| "engine poisoned".to_string())?;
        eng.organize(&model).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("organize task failed: {e}"))?
}

// ---- timeline / calendar -------------------------------------------------

#[tauri::command]
pub fn list_scheduled(state: State<AppState>) -> Result<Vec<Memory>, String> {
    lock_engine(&state.engine)?
        .list_scheduled()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_task_done(state: State<AppState>, id: String, done: bool) -> Result<(), String> {
    lock_engine(&state.engine)?
        .set_done(&id, done)
        .map_err(|e| e.to_string())
}

// ---- vault (documents) ---------------------------------------------------

#[derive(Serialize, Clone)]
pub struct DocIngestResult {
    pub id: String,
    pub title: String,
    pub chunks: usize,
}

/// Ingest a document from base64-encoded bytes (the UI reads the file locally
/// and sends its content). Parses, chunks, embeds, and stores it — all offline.
#[tauri::command]
pub async fn ingest_document(
    state: State<'_, AppState>,
    name: String,
    data_base64: String,
) -> Result<DocIngestResult, String> {
    let engine = state.engine.clone();
    let fast_model = state.fast_model.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_base64.as_bytes())
            .map_err(|e| format!("invalid file data: {e}"))?;
        let eng = engine.lock().map_err(|_| "engine poisoned".to_string())?;
        let (doc, chunks) = eng
            .ingest_document(&name, &bytes, &fast_model)
            .map_err(|e| e.to_string())?;
        // Rolling summary (and a good name) for the subject this document
        // joined, so chat and routing immediately know what it is about.
        if let Ok(Some(stored)) = eng.get(&doc.id) {
            if let Some(tid) = stored.topic_id {
                let _ = eng.refresh_topic_if_due(&tid, &fast_model, 1);
            }
        }
        Ok(DocIngestResult {
            id: doc.id,
            title: doc.title,
            chunks,
        })
    })
    .await
    .map_err(|e| format!("ingest task failed: {e}"))?
}

#[tauri::command]
pub fn list_documents(state: State<AppState>) -> Result<Vec<Memory>, String> {
    lock_engine(&state.engine)?
        .list_documents()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_document(state: State<AppState>, id: String) -> Result<(), String> {
    lock_engine(&state.engine)?
        .delete_document(&id)
        .map_err(|e| e.to_string())
}

// ---- sync (edge <-> cloud) -----------------------------------------------

#[derive(Serialize, Clone)]
pub struct SyncStatus {
    pub online: bool,    // user toggle (airplane mode off = true)
    pub reachable: bool, // server responded
    pub server_url: String,
    pub counts: SyncCounts,
    pub last_sync: Option<String>,
    pub last_report: Option<SyncReport>,
    pub auto: bool,
    pub pull_all: bool,
    pub device_id: String,
    pub history: Vec<SyncEvent>,
    /// Whether an API key is configured (the key itself is never exposed).
    pub has_api_key: bool,
}

fn lock_sync<'a>(
    s: &'a Arc<Mutex<SyncRuntime>>,
) -> Result<std::sync::MutexGuard<'a, SyncRuntime>, String> {
    s.lock().map_err(|_| "sync runtime poisoned".to_string())
}

/// Sync status (async: the reachability ping must never block the UI thread).
#[tauri::command]
pub async fn sync_status(state: State<'_, AppState>) -> Result<SyncStatus, String> {
    let engine = state.engine.clone();
    let sync = state.sync.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (online, cfg, last_sync, last_report, auto, pull_all, history) = {
            let s = lock_sync(&sync)?;
            (
                s.online,
                s.cfg.clone(),
                s.last_sync.clone(),
                s.last_report.clone(),
                s.auto,
                s.pull_all,
                s.history.clone(),
            )
        };
        let url = cfg.url.clone();
        let has_api_key = cfg.api_key.is_some();
        // Only ping the server when "online" (airplane mode off).
        let reachable = online && SyncClient::new(cfg).reachable();
        if let Ok(mut s) = lock_sync(&sync) {
            s.reachable = reachable;
        }
        let (counts, device_id) = {
            let eng = lock_engine(&engine)?;
            (eng.sync_counts().map_err(|e| e.to_string())?, eng.device_id())
        };
        Ok(SyncStatus {
            online,
            reachable,
            server_url: url,
            counts,
            last_sync,
            last_report,
            auto,
            pull_all,
            device_id,
            history,
            has_api_key,
        })
    })
    .await
    .map_err(|e| format!("status task failed: {e}"))?
}

#[tauri::command]
pub fn set_auto_sync(state: State<AppState>, enabled: bool) -> Result<(), String> {
    let mut s = lock_sync(&state.sync)?;
    s.auto = enabled;
    s.save();
    Ok(())
}

#[tauri::command]
pub fn set_pull_all(state: State<AppState>, enabled: bool) -> Result<(), String> {
    let mut s = lock_sync(&state.sync)?;
    s.pull_all = enabled;
    s.save();
    Ok(())
}

/// What stays on the device and why (sync policy overview).
#[tauri::command]
pub async fn policy_summary(state: State<'_, AppState>) -> Result<PolicySummary, String> {
    let engine = state.engine.clone();
    tauri::async_runtime::spawn_blocking(move || {
        lock_engine(&engine)?.policy_summary().map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("policy task failed: {e}"))?
}

/// User override: allow a memory to sync, or keep it on this device.
#[tauri::command]
pub fn set_sync_override(
    state: State<AppState>,
    id: String,
    share: bool,
) -> Result<SyncDecision, String> {
    lock_engine(&state.engine)?
        .set_sync_override(&id, share)
        .map_err(|e| e.to_string())
}

/// A memory's version chain, oldest first.
#[tauri::command]
pub fn version_chain(state: State<AppState>, id: String) -> Result<Vec<Memory>, String> {
    lock_engine(&state.engine)?
        .version_chain(&id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_online(state: State<AppState>, online: bool) -> Result<(), String> {
    let mut s = lock_sync(&state.sync)?;
    s.online = online;
    s.save();
    Ok(())
}

#[tauri::command]
pub fn set_server_url(state: State<AppState>, url: String) -> Result<(), String> {
    let mut s = lock_sync(&state.sync)?;
    s.cfg.url = url.trim().to_string();
    s.save();
    Ok(())
}

/// Set (or clear, with an empty string) the Qdrant API key, e.g. for Qdrant
/// Cloud. Stored locally in the app's data directory; never returned to the UI.
#[tauri::command]
pub fn set_api_key(state: State<AppState>, key: String) -> Result<(), String> {
    let mut s = lock_sync(&state.sync)?;
    let k = key.trim().to_string();
    s.cfg.api_key = if k.is_empty() { None } else { Some(k) };
    s.save();
    Ok(())
}

/// Run a two-way sync now. Requires online (airplane mode off) and a reachable server.
#[tauri::command]
pub async fn sync_now(state: State<'_, AppState>) -> Result<SyncReport, String> {
    let engine = state.engine.clone();
    let sync = state.sync.clone();
    tauri::async_runtime::spawn_blocking(move || run_sync(&engine, &sync, "manual"))
        .await
        .map_err(|e| format!("sync task failed: {e}"))?
}

fn chrono_now() -> String {
    // Lightweight RFC3339-ish timestamp without pulling chrono into this crate.
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}

// ---- chat status ---------------------------------------------------------

#[derive(Serialize, Clone)]
pub struct ChatStatus {
    pub ollama_up: bool,
    pub models: Vec<String>,
    pub chat_model: String,
    pub fast_model: String,
}

// ---- conversation commands ----------------------------------------------

#[tauri::command]
pub fn list_conversations(state: State<AppState>) -> Result<Vec<ConversationSummary>, String> {
    Ok(lock_convos(&state.conversations)?.list())
}

#[tauri::command]
pub fn get_conversation(state: State<AppState>, id: String) -> Result<Option<Conversation>, String> {
    Ok(lock_convos(&state.conversations)?.get(&id))
}

#[tauri::command]
pub fn create_conversation(state: State<AppState>) -> Result<Conversation, String> {
    lock_convos(&state.conversations)?
        .create(None)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn rename_conversation(
    state: State<AppState>,
    id: String,
    title: String,
) -> Result<(), String> {
    lock_convos(&state.conversations)?
        .rename(&id, title)
        .map_err(|e| e.to_string())
}

/// Delete a conversation. When `delete_memories` is true, also removes the
/// memories captured during it.
#[tauri::command]
pub fn delete_conversation(
    state: State<AppState>,
    id: String,
    delete_memories: bool,
) -> Result<(), String> {
    let memory_ids = lock_convos(&state.conversations)?
        .delete(&id)
        .map_err(|e| e.to_string())?;
    if delete_memories {
        let eng = lock_engine(&state.engine)?;
        for mid in memory_ids {
            let _ = eng.delete(&mid);
        }
    }
    Ok(())
}

#[tauri::command]
pub fn chat_status(state: State<AppState>) -> ChatStatus {
    let ollama_up = state.ollama.is_up();
    let models = if ollama_up {
        state.ollama.list_models().unwrap_or_default()
    } else {
        Vec::new()
    };
    ChatStatus {
        ollama_up,
        models,
        chat_model: state.chat_model.clone(),
        fast_model: state.fast_model.clone(),
    }
}

// ---- chat turn -----------------------------------------------------------

#[derive(Serialize, Clone)]
struct StageEvent {
    stage: String,
    detail: String,
}

#[derive(Serialize, Clone)]
struct TokenEvent {
    channel: String, // "thinking" | "answer"
    text: String,
}

#[derive(Serialize, Clone)]
pub struct RetrievedItem {
    id: String,
    title: String,
    kind: String,
    score: f32,
    /// "device" (on-device memory) or "cloud" (another device's memory found
    /// through cloud search while online).
    source: String,
}

#[derive(Serialize, Clone)]
pub struct CapturedItem {
    id: String,
    title: String,
    kind: String,
}

#[derive(Serialize, Clone)]
struct ScheduledEvent {
    id: String,
    title: String,
    due_at: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct ChatTurnResult {
    answer: String,
    retrieved: Vec<RetrievedItem>,
    captured: Vec<CapturedItem>,
}

fn stage(app: &AppHandle, stage: &str, detail: impl Into<String>) {
    let _ = app.emit(
        "chat:stage",
        StageEvent {
            stage: stage.to_string(),
            detail: detail.into(),
        },
    );
}

/// One chat turn: retrieve relevant memory (RAG) -> stream the model's thinking
/// and answer -> auto-capture durable memories from the turn. All local.
#[tauri::command]
pub async fn chat(
    app: AppHandle,
    state: State<'_, AppState>,
    conversation_id: String,
    message: String,
    history: Vec<ChatMessage>,
) -> Result<ChatTurnResult, String> {
    let engine = state.engine.clone();
    let conversations = state.conversations.clone();
    let ollama = state.ollama.clone();
    let chat_model = state.chat_model.clone();
    let fast_model = state.fast_model.clone();
    let sync = state.sync.clone();
    let last_chat = state.last_chat.clone();

    tauri::async_runtime::spawn_blocking(move || {
        last_chat.store(now_secs(), std::sync::atomic::Ordering::Relaxed);
        let result = run_turn(
            app,
            engine,
            conversations,
            ollama,
            sync,
            chat_model,
            fast_model,
            conversation_id,
            message,
            history,
        );
        last_chat.store(now_secs(), std::sync::atomic::Ordering::Relaxed);
        result
    })
    .await
    .map_err(|e| format!("chat task failed: {e}"))?
}

#[allow(clippy::too_many_arguments)]
fn run_turn(
    app: AppHandle,
    engine: Arc<Mutex<MemoryEngine>>,
    conversations: Arc<Mutex<ConversationStore>>,
    ollama: Arc<Ollama>,
    sync: Arc<Mutex<SyncRuntime>>,
    chat_model: String,
    fast_model: String,
    conversation_id: String,
    message: String,
    history: Vec<ChatMessage>,
) -> Result<ChatTurnResult, String> {
    // 1. Retrieve relevant memories for grounding: topic-aware retrieval
    //    (subject first, then details) plus the subjects' rolling summaries.
    stage(&app, "retrieving", "searching device memory");
    let ctx = {
        let eng = lock_engine(&engine)?;
        eng.retrieve_context(&message, 6).ok()
    };
    let (results, topics) = match ctx {
        Some(c) => (c.results, c.topics),
        None => (Vec::new(), Vec::new()),
    };
    // Entity hub nodes are bare names; keep them out of the chat grounding.
    let results: Vec<_> = results
        .into_iter()
        .filter(|r| r.memory.kind != MemoryKind::Entity)
        .collect();
    let overview: Vec<(String, String)> = topics
        .iter()
        .filter(|t| !t.summary.is_empty())
        .map(|t| (t.name.clone(), t.summary.clone()))
        .collect();
    // 1b. Online: also ask the cloud for knowledge from other devices that is
    //     not on this device (tiered pull leaves their raw memories there).
    //     Uses the cached reachability and a short timeout; offline = skipped.
    let (online, reachable, cfg) = lock_sync(&sync)
        .map(|s| (s.online, s.reachable, s.cfg.clone()))
        .unwrap_or((false, false, SyncConfig::default()));
    let cloud: Vec<_> = if online && reachable {
        stage(&app, "retrieving", "asking your other devices (cloud)");
        let eng = lock_engine(&engine)?;
        eng.cloud_search(&SyncClient::new(cfg), &message, 2)
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut retrieved: Vec<RetrievedItem> = results
        .iter()
        .map(|r| RetrievedItem {
            id: r.memory.id.clone(),
            title: r.memory.title.clone(),
            kind: kind_str(&r.memory.kind),
            score: r.score,
            source: "device".into(),
        })
        .collect();
    retrieved.extend(cloud.iter().map(|r| RetrievedItem {
        id: r.memory.id.clone(),
        title: r.memory.title.clone(),
        kind: kind_str(&r.memory.kind),
        score: r.score,
        source: "cloud".into(),
    }));
    let subject = topics
        .first()
        .map(|t| format!(" in {}", t.name))
        .unwrap_or_default();
    let cloud_note = if cloud.is_empty() {
        String::new()
    } else {
        format!(" + {} from the cloud", cloud.len())
    };
    stage(
        &app,
        "retrieved",
        format!("{} relevant memories{subject}{cloud_note}", results.len()),
    );
    {
        // Retrieval counts as usage -> feeds salience.
        let eng = lock_engine(&engine)?;
        let ids: Vec<String> = results.iter().map(|r| r.memory.id.clone()).collect();
        eng.bump_access(&ids);
    }

    // 2. Build the prompt and stream the answer + thinking.
    let context = format!(
        "{}{}{}",
        format_topic_overview(&overview),
        format_context(&results),
        format_cloud_context(&cloud)
    );
    let mut messages = vec![ChatMessage::system(system_prompt(&context))];
    messages.extend(history);
    messages.push(ChatMessage::user(message.clone()));

    stage(&app, "thinking", &chat_model);
    let app_cb = app.clone();
    let mut answering = false;
    let answer = ollama
        .chat_stream(&chat_model, &messages, true, |delta| match delta {
            Delta::Thinking(t) => {
                let _ = app_cb.emit(
                    "chat:token",
                    TokenEvent { channel: "thinking".into(), text: t.to_string() },
                );
            }
            Delta::Answer(a) => {
                if !answering {
                    answering = true;
                    stage(&app_cb, "answering", "");
                }
                let _ = app_cb.emit(
                    "chat:token",
                    TokenEvent { channel: "answer".into(), text: a.to_string() },
                );
            }
        })
        .map_err(|e| e.to_string())?;

    // 3. Auto-capture durable memories from this turn (deduped).
    stage(&app, "extracting", "distilling memories");
    let extracted = extract_memories(&ollama, &fast_model, &message, &answer).unwrap_or_default();
    let mut captured = Vec::new();
    let mut captured_ids = Vec::new();
    if !extracted.is_empty() {
        let eng = lock_engine(&engine)?;
        for nm in extracted {
            // add_if_novel skips near-duplicates so restated facts don't pile up.
            if let Ok(Some(m)) = eng.add_if_novel(nm, DEDUP_THRESHOLD) {
                // Link the new memory to the entities it mentions.
                let _ = eng.attach_entities(&m.id, &m.text, &fast_model);
                // File it under a topic (schema layer): join the nearest subject
                // or start a new one. Keeps a research thread coherent. A new
                // subject gets its summary right away; an existing one rolls
                // its summary forward every few new memories.
                if let Some(tid) = eng.route_and_assign(&m.id, &m.text, &fast_model) {
                    let _ = eng.refresh_topic_if_due(&tid, &fast_model, 3);
                }
                // Evolving memory: if this revises an earlier fact ("exam moved
                // to Monday"), chain it and mark the old one outdated. The chat
                // model judges (benchmark: best recall at precision 1.0).
                if let Ok(Some(_)) = eng.link_versions(&m.id, &chat_model) {
                    stage(&app, "updated", format!("updated an earlier memory: {}", m.title));
                }
                let item = CapturedItem {
                    id: m.id.clone(),
                    title: m.title.clone(),
                    kind: kind_str(&m.kind),
                };
                let _ = app.emit("chat:captured", item.clone());
                captured_ids.push(m.id.clone());
                captured.push(item);
            }
        }
    }

    // 3b. Detect tasks / deadlines / dated events and add them to the timeline.
    let events = extract_events(&ollama, &fast_model, &message, &today()).unwrap_or_default();
    if !events.is_empty() {
        stage(&app, "scheduling", "adding to timeline");
        let eng = lock_engine(&engine)?;
        for nm in events {
            if let Ok(m) = eng.add(nm) {
                // A rescheduled deadline replaces the old date on the timeline.
                let _ = eng.link_versions(&m.id, &chat_model);
                let _ = app.emit(
                    "chat:scheduled",
                    ScheduledEvent {
                        id: m.id.clone(),
                        title: m.title.clone(),
                        due_at: m.due_at.clone(),
                    },
                );
            }
        }
    }

    // 4. Persist the turn to the conversation (links captured memory ids).
    if !conversation_id.is_empty() {
        if let Ok(mut store) = lock_convos(&conversations) {
            let _ = store.append_turn(&conversation_id, &message, &answer, &captured_ids);
        }
    }

    stage(&app, "done", format!("captured {}", captured.len()));

    Ok(ChatTurnResult {
        answer,
        retrieved,
        captured,
    })
}
