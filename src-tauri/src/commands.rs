//! Tauri commands: the bridge between the React UI and mint-core.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use base64::Engine as _;
use mint_core::chat::{extract_events, extract_memories, format_context, system_prompt, today};
use mint_core::conversations::{Conversation, ConversationStore, ConversationSummary};
use mint_core::engine::{MaintenanceReport, SyncCounts};
use mint_core::graph::GraphData;
use mint_core::ollama::{ChatMessage, Delta, Ollama};
use mint_core::record::{
    Memory, MemoryKind, NewMemory, SearchMode, SearchRequest, SearchResponse, Stats,
};
use mint_core::sync::{SyncClient, SyncConfig, SyncReport};
use mint_core::MemoryEngine;

/// Sync runtime: the online toggle (airplane mode) + server config + last result.
pub struct SyncRuntime {
    pub online: bool,
    pub cfg: SyncConfig,
    pub last_sync: Option<String>,
    pub last_report: Option<SyncReport>,
}

impl Default for SyncRuntime {
    fn default() -> Self {
        let url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into());
        let api_key = std::env::var("QDRANT_API_KEY").ok().filter(|k| !k.is_empty());
        Self {
            online: true,
            cfg: SyncConfig { url, api_key },
            last_sync: None,
            last_report: None,
        }
    }
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
    lock_engine(&state.engine)?.add(input).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn search_memories(
    state: State<AppState>,
    req: SearchRequest,
) -> Result<SearchResponse, String> {
    lock_engine(&state.engine)?.search(req).map_err(|e| e.to_string())
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
    pub online: bool,      // user toggle (airplane mode off = true)
    pub reachable: bool,   // server responded
    pub server_url: String,
    pub counts: SyncCounts,
    pub last_sync: Option<String>,
    pub last_report: Option<SyncReport>,
}

fn lock_sync<'a>(
    s: &'a Arc<Mutex<SyncRuntime>>,
) -> Result<std::sync::MutexGuard<'a, SyncRuntime>, String> {
    s.lock().map_err(|_| "sync runtime poisoned".to_string())
}

#[tauri::command]
pub fn sync_status(state: State<AppState>) -> Result<SyncStatus, String> {
    let (online, url, api_key, last_sync, last_report) = {
        let s = lock_sync(&state.sync)?;
        (
            s.online,
            s.cfg.url.clone(),
            s.cfg.api_key.clone(),
            s.last_sync.clone(),
            s.last_report.clone(),
        )
    };
    // Only ping the server when "online" (airplane mode off).
    let reachable = if online {
        SyncClient::new(SyncConfig { url: url.clone(), api_key }).reachable()
    } else {
        false
    };
    let counts = lock_engine(&state.engine)?
        .sync_counts()
        .map_err(|e| e.to_string())?;
    Ok(SyncStatus {
        online,
        reachable,
        server_url: url,
        counts,
        last_sync,
        last_report,
    })
}

#[tauri::command]
pub fn set_online(state: State<AppState>, online: bool) -> Result<(), String> {
    lock_sync(&state.sync)?.online = online;
    Ok(())
}

#[tauri::command]
pub fn set_server_url(state: State<AppState>, url: String) -> Result<(), String> {
    lock_sync(&state.sync)?.cfg.url = url;
    Ok(())
}

/// Run a two-way sync now. Requires online (airplane mode off) and a reachable server.
#[tauri::command]
pub async fn sync_now(state: State<'_, AppState>) -> Result<SyncReport, String> {
    let (online, cfg) = {
        let s = lock_sync(&state.sync)?;
        (s.online, s.cfg.clone())
    };
    if !online {
        return Err("offline (airplane mode is on)".into());
    }
    let engine = state.engine.clone();
    let sync = state.sync.clone();

    let report = tauri::async_runtime::spawn_blocking(move || {
        let client = SyncClient::new(cfg);
        if !client.reachable() {
            return Err("Qdrant server not reachable".to_string());
        }
        let eng = engine.lock().map_err(|_| "engine poisoned".to_string())?;
        eng.sync(&client).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("sync task failed: {e}"))??;

    if let Ok(mut s) = lock_sync(&sync) {
        s.last_sync = Some(chrono_now());
        s.last_report = Some(report.clone());
    }
    Ok(report)
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

    tauri::async_runtime::spawn_blocking(move || {
        run_turn(
            app,
            engine,
            conversations,
            ollama,
            chat_model,
            fast_model,
            conversation_id,
            message,
            history,
        )
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
    chat_model: String,
    fast_model: String,
    conversation_id: String,
    message: String,
    history: Vec<ChatMessage>,
) -> Result<ChatTurnResult, String> {
    // 1. Retrieve relevant memories (hybrid) for grounding.
    stage(&app, "retrieving", "searching device memory");
    let results = {
        let eng = lock_engine(&engine)?;
        eng.search(SearchRequest {
            query: message.clone(),
            mode: SearchMode::Hybrid,
            limit: 6,
            site_id: None,
            kind: None,
        })
        .map(|r| r.results)
        .unwrap_or_default()
    };
    // Entity hub nodes are bare names; keep them out of the chat grounding.
    let results: Vec<_> = results
        .into_iter()
        .filter(|r| r.memory.kind != MemoryKind::Entity)
        .collect();
    let retrieved: Vec<RetrievedItem> = results
        .iter()
        .map(|r| RetrievedItem {
            id: r.memory.id.clone(),
            title: r.memory.title.clone(),
            kind: kind_str(&r.memory.kind),
            score: r.score,
        })
        .collect();
    stage(&app, "retrieved", format!("{} relevant memories", retrieved.len()));

    // 2. Build the prompt and stream the answer + thinking.
    let context = format_context(&results);
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
