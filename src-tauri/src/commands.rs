//! Tauri commands: the bridge between the React UI and mint-core.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use base64::Engine as _;
use mint_core::chat::{extract_memories, format_context, system_prompt};
use mint_core::conversations::{Conversation, ConversationStore, ConversationSummary};
use mint_core::graph::GraphData;
use mint_core::ollama::{ChatMessage, Delta, Ollama};
use mint_core::record::{
    Memory, MemoryKind, NewMemory, SearchMode, SearchRequest, SearchResponse, Stats,
};
use mint_core::MemoryEngine;

/// Similarity above which an auto-captured memory is treated as a duplicate.
const DEDUP_THRESHOLD: f32 = 0.90;

/// Shared app state. Arc so the chat command can move handles into a blocking task.
pub struct AppState {
    pub engine: Arc<Mutex<MemoryEngine>>,
    pub conversations: Arc<Mutex<ConversationStore>>,
    pub ollama: Arc<Ollama>,
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
    lock_engine(&state.engine)?.list().map_err(|e| e.to_string())
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
    tauri::async_runtime::spawn_blocking(move || {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_base64.as_bytes())
            .map_err(|e| format!("invalid file data: {e}"))?;
        let eng = engine.lock().map_err(|_| "engine poisoned".to_string())?;
        let (doc, chunks) = eng.ingest_document(&name, &bytes).map_err(|e| e.to_string())?;
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
