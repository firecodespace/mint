//! Tauri commands: the bridge between the React UI and mint-core.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use mint_core::chat::{extract_memories, format_context, system_prompt};
use mint_core::ollama::{ChatMessage, Delta, Ollama};
use mint_core::record::{
    Memory, MemoryKind, NewMemory, SearchMode, SearchRequest, SearchResponse, Stats,
};
use mint_core::MemoryEngine;

/// Shared app state. Arc so the chat command can move handles into a blocking task.
pub struct AppState {
    pub engine: Arc<Mutex<MemoryEngine>>,
    pub ollama: Arc<Ollama>,
    pub chat_model: String,
    pub fast_model: String,
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

// ---- chat status ---------------------------------------------------------

#[derive(Serialize, Clone)]
pub struct ChatStatus {
    pub ollama_up: bool,
    pub models: Vec<String>,
    pub chat_model: String,
    pub fast_model: String,
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
    message: String,
    history: Vec<ChatMessage>,
) -> Result<ChatTurnResult, String> {
    let engine = state.engine.clone();
    let ollama = state.ollama.clone();
    let chat_model = state.chat_model.clone();
    let fast_model = state.fast_model.clone();

    tauri::async_runtime::spawn_blocking(move || {
        run_turn(app, engine, ollama, chat_model, fast_model, message, history)
    })
    .await
    .map_err(|e| format!("chat task failed: {e}"))?
}

fn run_turn(
    app: AppHandle,
    engine: Arc<Mutex<MemoryEngine>>,
    ollama: Arc<Ollama>,
    chat_model: String,
    fast_model: String,
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

    // 3. Auto-capture durable memories from this turn.
    stage(&app, "extracting", "distilling memories");
    let extracted = extract_memories(&ollama, &fast_model, &message, &answer).unwrap_or_default();
    let mut captured = Vec::new();
    if !extracted.is_empty() {
        let eng = lock_engine(&engine)?;
        for nm in extracted {
            if let Ok(m) = eng.add(nm) {
                let item = CapturedItem {
                    id: m.id.clone(),
                    title: m.title.clone(),
                    kind: kind_str(&m.kind),
                };
                let _ = app.emit("chat:captured", item.clone());
                captured.push(item);
            }
        }
    }
    stage(&app, "done", format!("captured {}", captured.len()));

    Ok(ChatTurnResult {
        answer,
        retrieved,
        captured,
    })
}
