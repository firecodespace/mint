//! Tauri commands: the bridge between the React UI and the Rust memory engine.

use std::sync::Mutex;

use tauri::State;

use crate::memory::record::{Memory, NewMemory, SearchRequest, SearchResponse, Stats};
use crate::memory::MemoryEngine;

/// App state: the memory engine behind a Mutex (embedding needs &mut internally,
/// and Phase 1 favors simplicity over read parallelism).
pub struct AppState {
    pub engine: Mutex<MemoryEngine>,
}

fn lock<'a>(state: &'a State<AppState>) -> Result<std::sync::MutexGuard<'a, MemoryEngine>, String> {
    state
        .engine
        .lock()
        .map_err(|_| "memory engine mutex poisoned".to_string())
}

#[tauri::command]
pub fn add_memory(state: State<AppState>, input: NewMemory) -> Result<Memory, String> {
    lock(&state)?.add(input).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn search_memories(
    state: State<AppState>,
    req: SearchRequest,
) -> Result<SearchResponse, String> {
    lock(&state)?.search(req).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_memories(state: State<AppState>) -> Result<Vec<Memory>, String> {
    lock(&state)?.list().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_memory(state: State<AppState>, id: String) -> Result<(), String> {
    lock(&state)?.delete(&id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn stats(state: State<AppState>) -> Result<Stats, String> {
    lock(&state)?.stats().map_err(|e| e.to_string())
}
