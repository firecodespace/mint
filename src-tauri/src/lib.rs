mod commands;

use std::sync::{Arc, Mutex};

use tauri::Manager;

use commands::{AppState, SyncRuntime};
use mint_core::ollama::{Ollama, DEFAULT_CHAT_MODEL, DEFAULT_FAST_MODEL};
use mint_core::{ConversationStore, MemoryEngine};

/// Pick a model: env override -> preferred if pulled -> first available -> default.
fn choose_model(env_key: &str, preferred: &str, available: &[String]) -> String {
    if let Ok(m) = std::env::var(env_key) {
        if !m.trim().is_empty() {
            return m;
        }
    }
    if available.iter().any(|m| m == preferred) {
        return preferred.to_string();
    }
    available.first().cloned().unwrap_or_else(|| preferred.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Where the on-device memory lives. MINT_DATA_DIR overrides the
            // default (used in dev to keep the store inside the project).
            let data_dir = match std::env::var("MINT_DATA_DIR") {
                Ok(dir) if !dir.trim().is_empty() => std::path::PathBuf::from(dir),
                _ => app
                    .path()
                    .app_data_dir()
                    .map_err(|e| format!("no app data dir: {e}"))?
                    .join("mint-memory"),
            };
            log::info!("Mint data dir: {}", data_dir.display());

            let engine = MemoryEngine::open(&data_dir).map_err(|e| e.to_string())?;
            let conversations = ConversationStore::open(&data_dir).map_err(|e| e.to_string())?;

            let ollama = Ollama::new();
            let available = ollama.list_models().unwrap_or_default();
            let chat_model = choose_model("MINT_CHAT_MODEL", DEFAULT_CHAT_MODEL, &available);
            let fast_model = choose_model("MINT_FAST_MODEL", DEFAULT_FAST_MODEL, &available);
            log::info!("Ollama chat model: {chat_model} | fast model: {fast_model}");

            app.manage(AppState {
                engine: Arc::new(Mutex::new(engine)),
                conversations: Arc::new(Mutex::new(conversations)),
                ollama: Arc::new(ollama),
                sync: Arc::new(Mutex::new(SyncRuntime::default())),
                chat_model,
                fast_model,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::add_memory,
            commands::search_memories,
            commands::list_memories,
            commands::delete_memory,
            commands::stats,
            commands::chat_status,
            commands::chat,
            commands::list_conversations,
            commands::get_conversation,
            commands::create_conversation,
            commands::rename_conversation,
            commands::delete_conversation,
            commands::graph_data,
            commands::ingest_document,
            commands::list_documents,
            commands::delete_document,
            commands::sync_status,
            commands::set_online,
            commands::set_server_url,
            commands::sync_now,
            commands::run_maintenance,
            commands::clear_archive,
            commands::list_scheduled,
            commands::set_task_done,
            commands::list_topics,
            commands::rename_topic,
            commands::merge_topics,
            commands::move_to_topic,
            commands::refresh_topic,
            commands::organize_topics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
